#!/usr/bin/env python3
"""Run multiple encoder/solver experiments with varied weights in parallel."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional

import encode_sat_from_components as encoder
from summarize_results import choose_best_model, _load_csv


@dataclass
class Experiment:
    name: str
    overrides: Dict
    scales: Dict[str, float]

    def to_overrides(self, base_cfg: dict) -> dict:
        """Apply weight scales to a copy of ``base_cfg`` and return overrides."""
        cfg = copy.deepcopy(base_cfg)
        weights = cfg.setdefault("WEIGHTS", {})
        for key, factor in (self.scales or {}).items():
            if key not in weights:
                raise KeyError(f"Unknown weight '{key}' in experiment '{self.name}'")
            weights[key] = weights[key] * factor
        encoder.deep_update(cfg, self.overrides or {})
        return cfg


@dataclass
class ExperimentResult:
    name: str
    returncode: int
    objective: Optional[float]
    penalties: Dict[str, str]
    total_penalties: Optional[int] = None
    load_std: Optional[float] = None
    load_range: Optional[float] = None
    note: str = ""


PENALTY_FIELDS = [
    "n_CooldownPRI",
    "n_CooldownNON",
    "n_CooldownGeoPRI",
    "n_CooldownGeoNON",
    "n_CooldownStreak",
    "n_RepeatOverPRI",
    "n_RepeatOverNON",
    "n_BothFallback",
    "n_PreferredMiss",
    "n_OneTaskDay",
    "n_TwoDaySoft",
    "n_DeprioritizedPair",
    "n_PriorityMiss",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run weight experiments in parallel")
    ap.add_argument("--plans", type=Path, required=True,
                    help="JSON array of experiments with 'name', optional 'overrides' and 'scales'")
    ap.add_argument("--components", type=Path, default=Path("components_all.csv"))
    ap.add_argument("--backend", type=Path, default=Path("backend.csv"))
    ap.add_argument("--family-registry", type=Path, default=Path("family_registry.json"))
    ap.add_argument("--base-config", type=Path,
                    help="Optional JSON overrides applied before per-experiment scales/overrides")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--interrupt-grace", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--metric", choices=["count", "taskcount", "effort"], default="effort")
    return ap.parse_args()


def load_plans(path: Path) -> List[Experiment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    plans: List[Experiment] = []
    for entry in data:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each plan must be an object with a 'name' field")
        plans.append(
            Experiment(
                name=str(entry["name"]),
                overrides=entry.get("overrides") or {},
                scales=entry.get("scales") or {},
            )
        )
    return plans


def best_model_row(models_summary: Path) -> Optional[dict]:
    if not models_summary.exists():
        return None
    rows = _load_csv(models_summary)
    return choose_best_model(rows)


def compute_penalty_totals(model: dict) -> Optional[int]:
    if not model:
        return None
    total = 0
    for key in PENALTY_FIELDS:
        val = model.get(key)
        if val is None or val == "":
            continue
        try:
            total += int(float(val))
        except ValueError:
            continue
    return total


def load_evenness(loads_path: Path) -> tuple[Optional[float], Optional[float]]:
    if not loads_path.exists():
        return None, None
    rows = _load_csv(loads_path)
    totals: List[float] = []
    for row in rows:
        try:
            totals.append(float(row.get("TotalLoad", 0.0)))
        except ValueError:
            continue
    if not totals:
        return None, None
    mean = sum(totals) / len(totals)
    variance = sum((v - mean) ** 2 for v in totals) / len(totals)
    return math.sqrt(variance), max(totals) - min(totals)


def run_single(plan: Experiment, base_cfg: dict, args: argparse.Namespace) -> ExperimentResult:
    workdir = args.out_dir / plan.name.replace(" ", "_")
    workdir.mkdir(parents=True, exist_ok=True)

    cfg_override = plan.to_overrides(base_cfg)

    schedule = workdir / "schedule.opb"
    varmap = workdir / "varmap.json"
    stats = workdir / "stats.txt"

    try:
        encoder.run_encoder(
            components=args.components,
            backend=args.backend,
            out=schedule,
            map_path=varmap,
            stats_path=stats,
            family_registry=args.family_registry,
            overrides=cfg_override,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration use
        return ExperimentResult(plan.name, returncode=1, objective=None, penalties={}, note=str(exc))

    models_out = workdir / "models.txt"
    assigned_out = workdir / "assigned.csv"
    models_summary = workdir / "models_summary.csv"
    loads_out = workdir / "loads.csv"
    penalties_out = workdir / "penalties.csv"
    cooldown_out = workdir / "cooldown.csv"

    solver_script = Path(__file__).with_name("run_solver.py")

    cmd = [
        os.fspath(solver_script),
        "--opb",
        os.fspath(schedule),
        "--log",
        os.fspath(workdir / "solver.log"),
        "--models-out",
        os.fspath(models_out),
        "--varmap",
        os.fspath(varmap),
        "--components",
        os.fspath(args.components),
        "--assigned-out",
        os.fspath(assigned_out),
        "--models-summary",
        os.fspath(models_summary),
        "--loads-out",
        os.fspath(loads_out),
        "--penalties-out",
        os.fspath(penalties_out),
        "--cooldown-debug-out",
        os.fspath(cooldown_out),
        "--plots-bars",
        os.fspath(workdir / "bars.png"),
        "--plots-lorenz",
        os.fspath(workdir / "lorenz.png"),
        "--timeout",
        str(args.timeout),
        "--interrupt-grace",
        str(args.interrupt_grace),
        "--metric",
        args.metric,
    ]

    proc = subprocess.run([os.fspath(Path(sys.executable))] + cmd, check=False)
    if proc.returncode not in (0, 20, 30, 124):  # 20/30/124 = timeout/interrupt codes
        note = f"solver returned {proc.returncode}"
        return ExperimentResult(plan.name, proc.returncode, None, {}, note=note)

    model = best_model_row(models_summary)
    penalties: Dict[str, str] = {}
    objective: Optional[float] = None
    total_penalties: Optional[int] = None
    load_std: Optional[float] = None
    load_range: Optional[float] = None
    if model:
        for key in PENALTY_FIELDS:
            if key in model:
                penalties[key] = model.get(key, "0")
        if model.get("objective"):
            try:
                objective = float(model["objective"])
            except ValueError:
                objective = None
        total_penalties = compute_penalty_totals(model)
        load_std, load_range = load_evenness(loads_out)
    return ExperimentResult(
        plan.name,
        proc.returncode,
        objective,
        penalties,
        total_penalties=total_penalties,
        load_std=load_std,
        load_range=load_range,
    )


def summarize_experiments(results: List[ExperimentResult], out_dir: Path) -> tuple[Optional[ExperimentResult], Optional[ExperimentResult]]:
    best_penalty = None
    for res in results:
        if res.total_penalties is None:
            continue
        if best_penalty is None or res.total_penalties < best_penalty.total_penalties:
            best_penalty = res

    best_even = None
    for res in results:
        if res.load_std is None:
            continue
        if best_even is None or res.load_std < best_even.load_std:
            best_even = res

    try:  # pragma: no cover - optional plotting dependency
        import matplotlib.pyplot as plt

        names = [r.name for r in results if r.total_penalties is not None]
        totals = [r.total_penalties for r in results if r.total_penalties is not None]
        if names and totals:
            plt.figure(figsize=(10, 5))
            plt.bar(names, totals, color="#4c72b0")
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Total penalties")
            plt.title("Penalty counts per experiment")
            plt.tight_layout()
            plt.savefig(out_dir / "penalties_bar.png", dpi=150)
            plt.close()

        names_std = [r.name for r in results if r.load_std is not None]
        stds = [r.load_std for r in results if r.load_std is not None]
        if names_std and stds:
            plt.figure(figsize=(10, 5))
            plt.bar(names_std, stds, color="#55a868")
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Load std dev (TotalLoad)")
            plt.title("Load evenness per experiment")
            plt.tight_layout()
            plt.savefig(out_dir / "load_evenness_bar.png", dpi=150)
            plt.close()
    except ImportError:
        pass

    return best_penalty, best_even


def main() -> None:
    args = parse_args()
    plans = load_plans(args.plans)
    base_cfg = encoder.build_config(json.loads(args.base_config.read_text(encoding="utf-8")) if args.base_config else {})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: List[ExperimentResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_single, plan, base_cfg, args): plan for plan in plans}
        for fut in as_completed(futures):
            results.append(fut.result())

    print("name,returncode,objective,total_penalties,load_std,load_range," + ",".join(PENALTY_FIELDS))
    for res in sorted(results, key=lambda r: r.name):
        penalty_values = [res.penalties.get(key, "") for key in PENALTY_FIELDS]
        objective = "" if res.objective is None else res.objective
        row = [
            res.name,
            str(res.returncode),
            str(objective),
            "" if res.total_penalties is None else str(res.total_penalties),
            "" if res.load_std is None else f"{res.load_std:.6f}",
            "" if res.load_range is None else f"{res.load_range:.6f}",
        ] + penalty_values
        print(",".join(row))
        if res.note:
            print(f"# {res.name}: {res.note}")

    best_penalty, best_even = summarize_experiments(results, args.out_dir)
    if best_penalty:
        print(
            f"# Lowest total penalties: {best_penalty.name} (total={best_penalty.total_penalties})"
        )
    if best_even:
        print(
            f"# Most even load (std dev): {best_even.name} (std={best_even.load_std:.3f}, range={best_even.load_range:.3f})"
        )


if __name__ == "__main__":
    main()
