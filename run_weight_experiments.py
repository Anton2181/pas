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
    ap.add_argument("--family-registry", type=Path, default=Path("family_registry.csv"))
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
    if model:
        for key in PENALTY_FIELDS:
            if key in model:
                penalties[key] = model.get(key, "0")
        if model.get("objective"):
            try:
                objective = float(model["objective"])
            except ValueError:
                objective = None
    return ExperimentResult(plan.name, proc.returncode, objective, penalties)


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

    print("name,returncode,objective," + ",".join(PENALTY_FIELDS))
    for res in sorted(results, key=lambda r: r.name):
        penalty_values = [res.penalties.get(key, "") for key in PENALTY_FIELDS]
        objective = "" if res.objective is None else res.objective
        row = [res.name, str(res.returncode), str(objective)] + penalty_values
        print(",".join(row))
        if res.note:
            print(f"# {res.name}: {res.note}")


if __name__ == "__main__":
    main()
