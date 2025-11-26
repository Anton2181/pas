from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.utils import backend_row, component_row, run_encoder_for_rows


ROOT = Path(__file__).resolve().parents[1]


def test_debug_unassigned_penalty_counts(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Task A",
            candidates=["Alex", "Blair"],
            sibling_key="Fam",
        )
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "DEBUG_ALLOW_UNASSIGNED": True,
        "WEIGHTS": {"W_DEBUG_UNASSIGNED": 42},
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_dbg")
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    drop_var = next(iter(varmap.get("component_drop_vars", {}).values()))

    models_txt = tmp_path / "models.txt"
    models_txt.write_text(f"v {drop_var}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties.csv"
    models_out = tmp_path / "models_summary.csv"
    assigned_out = tmp_path / "assigned.csv"
    loads_out = tmp_path / "loads.csv"
    bars_out = tmp_path / "bars.png"
    lorenz_out = tmp_path / "lorenz.png"

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "consume_saved_models.py"),
            "--models",
            str(models_txt),
            "--varmap",
            str(paths["map"]),
            "--components",
            str(paths["components"]),
            "--metric",
            "effort",
            "--plots-bars",
            str(bars_out),
            "--plots-lorenz",
            str(lorenz_out),
            "--assigned-out",
            str(assigned_out),
            "--models-out",
            str(models_out),
            "--loads-out",
            str(loads_out),
            "--penalties-out",
            str(penalties_out),
        ],
        cwd=ROOT,
        env=env,
    )

    with penalties_out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["Category"] == "DebugUnassigned" and r["Var"] == drop_var for r in rows)

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))
    assert models_rows and models_rows[0].get("n_DebugUnassigned") == "1"
    assert "DebugUnassigned" in (models_rows[0].get("penalties") or "")


def test_debug_unassigned_ignored_when_assigned(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Task A",
            candidates=["Alex", "Blair"],
            sibling_key="Fam",
        )
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "DEBUG_ALLOW_UNASSIGNED": True,
        "WEIGHTS": {"W_DEBUG_UNASSIGNED": 42},
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(
        tmp_path,
        components=comps,
        backend=backend,
        overrides=overrides,
        prefix="consume_dbg_assigned",
    )
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    drop_var = next(iter(varmap.get("component_drop_vars", {}).values()))
    assign_var = next(v for v, lbl in varmap.get("x_to_label", {}).items() if lbl.endswith("::C1::Alex"))

    models_txt = tmp_path / "models_with_drop_and_assign.txt"
    models_txt.write_text(f"v {assign_var} {drop_var}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties.csv"
    models_out = tmp_path / "models_summary.csv"
    assigned_out = tmp_path / "assigned.csv"
    loads_out = tmp_path / "loads.csv"
    bars_out = tmp_path / "bars.png"
    lorenz_out = tmp_path / "lorenz.png"

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "consume_saved_models.py"),
            "--models",
            str(models_txt),
            "--varmap",
            str(paths["map"]),
            "--components",
            str(paths["components"]),
            "--metric",
            "effort",
            "--plots-bars",
            str(bars_out),
            "--plots-lorenz",
            str(lorenz_out),
            "--assigned-out",
            str(assigned_out),
            "--models-out",
            str(models_out),
            "--loads-out",
            str(loads_out),
            "--penalties-out",
            str(penalties_out),
        ],
        cwd=ROOT,
        env=env,
    )

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))

    assert models_rows and models_rows[0].get("n_DebugUnassigned") == "0"
    assert "DebugUnassigned" not in (models_rows[0].get("penalties") or "")

    with assigned_out.open("r", encoding="utf-8") as fh:
        assigned_rows = list(csv.DictReader(fh))

    assert any(r["ComponentId"] == "C1" and r.get("Assigned?") == "YES" and r.get("Assigned To") == "Alex" for r in assigned_rows)
    assert not penalties_out.exists()


def test_effort_floor_penalty_counts(tmp_path: Path) -> None:
    comps = [
        component_row(cid=f"AX{i}", week="Week 1", day="Tuesday", task_name=f"Task {i}", candidates=["Alex", "Blair"], effort=2.0)
        for i in range(1, 9)
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "WEIGHTS": {"W_EFFORT_FLOOR": 17},
        "EFFORT_FLOOR_TARGET": 1,
        "EFFORT_FLOOR_HARD": False,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_effort_floor")
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    floor_vars = varmap.get("effort_floor_vars", {})
    assert floor_vars
    floor_var = next(iter(floor_vars.keys()))

    models_txt = tmp_path / "models_effort_floor.txt"
    models_txt.write_text(f"v {floor_var}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties_floor.csv"
    models_out = tmp_path / "models_summary_floor.csv"
    assigned_out = tmp_path / "assigned_floor.csv"
    loads_out = tmp_path / "loads_floor.csv"
    bars_out = tmp_path / "bars_floor.png"
    lorenz_out = tmp_path / "lorenz_floor.png"

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "consume_saved_models.py"),
            "--models",
            str(models_txt),
            "--varmap",
            str(paths["map"]),
            "--components",
            str(paths["components"]),
            "--metric",
            "effort",
            "--plots-bars",
            str(bars_out),
            "--plots-lorenz",
            str(lorenz_out),
            "--assigned-out",
            str(assigned_out),
            "--models-out",
            str(models_out),
            "--loads-out",
            str(loads_out),
            "--penalties-out",
            str(penalties_out),
        ],
        cwd=ROOT,
        env=env,
    )

    with penalties_out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["Category"] == "EffortFloor" and r["Var"] == floor_var for r in rows)

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))
    assert models_rows and models_rows[0].get("n_EffortFloor") == "1"
    assert "EffortFloor" in (models_rows[0].get("penalties") or "")

