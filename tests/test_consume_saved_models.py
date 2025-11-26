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


def test_manual_assignments_not_counted_as_unassigned(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C2",
            week="Week 1",
            day="Wednesday",
            task_name="Task Manual",
            candidates=["Alex", "Blair"],
            assigned=True,
            assigned_to="Alex",
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
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_manual_dbg"
    )
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

    if penalties_out.exists():
        with penalties_out.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert not any(r["Category"] == "DebugUnassigned" for r in rows)
    else:
        # No penalties at all → no file emitted
        assert not penalties_out.exists()

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))
    assert models_rows and models_rows[0].get("n_DebugUnassigned") == "0"
    assert "DebugUnassigned" not in (models_rows[0].get("penalties") or "")


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


def test_penalties_include_weights(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="R1",
            week="Week 1",
            day="Tuesday",
            task_name="Repeat A",
            candidates=["Alex", "Blair"],
            sibling_key="FamRepeat",
            priority=True,
        ),
        component_row(
            cid="R2",
            week="Week 2",
            day="Tuesday",
            task_name="Repeat B",
            candidates=["Alex", "Blair"],
            sibling_key="FamRepeat",
            priority=True,
        ),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "REPEAT_OVER_GEO": 3,
        "REPEAT_LIMIT": {"PRI": 1, "NON": 1},
        "WEIGHTS": {"W1_REPEAT": 100},
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_weight")
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))

    repeat_vars = varmap.get("repeat_limit_pri_vars", {}) or varmap.get("repeat_limit_non_vars", {})
    assert repeat_vars
    target_var, target_label = next(iter(repeat_vars.items()))
    penalty_weights = varmap.get("penalty_weights", {})

    models_txt = tmp_path / "models_repeat.txt"
    models_txt.write_text(f"v {target_var}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties_repeat.csv"
    models_out = tmp_path / "models_summary_repeat.csv"
    assigned_out = tmp_path / "assigned_repeat.csv"
    loads_out = tmp_path / "loads_repeat.csv"
    bars_out = tmp_path / "bars_repeat.png"
    lorenz_out = tmp_path / "lorenz_repeat.png"

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
    row = next(r for r in rows if r["Var"] == target_var)

    over_t = int(row.get("OverT") or 0)
    over_limit = int(row.get("OverLimit") or 0)
    over_count = over_t - over_limit

    cfg = varmap["config"]
    is_priority = "::PRI::" in target_label
    base_weight = cfg["WEIGHTS"]["W1_REPEAT" if is_priority else "W2_REPEAT"]
    expected_weight = int(base_weight) * (int(cfg["REPEAT_OVER_GEO"]) ** (over_count - 1))

    assert row.get("Weight") == str(expected_weight)
    assert penalty_weights.get(target_var) == expected_weight

