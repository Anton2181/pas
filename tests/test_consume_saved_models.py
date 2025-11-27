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
        "WEIGHTS": {
            "W_DEBUG_UNASSIGNED_PRIORITY": 42,
            "W_DEBUG_UNASSIGNED_NON_PRIORITY": 42,
        },
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
        "WEIGHTS": {
            "W_DEBUG_UNASSIGNED_PRIORITY": 42,
            "W_DEBUG_UNASSIGNED_NON_PRIORITY": 42,
        },
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_manual_dbg"
    )
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    # Manual components do not emit debug drop variables when DEBUG_ALLOW_UNASSIGNED is enabled.
    assert varmap.get("component_drop_vars", {}) == {}

    # Use the lone assignment variable to drive consumer output without any drop indicators.
    assignment_var = next(iter(varmap["x_to_label"].keys()))
    models_txt = tmp_path / "models.txt"
    models_txt.write_text(f"v {assignment_var}\n", encoding="utf-8")

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


def test_debug_relax_weights_are_recorded(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C3",
            week="Week 1",
            day="Tuesday",
            task_name="Task Relax",
            candidates=["Alex"],
        )
    ]
    backend = [backend_row("Alex", top_task="Conducting the lesson - Wednesday")]
    overrides = {
        "DEBUG_RELAX": True,
        "W_HARD": 123,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_relax_weight"
    )
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    relax_vars = varmap.get("selectors_by_var", {})
    assert relax_vars, "Expected debug relax selectors to be present"
    relax_var = next(iter(relax_vars.keys()))

    models_txt = tmp_path / "models_relax_weight.txt"
    models_txt.write_text(f"v {relax_var}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties_relax_weight.csv"
    models_out = tmp_path / "models_summary_relax_weight.csv"
    assigned_out = tmp_path / "assigned_relax_weight.csv"
    loads_out = tmp_path / "loads_relax_weight.csv"
    bars_out = tmp_path / "bars_relax_weight.png"
    lorenz_out = tmp_path / "lorenz_relax_weight.png"

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
    assert any(r["Category"] == "DebugRelax" and r["Var"] == relax_var and r["Weight"] == "123" for r in rows)

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))
    assert models_rows and models_rows[0].get("n_DebugRelax") == "1"
    assert "DebugRelax" in (models_rows[0].get("penalties") or "")


def test_unknown_penalties_surface_with_weights(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="CU1",
            week="Week 1",
            day="Tuesday",
            task_name="Task Unknown",
            candidates=["Alex"],
        )
    ]
    backend = [backend_row("Alex", top_task="Conducting the lesson - Wednesday")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_unknown_weight"
    )
    vm = json.loads(paths["map"].read_text(encoding="utf-8"))
    mystery_var = "x9999"
    vm.setdefault("penalty_weights", {})[mystery_var] = 777
    vm.setdefault("x_to_label", {})[mystery_var] = "mystery::flag"
    paths["map"].write_text(json.dumps(vm), encoding="utf-8")

    models_txt = tmp_path / "models.txt"
    models_txt.write_text(f"v {mystery_var}\n", encoding="utf-8")

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
    assert rows and rows[0]["Category"] == "UnknownPenalty"
    assert rows[0]["Var"] == mystery_var
    assert rows[0]["Weight"] == "777"
    assert rows[0]["Label"] == "mystery::flag"

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))
    assert models_rows and models_rows[0].get("n_UnknownPenalty") == "1"


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


def test_both_fallback_counts_are_reported(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Task Both",
            candidates=["Alex", "Blair"],
            both_candidates=["Alex"],
        )
    ]
    comps[0]["Role-Filtered Candidates"] = "Blair"

    backend = [backend_row("Alex", both=True), backend_row("Blair")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="consume_both"
    )
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))
    x_alex = next(k for k, v in varmap["x_to_label"].items() if v == "x::C1::Alex")

    models_txt = tmp_path / "models_both.txt"
    models_txt.write_text(f"v {x_alex}\n", encoding="utf-8")

    penalties_out = tmp_path / "penalties_both.csv"
    models_out = tmp_path / "models_summary_both.csv"
    assigned_out = tmp_path / "assigned_both.csv"
    loads_out = tmp_path / "loads_both.csv"
    bars_out = tmp_path / "bars_both.png"
    lorenz_out = tmp_path / "lorenz_both.png"

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

    both_total = len(varmap.get("both_fallback_vars", {}))
    assert both_total >= 1

    with models_out.open("r", encoding="utf-8") as fh:
        models_rows = list(csv.DictReader(fh))

    assert models_rows[0].get("n_BothFallbackTotal") == str(both_total)
    assert models_rows[0].get("n_BothFallback") == "1"

    with penalties_out.open("r", encoding="utf-8") as fh:
        penalties_rows = list(csv.DictReader(fh))

    assert any(r["Category"] == "BothFallback" for r in penalties_rows)


def test_repeat_over_geo_penalizes_each_overage_step(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid=f"C{i}",
            week=f"Week {i}",
            day="Wednesday",
            task_name="Conducting the lesson - Wednesday",
            candidates=["Alex"],
            sibling_key="Conducting the lesson - Wednesday",
            priority=True,
        )
        for i in range(1, 4)
    ]
    backend = [backend_row("Alex", top_task="Conducting the lesson - Wednesday")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
        "REPEAT_LIMIT": {"PRI": 1, "NON": 1},
        "REPEAT_LIMIT_HARD": {"PRI": False, "NON": False},
    }

    paths = run_encoder_for_rows(
        tmp_path, components=comps, backend=backend, overrides=overrides, prefix="repeat_over"
    )
    varmap = json.loads(paths["map"].read_text(encoding="utf-8"))

    x_vars = [
        next(k for k, v in varmap["x_to_label"].items() if v == f"x::{cid}::Alex")
        for cid in ["C1", "C2", "C3"]
    ]
    repeat_vars = list(varmap.get("repeat_limit_pri_vars", {}).keys())

    models_txt = tmp_path / "models_repeat.txt"
    models_txt.write_text("v " + " ".join(x_vars + repeat_vars) + "\n", encoding="utf-8")

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

    rows = list(csv.DictReader(penalties_out.open("r", encoding="utf-8")))
    repeat_rows = [r for r in rows if r["Label"].startswith("repeat_over_geo::PRI::person=Alex")]
    labels = {r["Label"] for r in repeat_rows}

    assert any("::t=2::limit=1" in label for label in labels)
    assert any("::t=3::limit=1" in label for label in labels)

