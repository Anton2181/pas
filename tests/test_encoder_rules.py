from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import backend_row, component_row, run_encoder_for_rows, write_components


ROOT = Path(__file__).resolve().parents[1]


def _script_path(name: str) -> str:
    candidate = ROOT / name
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Unable to locate {name} relative to repo root {ROOT}")


def _load_varmap(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_day_rule_modes(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task Tue", candidates=["Alex", "Blair"], sibling_key="Fam"),
        component_row(cid="C2", week="Week 1", day="Sunday", task_name="Task Sun", candidates=["Alex", "Blair"], sibling_key="Fam"),
        component_row(cid="C3", week="Week 1", day="Wednesday", task_name="Task Wed", candidates=["Alex", "Blair"], sibling_key="Fam"),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]

    base_overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths_soft = run_encoder_for_rows(
        tmp_path,
        components=comps,
        backend=backend,
        overrides={**base_overrides, "SUNDAY_TWO_DAY_SOFT": True, "TWO_DAY_SOFT_ALL": True},
        prefix="soft_all",
    )
    varmap_soft = _load_varmap(paths_soft["map"])
    assert len(varmap_soft["two_day_soft_vars"]) == 6
    labels_soft = list(varmap_soft["two_day_soft_vars"].values())
    assert any("sunday_two_day_soft" in label for label in labels_soft)
    assert any("two_day_soft" in label and "sunday" not in label for label in labels_soft)
    stats_soft = paths_soft["stats"].read_text(encoding="utf-8")
    assert "Two-day soft (both AUTO) counted pairs: 6" in stats_soft

    paths_sun_only = run_encoder_for_rows(
        tmp_path,
        components=comps,
        backend=backend,
        overrides={**base_overrides, "SUNDAY_TWO_DAY_SOFT": True, "TWO_DAY_SOFT_ALL": False},
        prefix="soft_sun",
    )
    varmap_sun = _load_varmap(paths_sun_only["map"])
    assert len(varmap_sun["two_day_soft_vars"]) == 4
    for label in varmap_sun["two_day_soft_vars"].values():
        assert "sunday_two_day_soft" in label
    stats_sun = paths_sun_only["stats"].read_text(encoding="utf-8")
    assert "Two-day soft (both AUTO) counted pairs: 4" in stats_sun

    paths_hard = run_encoder_for_rows(
        tmp_path,
        components=comps,
        backend=backend,
        overrides={**base_overrides, "SUNDAY_TWO_DAY_SOFT": False, "TWO_DAY_SOFT_ALL": False},
        prefix="hard",
    )
    varmap_hard = _load_varmap(paths_hard["map"])
    assert len(varmap_hard["two_day_soft_vars"]) == 0
    stats_hard = paths_hard["stats"].read_text(encoding="utf-8")
    assert "Two-day soft (both AUTO) counted pairs: 0" in stats_hard


def test_auto_soften_marks_scarce_families(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Fam Task", candidates=["Solo"], sibling_key="Fam"),
        component_row(cid="C2", week="Week 2", day="Wednesday", task_name="Fam Task", candidates=["Solo"], sibling_key="Fam"),
    ]
    backend = [backend_row("Solo")]

    overrides = {
        "AUTO_SOFTEN": {
            "ENABLED": True,
            "MIN_UNIQUE_CANDIDATES": 2,
            "MAX_SLOTS_PER_PERSON": 1.1,
            "RELAX_COOLDOWN": True,
            "RELAX_REPEAT": True,
        },
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="soften")
    varmap = _load_varmap(paths["map"])
    assert "Fam" in varmap["auto_soften_families"]


def test_repeat_penalty_skips_manual_only(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Fam",
            candidates=["Alex"],
            sibling_key="Fam",
            assigned=True,
            assigned_to="Alex",
        ),
        component_row(
            cid="C2",
            week="Week 2",
            day="Tuesday",
            task_name="Fam",
            candidates=["Alex"],
            sibling_key="Fam",
            assigned=True,
            assigned_to="Alex",
        ),
    ]
    backend = [backend_row("Alex")]
    overrides = {"AUTO_SOFTEN": {"ENABLED": False}, "BANNED_SIBLING_PAIRS": [], "BANNED_SAME_DAY_PAIRS": []}
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="repeat")
    varmap = _load_varmap(paths["map"])
    assert varmap["repeat_limit_non_vars"] == {}


@pytest.mark.parametrize("mode", ["global", "family"])
def test_priority_coverage_modes(tmp_path: Path, mode: str) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Top Task", candidates=["Alex"], priority=True),
        component_row(cid="C2", week="Week 1", day="Wednesday", task_name="Second Task", candidates=["Blair"], priority=False),
    ]
    backend = [
        backend_row("Alex", top_task="Top Task"),
        backend_row("Blair", second_task="Second Task"),
    ]
    overrides = {
        "PRIORITY_COVERAGE_MODE": mode,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix=f"prio_{mode}")
    varmap = _load_varmap(paths["map"])
    labels = list(varmap["priority_coverage_vars_top"].values())
    assert labels, "expected coverage selectors"
    token = "FAMILY" if mode == "family" else "GLOBAL"
    assert all(token in label for label in labels)


def test_priority_miss_guard_records_people(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Top Task", candidates=["Alex", "Blair"], priority=True)
    ]
    backend = [backend_row("Alex", top_task="Top Task"), backend_row("Blair")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="prio_guard")
    varmap = _load_varmap(paths["map"])
    required = varmap.get("priority_required_vars", {})
    assert required, "expected priority miss guard selectors"
    assert any("person=Alex" in label for label in required.values())


def test_fairness_availability_scaling(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task 1", candidates=["Alex", "Blair"]),
        component_row(cid="C2", week="Week 1", day="Thursday", task_name="Task 2", candidates=["Alex"]),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
        "WEIGHTS": {
            "FAIRNESS_AVAILABILITY": {
                "ENABLED": True,
                "REFERENCE": "auto",
                "MIN_RATIO": 0.1,
                "MAX_RATIO": 2.0,
                "POWER": 1.0,
            }
        },
    }
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="fair")
    varmap = _load_varmap(paths["map"])
    targets = varmap["fairness_targets"]
    assert targets["Alex"] > targets["Blair"]
    fairness_info = varmap["fairness_availability"]
    assert fairness_info["Alex"]["raw_slots"] > fairness_info["Blair"]["raw_slots"]


def test_pipeline_produces_assignments(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Task A",
            candidates=["Alex"],
            assigned=True,
            assigned_to="Alex",
        ),
        component_row(
            cid="C2",
            week="Week 1",
            day="Wednesday",
            task_name="Task B",
            candidates=["Blair"],
            assigned=True,
            assigned_to="Blair",
        ),
        component_row(
            cid="C3",
            week="Week 2",
            day="Sunday",
            task_name="Task C",
            candidates=["Casey"],
            assigned=True,
            assigned_to="Casey",
        ),
    ]
    backend = [backend_row("Alex"), backend_row("Blair"), backend_row("Casey")]
    overrides = {"AUTO_SOFTEN": {"ENABLED": False}, "BANNED_SIBLING_PAIRS": [], "BANNED_SAME_DAY_PAIRS": []}
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="pipeline")

    log_path = tmp_path / "solver.log"
    models_path = tmp_path / "models.txt"
    assigned_path = tmp_path / "assigned.csv"
    models_summary = tmp_path / "models_summary.csv"
    loads_path = tmp_path / "loads.csv"
    penalties_path = tmp_path / "penalties.csv"
    cooldown_path = tmp_path / "cooldown.csv"
    bars_path = tmp_path / "bars.png"
    lorenz_path = tmp_path / "lorenz.png"

    cmd = [
        sys.executable,
        _script_path("run_solver.py"),
        "--opb",
        str(paths["schedule"]),
        "--log",
        str(log_path),
        "--models-out",
        str(models_path),
        "--varmap",
        str(paths["map"]),
        "--components",
        str(paths["components"]),
        "--assigned-out",
        str(assigned_path),
        "--models-summary",
        str(models_summary),
        "--loads-out",
        str(loads_path),
        "--penalties-out",
        str(penalties_path),
        "--cooldown-debug-out",
        str(cooldown_path),
        "--plots-bars",
        str(bars_path),
        "--plots-lorenz",
        str(lorenz_path),
        "--timeout",
        "30",
    ]
    result = subprocess.run(cmd, check=False)
    assert result.returncode in (0, 20, 30, 124)
    assert assigned_path.exists()
    rows = list(csv.DictReader(assigned_path.open(encoding="utf-8")))
    assert len(rows) == 3


def test_visualization_script_creates_graph(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task A", candidates=["Alex"], sibling_key="Fam"),
        component_row(cid="C2", week="Week 1", day="Tuesday", task_name="Task B", candidates=["Alex"], sibling_key="Fam"),
    ]
    backend = [backend_row("Alex", exclusion=("Task A", "Task B"))]
    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides={"AUTO_SOFTEN": {"ENABLED": False}}, prefix="viz")
    graph_dir = tmp_path / "graphs"
    graph_prefix = "components_graph"
    cmd = [
        sys.executable,
        _script_path("visualize_components.py"),
        "--components",
        str(paths["components"]),
        "--backend",
        str(paths["backend"]),
        "--out-dir",
        str(graph_dir),
        "--out-prefix",
        graph_prefix,
        "--analysis-dir",
        str(tmp_path / "analysis"),
        "--layouts",
        "grid",
        "calendar",
    ]
    subprocess.run(cmd, check=True)
    grid_path = graph_dir / f"{graph_prefix}_grid.png"
    calendar_path = graph_dir / f"{graph_prefix}_calendar.png"
    assert grid_path.exists() and grid_path.stat().st_size > 0
    assert calendar_path.exists() and calendar_path.stat().st_size > 0
    hist_path = tmp_path / "analysis" / f"{graph_prefix}_candidate_hist.png"
    heatmap_path = tmp_path / "analysis" / f"{graph_prefix}_week_day_heatmap.png"
    scatter_path = tmp_path / "analysis" / f"{graph_prefix}_degree_scatter.png"
    for chart in (hist_path, heatmap_path, scatter_path):
        assert chart.exists() and chart.stat().st_size > 0


def test_assignment_report(tmp_path: Path) -> None:
    components = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Priority Slot",
            candidates=["Alex", "Blair"],
            sibling_key="Fam",
            priority=True,
        ),
        component_row(
            cid="C2",
            week="Week 1",
            day="Wednesday",
            task_name="Repeat Slot",
            candidates=["Alex"],
            sibling_key="Fam",
        ),
        component_row(
            cid="C3",
            week="Week 1",
            day="Thursday",
            task_name="Solo Slot",
            candidates=["Blair"],
            sibling_key="Solo",
        ),
    ]
    components_path = tmp_path / "components.csv"
    write_components(components_path, components)
    assigned_rows = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Priority Slot",
            candidates=["Alex", "Blair"],
            sibling_key="Fam",
            priority=True,
            assigned=True,
            assigned_to="Alex",
        ),
        component_row(
            cid="C2",
            week="Week 1",
            day="Wednesday",
            task_name="Repeat Slot",
            candidates=["Alex"],
            sibling_key="Fam",
            assigned=True,
            assigned_to="Alex",
        ),
        component_row(
            cid="C3",
            week="Week 1",
            day="Thursday",
            task_name="Solo Slot",
            candidates=["Blair"],
            sibling_key="Solo",
            assigned=True,
            assigned_to="Blair",
        ),
    ]
    assigned_path = tmp_path / "assigned.csv"
    write_components(assigned_path, assigned_rows)
    report_path = tmp_path / "report.csv"
    summary_path = tmp_path / "summary.txt"
    cmd = [
        sys.executable,
        _script_path("report_assignments.py"),
        "--assigned",
        str(assigned_path),
        "--components",
        str(components_path),
        "--out",
        str(report_path),
        "--summary",
        str(summary_path),
    ]
    subprocess.run(cmd, check=True)
    rows = list(csv.DictReader(report_path.open(encoding="utf-8")))
    assert len(rows) == 2
    alex = next(row for row in rows if row["Person"] == "Alex")
    blair = next(row for row in rows if row["Person"] == "Blair")
    assert int(alex["RepeatAssignments"]) == 1
    assert alex["RepeatFamilies"] == "Fam"
    assert alex["ReceivedPriority"] == "YES"
    assert blair["ReceivedPriority"] == "NO"
    assert blair["CouldHavePriority"] == "YES"
    assert summary_path.exists()
