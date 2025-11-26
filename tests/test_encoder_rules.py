from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import encode_sat_from_components as encoder
from tests.utils import backend_row, component_row, run_encoder_for_rows, write_components


ROOT = Path(__file__).resolve().parents[1]


def _script_path(name: str) -> str:
    candidate = ROOT / name
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Unable to locate {name} relative to repo root {ROOT}")


def _load_varmap(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_resolves_default_paths_from_script_dir(tmp_path: Path, monkeypatch) -> None:
    script_dir = tmp_path / "repo"
    script_dir.mkdir()
    backend = script_dir / "backend.csv"
    backend.write_text("col1\nval\n", encoding="utf-8")

    monkeypatch.setattr(encoder, "SCRIPT_DIR", script_dir)
    monkeypatch.chdir(tmp_path)

    resolved = encoder.resolve_data_path(Path("backend.csv"))
    assert resolved == backend


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


def test_family_registry_includes_tasks(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task A", candidates=["Alex"], sibling_key="Fam"),
        component_row(cid="C2", week="Week 1", day="Wednesday", task_name="Task B", candidates=["Alex"], sibling_key="Fam"),
    ]
    backend = [backend_row("Alex")]

    paths = run_encoder_for_rows(
        tmp_path,
        components=comps,
        backend=backend,
        overrides={"AUTO_SOFTEN": {"ENABLED": False}, "BANNED_SIBLING_PAIRS": [], "BANNED_SAME_DAY_PAIRS": []},
        prefix="registry",
    )

    registry = json.loads(paths["family_registry"].read_text(encoding="utf-8"))
    fams = {entry["canonical"]: entry for entry in registry.get("families", [])}
    assert "Fam" in fams

    comp_map = {c["id"]: set(c.get("tasks", [])) for c in fams["Fam"].get("components", [])}
    assert comp_map == {"C1": {"Task A"}, "C2": {"Task B"}}


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


def test_debug_allow_unassigned_adds_drop_vars(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task A", candidates=["Alex", "Blair"], sibling_key="Fam"),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "DEBUG_ALLOW_UNASSIGNED": True,
        "WEIGHTS": {"W_DEBUG_UNASSIGNED": 999},
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="debug_drop")
    varmap = _load_varmap(paths["map"])

    drop_vars = varmap.get("component_drop_vars")
    assert drop_vars and set(drop_vars.keys()) == {"C1"}
    drop_var = drop_vars["C1"]
    assert varmap["x_to_label"][drop_var] == "drop::C1"

    stats_text = paths["stats"].read_text(encoding="utf-8")
    assert "Allow unassigned components: ON" in stats_text


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


def test_effort_floor_targets_only_eligible(tmp_path: Path) -> None:
    comps = [
        component_row(cid=f"AX{i}", week="Week 1", day="Tuesday", task_name=f"Task {i}", candidates=["Alex", "Blair"], effort=2.0)
        for i in range(1, 9)
    ] + [
        component_row(cid=f"CY{i}", week="Week 2", day="Wednesday", task_name=f"Casey Task {i}", candidates=["Casey"], effort=2.0)
        for i in range(1, 4)
    ]
    backend = [backend_row("Alex"), backend_row("Blair"), backend_row("Casey")]
    overrides = {
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
        "EFFORT_FLOOR_TARGET": 8,
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="effort_floor")
    varmap = _load_varmap(paths["map"])

    eligible = varmap.get("effort_floor_eligible", [])
    assert set(eligible) == {"Alex", "Blair"}

    labels = list(varmap.get("effort_floor_vars", {}).values())
    assert any("person=Alex" in lbl for lbl in labels)
    assert any("person=Blair" in lbl for lbl in labels)
    assert not any("Casey" in lbl for lbl in labels)


def test_effort_floor_hard_ignores_debug_relax(tmp_path: Path) -> None:
    comps = [
        component_row(cid=f"AX{i}", week="Week 1", day="Tuesday", task_name=f"Task {i}", candidates=["Alex", "Blair"], effort=2.0)
        for i in range(1, 9)
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "DEBUG_RELAX": True,
        "EFFORT_FLOOR_HARD": True,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="effort_floor_hard")
    varmap = _load_varmap(paths["map"])

    selectors = varmap.get("selectors", {})
    assert selectors  # baseline debug selectors still populate
    assert not any("effort_floor_hard::Alex" in label for label in selectors)


def test_effort_floor_hard_skips_when_infeasible(tmp_path: Path) -> None:
    comps = [
        component_row(cid="C1", week="Week 1", day="Tuesday", task_name="Task 1", candidates=["Alex", "Blair", "Casey"]),
        component_row(cid="C2", week="Week 1", day="Wednesday", task_name="Task 2", candidates=["Alex", "Blair", "Casey"]),
    ]
    backend = [backend_row("Alex"), backend_row("Blair"), backend_row("Casey")]
    overrides = {
        "EFFORT_FLOOR_TARGET": 1,
        "EFFORT_FLOOR_HARD": True,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="effort_floor_infeasible")
    varmap = _load_varmap(paths["map"])

    assert varmap.get("effort_floor_feasible") is False
    assert varmap.get("effort_floor_hard_applied") is False
    assert varmap.get("effort_floor_notes", {}).get("reason") == "insufficient_global_effort"


def test_effort_floor_hard_skips_when_slots_too_few(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Big Task",
            effort=5,
            candidates=["Alex", "Blair", "Casey"],
        ),
    ]
    backend = [backend_row("Alex"), backend_row("Blair"), backend_row("Casey")]
    overrides = {
        "EFFORT_FLOOR_TARGET": 1,
        "EFFORT_FLOOR_HARD": True,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="effort_floor_slot")
    varmap = _load_varmap(paths["map"])

    assert varmap.get("effort_floor_feasible") is False
    assert varmap.get("effort_floor_hard_applied") is False
    assert varmap.get("effort_floor_notes", {}).get("reason") == "insufficient_slot_capacity"


def test_effort_floor_excludes_people_who_cannot_clear_family_blocks(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="AX1",
            week="Week 1",
            day="Tuesday",
            task_name="Fam A slot 1",
            sibling_key="FAM_A",
            candidates=["Alex"],
            effort=2.0,
        ),
        component_row(
            cid="AX2",
            week="Week 1",
            day="Thursday",
            task_name="Fam A slot 2",
            sibling_key="FAM_A",
            candidates=["Alex"],
            effort=2.0,
        ),
        component_row(
            cid="BX1",
            week="Week 1",
            day="Friday",
            task_name="Blair slot",
            candidates=["Blair"],
            effort=4.0,
        ),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "EFFORT_FLOOR_TARGET": 3,
        "EFFORT_FLOOR_HARD": True,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="effort_floor_family")
    varmap = _load_varmap(paths["map"])

    assert varmap.get("effort_floor_attainable", {}).get("Alex") == 2
    assert varmap.get("effort_floor_eligible", []) == ["Blair"]
    notes = varmap.get("effort_floor_notes", {})
    assert notes.get("ineligible_by_attainable", {}).get("Alex") == 2
    assert varmap.get("effort_floor_feasible") is True
    assert varmap.get("effort_floor_hard_applied") is True


def test_priority_cooldown_hard_ignores_debug_relax(tmp_path: Path) -> None:
    comps = [
        component_row(
            cid="W1P", week="Week 1", day="Tuesday", task_name="Week1 Priority", candidates=["Alex", "Blair"], priority=True
        ),
        component_row(
            cid="W2P", week="Week 2", day="Tuesday", task_name="Week2 Priority", candidates=["Alex", "Blair"], priority=True
        ),
    ]
    backend = [backend_row("Alex"), backend_row("Blair")]
    overrides = {
        "DEBUG_RELAX": True,
        "PRIORITY_COOLDOWN_HARD": True,
        "AUTO_SOFTEN": {"ENABLED": False},
        "BANNED_SIBLING_PAIRS": [],
        "BANNED_SAME_DAY_PAIRS": [],
    }

    paths = run_encoder_for_rows(tmp_path, components=comps, backend=backend, overrides=overrides, prefix="cooldown_hard")
    selectors = _load_varmap(paths["map"]).get("selectors", {})

    assert selectors  # debug relax still captures other constraints
    assert not any("cooldown_prev_hard_PRI" in label for label in selectors)


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
    # Single-candidate or manually fixed slots should not count toward repeat tallies.
    assert int(alex["RepeatAssignments"]) == 0
    assert alex["RepeatFamilies"] == ""
    assert alex["ReceivedPriority"] == "YES"
    assert blair["ReceivedPriority"] == "NO"
    assert blair["CouldHavePriority"] == "YES"
    assert summary_path.exists()
