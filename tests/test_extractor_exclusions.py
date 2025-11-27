from collections import defaultdict

from extractor import Task, DecisionLogger, apply_header_assignments, exclusion_violation


def test_single_candidate_overrides_exclusions() -> None:
    day_assign = defaultdict(lambda: defaultdict(set))
    excl_map = {"Task B": {"Task A"}, "Task A": {"Task B"}}

    # Existing assignment on the same day to trigger the exclusion edge case
    existing = Task(0, "Week 13;Tuesday;22-00;Task A;1;Natalia", 0, 1.0)
    existing.available = [0]
    day_assign[f"{existing.week}|{existing.day}"][0].add(existing.name)

    # Single-candidate pool should bypass exclusions
    solo = Task(1, "Week 13;Tuesday;22-00;Task B;1;Natalia", 1, 1.0)
    solo.available = [0]
    assert exclusion_violation(day_assign, excl_map, solo, 0) is False

    # Multi-candidate pools still honor exclusions
    multi = Task(2, "Week 13;Tuesday;22-00;Task B;1;Natalia", 2, 1.0)
    multi.available = [0, 1]
    assert exclusion_violation(day_assign, excl_map, multi, 0) is True


def test_header_assignment_respects_pool_membership() -> None:
    # Header assignee not in the availability pool should be skipped.
    members = ["Antoni", "Maciek", "Ephraim"]
    t = Task(0, "Week 10;;22-00;Monthly Report;1;Ephraim", 0, 1.0)
    t.available = [0, 1]  # Ephraim (idx=2) not in pool
    log = DecisionLogger()

    apply_header_assignments([t], members, lambda *_: False, lambda *_: None, log)

    assert t.assigned_to is None
    assert any(r["Status"] == "Header assignee not in pool" for r in log.rows)
