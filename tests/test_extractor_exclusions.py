from collections import defaultdict

from extractor import Task, exclusion_violation


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
