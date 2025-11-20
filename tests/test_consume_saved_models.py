import math

from consume_saved_models import compute_fairness


def test_compute_fairness_respects_manual_loads():
    base_loads = {"ManualHeavy": 2.0}
    comp_info = {"C1": {"taskcount": 1.0, "effort": 1.0}}

    score, loads = compute_fairness([("C1", "AutoPerson")], "count", comp_info, base_loads)

    assert math.isclose(loads["ManualHeavy"], 2.0)
    assert math.isclose(loads["AutoPerson"], 1.0)
    # Max load should reflect the manual assignment, and the imbalance should factor both people.
    assert score == (2.0, 1.0, -1.0)
