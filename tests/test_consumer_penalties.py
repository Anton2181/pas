import consume_saved_models as consumer


def test_select_direct_components_excludes_first_occurrence() -> None:
    comp_meta = {
        "C1": {"Week": "Week 10", "Day": "Monday"},
        "C2": {"Week": "Week 11", "Day": "Sunday"},
        "C3": {"Week": "Week 11", "Day": "Tuesday"},
    }

    # The earliest (Week 10) should be treated as the baseline; later occurrences
    # face the direct penalties.
    direct = consumer.select_direct_components(["C1", "C2", "C3"], comp_meta)
    assert set(direct) == {"C2", "C3"}


def test_select_direct_components_falls_back_when_unordered() -> None:
    comp_meta = {
        "A": {"Week": "", "Day": ""},
        "B": {"Week": None, "Day": None},
    }

    # Without ordering hints, attribute the penalty to all participating components
    # so direct columns remain populated.
    direct = consumer.select_direct_components(["A", "B"], comp_meta)
    assert set(direct) == {"A", "B"}

