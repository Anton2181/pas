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


def test_direct_components_respects_penalty_people() -> None:
    comp_meta = {
        "C1": {"Week": "Week 10", "Day": "Monday"},
        "C2": {"Week": "Week 11", "Day": "Tuesday"},
    }
    assignment_lookup = {"C1": "Alice", "C2": "Bob"}
    # Penalty targets Bob; even though both components appear in support, only
    # Bob's assignment should count as directly penalized.
    direct = consumer.select_direct_components_for_penalty(
        ["C1", "C2"],
        "repeat_over_geo::NON::person=Bob::family=F::t=2::limit=1",
        comp_meta,
        assignment_lookup,
    )
    assert direct == ["C2"]


def test_filter_label_by_people_discards_other_penalties() -> None:
    label = (
        "repeat_over_geo::NON::person=Alice::family=F::t=2::limit=1; "
        "cooldown_geo::NON::person=Bob::family=G::t=1::3; "
        "cooldown_geo::NON::person=Alice::family=F::t=1::3"
    )

    kept = consumer.filter_label_by_people(label, {"Alice"})
    assert "Bob" not in kept
    assert "Alice" in kept
    assert kept.count("person=Alice") == 2

