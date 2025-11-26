import json
from pathlib import Path

import pytest

import encode_sat_from_components as encoder
from run_weight_experiments import (
    PENALTY_FIELDS,
    ExperimentResult,
    best_model_row,
    load_plans,
    summarize_experiments,
)


def test_load_plans_and_scaling(tmp_path: Path) -> None:
    plan_path = tmp_path / "plans.json"
    plan_path.write_text(
        json.dumps(
            [
                {"name": "base"},
                {
                    "name": "scaled",
                    "scales": {"W1_COOLDOWN": 2.0},
                    "overrides": {"WEIGHTS": {"W6_UNDER": 9}},
                },
            ]
        ),
        encoding="utf-8",
    )

    plans = load_plans(plan_path)
    assert [p.name for p in plans] == ["base", "scaled"]

    base_cfg = encoder.build_config()
    scaled = plans[1].to_overrides(base_cfg)
    assert scaled["WEIGHTS"]["W1_COOLDOWN"] == base_cfg["WEIGHTS"]["W1_COOLDOWN"] * 2
    assert scaled["WEIGHTS"]["W6_UNDER"] == 9


def test_weight_ladder_overrides_weights() -> None:
    overrides = {
        "WEIGHT_LADDER": {
            "ENABLED": True,
            "ORDER": ["W5", "W4", "W3"],
            "RATIO": 100,
            "TOP": 1_000_000,
        }
    }

    cfg = encoder.build_config(overrides)
    weights = cfg["WEIGHTS"]

    assert weights["W5"] == 1_000_000
    assert weights["W4"] == 10_000
    assert weights["W3"] == 100
    # Unlisted weights should remain anchored to the defaults
    assert weights["W1_COOLDOWN"] == encoder.DEFAULT_CONFIG["WEIGHTS"]["W1_COOLDOWN"]


def test_weight_ladder_anchors_to_first_weight_when_top_missing() -> None:
    cfg = encoder.build_config({"WEIGHT_LADDER": {"ENABLED": True, "ORDER": ["W5", "W3"]}})
    weights = cfg["WEIGHTS"]

    assert weights["W5"] == 100
    assert weights["W3"] == 1


def test_weight_ladder_requires_ratio_above_one() -> None:
    overrides = {"WEIGHT_LADDER": {"ENABLED": True, "ORDER": ["W5"], "RATIO": 1}}

    with pytest.raises(ValueError):
        encoder.build_config(overrides)


def test_best_model_row_picks_lowest_objective(tmp_path: Path) -> None:
    csv_path = tmp_path / "models_summary.csv"
    csv_path.write_text(
        "idx,objective,n_CooldownPRI\n"
        "1,10,5\n"
        "2,5,3\n",
        encoding="utf-8",
    )
    best = best_model_row(csv_path)
    assert best is not None
    assert best["idx"] == "2"
    assert PENALTY_FIELDS[0] in {"n_CooldownPRI", "n_CooldownNON"}


def test_summarize_experiments_prefers_low_penalties_and_evenness(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    res = [
        ExperimentResult("base", 0, None, {}, total_penalties=10, load_std=2.0, load_range=4.0),
        ExperimentResult("penalty", 0, None, {}, total_penalties=5, load_std=3.0, load_range=5.0),
        ExperimentResult("even", 0, None, {}, total_penalties=7, load_std=1.5, load_range=2.0),
    ]

    best_pen, best_even = summarize_experiments(res, tmp_path)
    assert best_pen and best_pen.name == "penalty"
    assert best_even and best_even.name == "even"

    assert (tmp_path / "penalties_bar.png").exists()
    assert (tmp_path / "load_evenness_bar.png").exists()
