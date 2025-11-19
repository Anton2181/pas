import json
from pathlib import Path

import encode_sat_from_components as encoder
from run_weight_experiments import PENALTY_FIELDS, best_model_row, load_plans


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
