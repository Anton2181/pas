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
        "WEIGHTS": {"W_DEBUG_UNASSIGNED": 42},
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

