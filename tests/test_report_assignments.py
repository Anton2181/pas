from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from tests.utils import COMPONENT_COLUMNS, component_row, write_components

ROOT = Path(__file__).resolve().parents[1]


def test_effort_threshold_summary_in_report(tmp_path: Path) -> None:
    components = [
        component_row(
            cid="C1",
            week="Week 1",
            day="Tuesday",
            task_name="Under Only",
            candidates=["Under"],
            assigned=True,
            assigned_to="Under",
            effort=4.0,
        ),
        component_row(
            cid="C2",
            week="Week 1",
            day="Wednesday",
            task_name="Shared A",
            candidates=["Blair", "Casey"],
            effort=4.0,
        ),
        component_row(
            cid="C3",
            week="Week 1",
            day="Thursday",
            task_name="Shared B",
            candidates=["Blair", "Casey"],
            effort=4.0,
        ),
    ]

    components_path = tmp_path / "components.csv"
    write_components(components_path, components)

    assigned_rows = []
    for row in components:
        assigned_row = dict(row)
        cid = assigned_row["ComponentId"]
        assigned_row["Assigned?"] = "YES"
        if cid == "C1":
            assigned_row["Assigned To"] = "Under"
        else:
            assigned_row["Assigned To"] = "Casey"
        assigned_rows.append(assigned_row)

    assigned_path = tmp_path / "assigned.csv"
    with assigned_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPONENT_COLUMNS)
        writer.writeheader()
        for row in assigned_rows:
            writer.writerow(row)

    summary_path = tmp_path / "summary.txt"
    report_path = tmp_path / "report.csv"

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "report_assignments.py"),
            "--assigned",
            str(assigned_path),
            "--components",
            str(components_path),
            "--summary",
            str(summary_path),
            "--out",
            str(report_path),
            "--effort-threshold",
            "8",
        ],
        cwd=ROOT,
    )

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "under-capable=1" in summary_text and "Under" in summary_text
    assert "Eligible but below threshold: 1" in summary_text and "Blair" in summary_text
    assert "Eligible and met threshold: 1" in summary_text
