from __future__ import annotations

import csv
from pathlib import Path

import export_task_assignments as exporter
from tests.utils import BACKEND_HEADER, COMPONENT_COLUMNS


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def test_banned_manual_preserved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # assigned_optimal.csv (no rows)
    assigned_opt = tmp_path / "assigned_optimal.csv"
    with assigned_opt.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPONENT_COLUMNS)
        writer.writeheader()

    # task_assignment.csv with a header-embedded manual assignee
    header = "Week 10; ; ; Monthly Report; 1; Maciek Oficialski"
    rows = [[""] for _ in range(exporter.HEADER_ROW_1BASED - 1)]
    header_row = [""] * exporter.FIRST_TASK_COL_1B
    header_row[exporter.FIRST_TASK_COL_1B - 1] = header
    rows.append(header_row)

    person_row = ["Maciek Oficialski", "1"]
    # pad up to the expected first task col
    while len(person_row) < exporter.FIRST_TASK_COL_1B:
        person_row.append("")
    rows.append(person_row)
    _write_csv(tmp_path / "task_assignment.csv", rows)

    # backend.csv with the task on the Ban list
    backend_rows = [BACKEND_HEADER]
    banned_row = [""] * len(BACKEND_HEADER)
    banned_row[26 - 1] = "Monthly Report"  # column Z / Ban list
    backend_rows.append(banned_row)
    _write_csv(tmp_path / "backend.csv", backend_rows)

    # optional inputs left absent: varmap.json, decision_log.csv

    exporter.main()

    out_rows = list(csv.DictReader((tmp_path / "task_assignments_output.csv").open(encoding="utf-8")))
    assert out_rows[0]["Task ID"] == header
    assert out_rows[0]["Assignee"] == "Maciek Oficialski"
    assert out_rows[0]["ManualAssignment"] == "YES"
