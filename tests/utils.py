"""Fixtures and helpers for encoder tests."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import encode_sat_from_components as encoder

COMPONENT_COLUMNS: Sequence[str] = (
    "Kind",
    "ComponentId",
    "Week",
    "Day",
    "Repeat",
    "RepeatMax",
    "Task Count",
    "Names",
    "SiblingKey",
    "Priority",
    "Assigned?",
    "Assigned To",
    "Candidate Count",
    "Candidates",
    "Role-Filtered Candidates",
    "Total Effort",
)

BACKEND_HEADER: Sequence[str] = (
    "",
    "Leader",
    "Follower",
    "Both",
    "",
    "Task",
    "Priority Assignment",
    "",
    "TODO",
    "EFFORT",
    "",
    "IgnoreAvailability",
    "",
    "complimentaryPairs 1",
    "complimentaryPairs 2",
    "DeprioritizedPairs1",
    "DeprioritizedPairs2",
    "exclusionPairs 1",
    "exclusionPairs 2",
    "",
    "Limit ignore",
    "",
    "Preferred Pair 1",
    "Preferred Pair 2",
    "",
    "Ban list",
    "",
    "CooldownEqual1",
    "CooldownEqual2",
    "",
    "Top Priority",
    "Second Priority",
)


def component_row(
    *,
    cid: str,
    week: str,
    day: str,
    task_name: str,
    candidates: Iterable[str],
    sibling_key: str = "",
    assigned: bool = False,
    assigned_to: str | None = None,
    priority: bool = False,
    effort: float = 1.0,
) -> Dict[str, str]:
    """Build a Dict row for ``components_all.csv``."""

    cand_list = [c for c in candidates]
    cand_csv = ",".join(cand_list)
    row = {col: "" for col in COMPONENT_COLUMNS}
    assignee = assigned_to if assigned_to else (cand_list[0] if assigned and cand_list else "")
    row.update(
        {
            "Kind": "Component",
            "ComponentId": cid,
            "Week": week,
            "Day": day,
            "Repeat": "1",
            "RepeatMax": "1",
            "Task Count": "1",
            "Names": task_name,
            "SiblingKey": sibling_key,
            "Priority": "YES" if priority else "NO",
            "Assigned?": "YES" if assigned else "NO",
            "Assigned To": assignee,
            "Candidate Count": str(len(cand_list)),
            "Candidates": cand_csv,
            "Role-Filtered Candidates": cand_csv,
            "Total Effort": f"{effort:.2f}",
        }
    )
    return row


def backend_row(
    name: str,
    *,
    both: bool = False,
    exclusion: tuple[str, str] | None = None,
    deprioritized: tuple[str, str] | None = None,
    preferred: tuple[str, str] | None = None,
    cooldown: tuple[str, str] | None = None,
    top_task: str | None = None,
    second_task: str | None = None,
) -> List[str]:
    row = ["" for _ in BACKEND_HEADER]
    row[0] = name
    row[3] = "1" if both else ""
    if deprioritized:
        row[15], row[16] = deprioritized
    if exclusion:
        row[17], row[18] = exclusion
    if preferred:
        row[22], row[23] = preferred
    if cooldown:
        row[27], row[28] = cooldown
    if top_task:
        row[30] = top_task
    if second_task:
        row[31] = second_task
    return row


def write_components(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPONENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_backend(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BACKEND_HEADER)
        for row in rows:
            writer.writerow(list(row))


def run_encoder_for_rows(
    tmp_path: Path,
    *,
    components: Iterable[Dict[str, str]],
    backend: Iterable[Sequence[str]],
    overrides: dict | None = None,
    prefix: str = "case",
) -> Dict[str, Path]:
    comp_path = tmp_path / f"{prefix}_components.csv"
    backend_path = tmp_path / f"{prefix}_backend.csv"
    write_components(comp_path, components)
    write_backend(backend_path, backend)
    out = tmp_path / f"{prefix}_schedule.opb"
    map_path = tmp_path / f"{prefix}_varmap.json"
    stats_path = tmp_path / f"{prefix}_stats.txt"
    registry_path = tmp_path / f"{prefix}_family_registry.json"

    encoder.run_encoder(
        components=comp_path,
        backend=backend_path,
        out=out,
        map_path=map_path,
        stats_path=stats_path,
        family_registry=registry_path,
        overrides=overrides,
    )
    return {
        "components": comp_path,
        "backend": backend_path,
        "schedule": out,
        "map": map_path,
        "stats": stats_path,
        "family_registry": registry_path,
    }
