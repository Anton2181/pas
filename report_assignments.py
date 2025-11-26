#!/usr/bin/env python3
"""Summarize assignment quality after solving.

Reads the solver outputs and emits a per-person CSV/console summary that tracks
how many total tasks, repeats, and priority vs non-priority slots each dancer
received, plus whether they were eligible for priority work in the first place.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

PRIORITY_MARKERS = {"YES", "T1", "T2"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a per-person assignment report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--assigned", default="assigned_optimal.csv", type=Path, help="CSV produced by consume_saved_models.py")
    ap.add_argument("--components", default="components_all.csv", type=Path, help="Original components table (for candidate info)")
    ap.add_argument("--out", default=Path("reports") / "assignment_report.csv", type=Path, help="Where to write the per-person CSV report")
    ap.add_argument("--summary", default=Path("reports") / "assignment_report.txt", type=Path, help="Optional plaintext summary (set to '-' to skip)")
    ap.add_argument("--effort-threshold", default=8.0, type=float, help="Effort target used for eligibility/threshold summaries")
    return ap.parse_args()


def load_components(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_component_lookup(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in rows:
        cid = (row.get("ComponentId") or "").strip()
        if not cid:
            continue
        candidate_count = 0
        try:
            candidate_count = int(row.get("Candidate Count", 0))
        except (TypeError, ValueError):
            candidate_count = 0
        lookup[cid] = {
            "candidate_count": candidate_count,
            "manual": (row.get("Assigned?") or "").strip().upper() == "YES",
        }
    return lookup


def normalize_people(value: str) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def parse_effort(value: str | float | int, default: float = 1.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(",", "."))
    except Exception:
        return float(default)


def extract_family(row: Dict[str, str]) -> str:
    key = (row.get("SiblingKey") or "").strip()
    if key:
        parts = [p.strip() for p in key.split("||") if p.strip()]
        if parts:
            return "||".join(parts)
    return (row.get("ComponentId") or "").strip()


def is_priority(row: Dict[str, str]) -> bool:
    marker = (row.get("Priority") or "").strip().upper()
    return marker in PRIORITY_MARKERS


def collect_priority_eligibility(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    eligible_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if not is_priority(row):
            continue
        if (row.get("Assigned?") or "").strip().upper() == "YES":
            # Manual slots already claimed shouldn't count toward eligibility
            continue
        candidates = normalize_people(row.get("Role-Filtered Candidates") or row.get("Candidates") or "")
        for person in candidates:
            eligible_counts[person] += 1
    return eligible_counts


def collect_potential_effort(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    potential: Dict[str, float] = defaultdict(float)
    for row in rows:
        eff = parse_effort(row.get("Total Effort", 1.0), 1.0)
        candidates = normalize_people(row.get("Role-Filtered Candidates") or row.get("Candidates") or "")
        assigned_flag = (row.get("Assigned?") or "").strip().upper() == "YES"
        assigned_to = (row.get("Assigned To") or "").strip()
        if assigned_flag and assigned_to:
            potential[assigned_to] += eff
            continue
        for person in candidates:
            potential[person] += eff
    return potential


def load_assignments(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("Assigned?") or "").strip().upper() == "YES" and row.get("Assigned To")
        ]
    return rows


def collect_assigned_effort(assignments: Iterable[Dict[str, str]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for row in assignments:
        person = (row.get("Assigned To") or "").strip()
        if not person:
            continue
        totals[person] += parse_effort(row.get("Total Effort", 1.0), 1.0)
    return totals


def classify_effort_thresholds(
    potential: Dict[str, float],
    assigned: Dict[str, float],
    threshold: float,
) -> Dict[str, List[str]]:
    eps = 1e-6
    everyone = set(potential) | set(assigned)
    under_capable = sorted(p for p in everyone if potential.get(p, 0.0) + eps < threshold)
    eligible = sorted(p for p in everyone if potential.get(p, 0.0) + eps >= threshold)
    eligible_under = [p for p in eligible if assigned.get(p, 0.0) + eps < threshold]
    eligible_met = [p for p in eligible if assigned.get(p, 0.0) + eps >= threshold]
    return {
        "under_capable": under_capable,
        "eligible_under": eligible_under,
        "eligible_met": eligible_met,
    }


def build_report(
    assignments: List[Dict[str, str]],
    priority_slots: Dict[str, int],
    component_meta: Dict[str, Dict[str, object]],
    assigned_effort: Dict[str, float],
    potential_effort: Dict[str, float],
    effort_threshold: float,
) -> List[Dict[str, str]]:
    by_person: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in assignments:
        person = (row.get("Assigned To") or "").strip()
        if person:
            by_person[person].append(row)

    report: List[Dict[str, str]] = []
    for person in sorted(by_person):
        rows = by_person[person]
        total = len(rows)
        prio = sum(1 for row in rows if is_priority(row))
        non_prio = total - prio
        family_counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            fam = extract_family(row)
            if not fam:
                continue
            meta = component_meta.get((row.get("ComponentId") or "").strip(), {})
            candidate_count = int(meta.get("candidate_count") or 0)
            is_manual = bool(meta.get("manual", False))
            # Do not count repeats that were effectively fixed upfront (manual or only one possible person).
            if is_manual or candidate_count <= 1:
                continue
            family_counts[fam] += 1
        repeats = sum(max(0, count - 1) for count in family_counts.values())
        repeat_families = sorted(fam for fam, count in family_counts.items() if count > 1)
        eligible_priority = priority_slots.get(person, 0)
        assigned_eff = assigned_effort.get(person, 0.0)
        potential_eff = potential_effort.get(person, 0.0)
        effort_shortfall = max(0.0, effort_threshold - assigned_eff)
        report.append(
            {
                "Person": person,
                "TotalTasks": total,
                "PriorityTasks": prio,
                "NonPriorityTasks": non_prio,
                "RepeatAssignments": repeats,
                "RepeatFamilies": " | ".join(repeat_families),
                "EligiblePrioritySlots": eligible_priority,
                "ReceivedPriority": "YES" if prio > 0 else "NO",
                "CouldHavePriority": "YES" if eligible_priority > 0 else "NO",
                "AssignedEffort": f"{assigned_eff:.2f}",
                "PotentialEffort": f"{potential_eff:.2f}",
                "EffortShortfall": f"{effort_shortfall:.2f}",
            }
        )
    return report


def write_report(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("Person,TotalTasks,PriorityTasks,NonPriorityTasks,RepeatAssignments,RepeatFamilies,EligiblePrioritySlots,ReceivedPriority,CouldHavePriority\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    rows: List[Dict[str, str]],
    path: Path,
    effort_threshold: float,
    classifications: Dict[str, List[str]] | None = None,
) -> None:
    if str(path) == "-":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Assignment report"]
    if not rows:
        lines.append("No assignments found.")
    else:
        total = sum(int(row["TotalTasks"]) for row in rows)
        lines.append(f"People with assignments: {len(rows)} (total tasks={total})")
        no_prio = [row["Person"] for row in rows if ["PriorityTasks"] == 0 and row["CouldHavePriority"] == "YES"]
        if no_prio:
            lines.append("People without priority work: " + ", ".join(no_prio))
        repeat_heavy = [row for row in rows if int(row["RepeatAssignments"]) > 0]
        if repeat_heavy:
            lines.append("Repeat-heavy dancers: " + ", ".join(f"{row['Person']} ({row['RepeatAssignments']})" for row in repeat_heavy))

    if classifications is not None:
        under = classifications.get("under_capable", [])
        eligible_under = classifications.get("eligible_under", [])
        eligible_met = classifications.get("eligible_met", [])
        lines.append(
            f"Effort threshold {effort_threshold:.2f}: under-capable={len(under)}"
            + (f" ({', '.join(under)})" if under else "")
        )
        lines.append(
            f"Eligible but below threshold: {len(eligible_under)}"
            + (f" ({', '.join(eligible_under)})" if eligible_under else "")
        )
        lines.append(f"Eligible and met threshold: {len(eligible_met)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    components = load_components(args.components)
    component_meta = build_component_lookup(components)
    priority_slots = collect_priority_eligibility(components)
    potential_effort = collect_potential_effort(components)
    assignments = load_assignments(args.assigned)
    assigned_effort = collect_assigned_effort(assignments)
    classifications = classify_effort_thresholds(potential_effort, assigned_effort, args.effort_threshold)
    rows = build_report(
        assignments,
        priority_slots,
        component_meta,
        assigned_effort,
        potential_effort,
        args.effort_threshold,
    )
    write_report(rows, args.out)
    write_summary(rows, args.summary, args.effort_threshold, classifications)
    print(f"Wrote report to {args.out}")
    if str(args.summary) != "-":
        print(f"Summary saved to {args.summary}")


if __name__ == "__main__":
    main()
