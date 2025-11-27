#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-task assignment export with fallback + provenance flags.

Inputs (cwd):
  - assigned_optimal.csv      (fully assigned components)
  - task_assignment.csv       (Task Assignments sheet; row 36 has task IDs)
  - backend.csv               (exclusions + cooldown families + IgnoreAvailability in col L)
  - varmap.json               (optional; to know which components are manual)
  - decision_log.csv          (optional; to mark pre-pass moves)

Output:
  - task_assignments_output.csv

Columns added:
  - PrePassMoved        : YES if this task header appears in decision_log phase 'pre-pass'
                          OR if the assignee's name appears in the sibling (other repeat) Task ID tail.
  - ManualAssignment    : YES if the component was marked manual in varmap.json (manual_components)
                          OR if the assignee's name appears in the current Task ID tail.
  - IgnoreAvailability  : YES if task name is listed in backend column L (everyone-eligible)

Other columns kept:
  - Fallback: OneTaskDay (named days only), CooldownPRI, CooldownNON
  - OneDayPerWeek_OK, Availability_OK, Exclusions_OK, PrevWeekCooldown_OK

Notes
  * We canonicalize Task IDs to the first 5 fields (Week; Day; Time; Name; Repeat)
    so comparisons survive pre-pass appending (e.g., "; Person").
"""

from __future__ import annotations
from pathlib import Path
import csv, io, json
from collections import defaultdict
from typing import Dict, List, Tuple, Set

ASSIGNED_OPT_PATH    = Path("assigned_optimal.csv")
TASK_ASSIGNMENT_PATH = Path("task_assignment.csv")
BACKEND_PATH         = Path("backend.csv")
VARMAP_PATH          = Path("varmap.json")
DECISION_LOG_PATH    = Path("decision_log.csv")
OUTPUT_PATH          = Path("task_assignments_output.csv")

HEADER_ROW_1BASED   = 36
NAMES_START_1BASED  = 37
FIRST_TASK_COL_1B   = 2
NAME_COL_1B         = 1

def trim(s: str) -> str:
    return (s or "").strip()

def read_csv_matrix(path: Path) -> List[List[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    rdr = csv.reader(io.StringIO(text), delimiter=',')
    return [list(row) for row in rdr]

# -------- Task header parsing & canonicalization --------

def parse_task_header(header: str) -> Tuple[str,str,str,str,int,str]:
    """
    Robust parser: accepts 'Week; Day; Time; Name; Repeat' and ignores any extra segments (e.g., '; Person').
    Returns: (week, day, time, name, repeat_num, canonical) where canonical is the 5-field string joined with '; '.
    """
    parts_raw = str(header).split(';')
    parts = [trim(x) for x in parts_raw]
    while len(parts) < 5:
        parts.append("")
    week = parts[0]
    day  = parts[1]
    time = parts[2]
    name = parts[3]
    rep_raw = parts[4]
    digits = "".join(ch for ch in rep_raw if ch.isdigit())
    rep_num = int(digits) if digits else 1
    canonical = "; ".join([week, day, time, name, str(rep_num)])
    return week, day, time, name, rep_num, canonical


def header_assignee(header: str) -> str:
    """
    Return the first assignee token embedded after the canonical 5 fields, if any.
    """
    toks = [trim(x) for x in str(header).split(';')]
    if len(toks) <= 5:
        return ""
    for tok in toks[5:]:
        if tok:
            return tok
    return ""

def canonicalize_header(header: str) -> str:
    """Return the first 5 fields canonicalized as 'Week; Day; Time; Name; RepeatNum'."""
    *_, canonical = (*parse_task_header(header),)
    return canonical

def header_has_person(header: str, person: str) -> bool:
    """
    Check if 'person' appears in the tail of Task ID (any fields after the first 5 canonical fields).
    Comparison is exact token match after trimming.
    """
    toks = [trim(x) for x in str(header).split(';')]
    if len(toks) <= 5:
        return False
    tail = toks[5:]
    return any(tok == person for tok in tail)

# -------------------------------------------------------

def week_num(week_label: str) -> int:
    digits = "".join(ch for ch in (week_label or "") if ch.isdigit())
    return int(digits) if digits else 0

def load_task_grid(path: Path):
    mat = read_csv_matrix(path)
    header_idx = HEADER_ROW_1BASED - 1
    people_start_idx = NAMES_START_1BASED - 1
    first_task_col_idx = FIRST_TASK_COL_1B - 1
    name_col_idx = NAME_COL_1B - 1

    header_row = mat[header_idx] if header_idx < len(mat) else []
    task_headers: List[str] = []
    c = first_task_col_idx
    while c < len(header_row):
        h = trim(header_row[c])
        if not h: break
        task_headers.append(h)
        c += 1

    people: List[str] = []
    r = people_start_idx
    while r < len(mat):
        row = mat[r] if r < len(mat) else []
        name = trim(row[name_col_idx] if name_col_idx < len(row) else "")
        if not name: break
        people.append(name)
        r += 1

    avail: List[List[int]] = []
    for row_idx in range(people_start_idx, people_start_idx + len(people)):
        row = mat[row_idx] if row_idx < len(mat) else []
        row_av: List[int] = []
        for col_idx in range(first_task_col_idx, first_task_col_idx + len(task_headers)):
            cell = trim(row[col_idx] if col_idx < len(row) else "")
            v = 1 if cell in ("1","TRUE","true","Yes","YES") else 0
            row_av.append(v)
        avail.append(row_av)

    key_to_header: Dict[Tuple[str,str,str,int], str] = {}
    header_to_col: Dict[str,int] = {}
    # also build group index for sibling lookup: (week, day, time, name) -> {repeat -> header}
    group_to_repeats: Dict[Tuple[str,str,str,str], Dict[int, str]] = defaultdict(dict)

    for j, h in enumerate(task_headers):
        wk, dy, tm, nm, rp, _canon = parse_task_header(h)
        key_to_header[(wk, dy, nm, rp)] = h
        header_to_col[h] = j
        group_to_repeats[(wk, dy, tm, nm)][rp] = h

    return task_headers, people, avail, key_to_header, header_to_col, group_to_repeats

def parse_backend(path: Path):
    mat = read_csv_matrix(path)
    def col(row, idx1b):
        i0 = idx1b - 1
        return trim(row[i0]) if i0 < len(row) else ""

    exclusions = defaultdict(set)                 # taskA -> set(taskB)
    for r in range(1, len(mat)):
        a = col(mat[r], 18); b = col(mat[r], 19)  # R:S
        if a and b:
            exclusions[a].add(b); exclusions[b].add(a)

    # cooldown families AB:AC
    adj = defaultdict(set)
    for r in range(1, len(mat)):
        a = col(mat[r], 28); b = col(mat[r], 29)
        if a and b:
            adj[a].add(b); adj[b].add(a)
    cooldown_key: Dict[str,str] = {}
    seen = set()
    for start in list(adj.keys()):
        if start in seen: continue
        stack = [start]; comp = []
        seen.add(start)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        canon = sorted(comp)[0]
        for name in comp:
            cooldown_key[name] = canon

    # IgnoreAvailability / Everyone-eligible L
    ignore_avail: Set[str] = set()
    for r in range(1, len(mat)):
        t = col(mat[r], 12)  # L
        if t:
            ignore_avail.add(t)

    # Ban list Z
    banned_tasks: Set[str] = set()
    for r in range(1, len(mat)):
        t = col(mat[r], 26)  # Z
        if t:
            banned_tasks.add(t)

    return exclusions, cooldown_key, ignore_avail, banned_tasks

def load_assigned_components(path: Path):
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    comps = []
    for r in rows:
        if (r.get("Kind","") == "Component") and (r.get("Assigned?","").upper() == "YES"):
            comps.append(r)
    return comps

def load_manual_flags(varmap_path: Path) -> Dict[str, bool]:
    if not varmap_path.exists():
        return {}
    try:
        vm = json.loads(varmap_path.read_text(encoding='utf-8'))
        return {k: bool(v) for k, v in vm.get("manual_components", {}).items()}
    except Exception:
        return {}

def load_prepass_headers(decision_log_path: Path) -> Set[str]:
    """Return the set of canonical TaskHeader strings that were touched during 'pre-pass'."""
    touched: Set[str] = set()
    if not decision_log_path.exists():
        return touched
    with decision_log_path.open('r', encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            if trim(r.get("Phase","")).lower() == "pre-pass":
                th_raw = trim(r.get("TaskHeader",""))
                if th_raw:
                    touched.add(canonicalize_header(th_raw))
    return touched

def main():
    # sanity
    for p in [ASSIGNED_OPT_PATH, TASK_ASSIGNMENT_PATH, BACKEND_PATH]:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    (task_headers, people, avail,
     key_to_header, header_to_col, group_to_repeats) = load_task_grid(TASK_ASSIGNMENT_PATH)
    exclusions, cooldown_key, ignore_avail, banned_tasks = parse_backend(BACKEND_PATH)
    comps = load_assigned_components(ASSIGNED_OPT_PATH)
    manual_flags = load_manual_flags(VARMAP_PATH)              # cid -> True if manual (single-candidate)
    prepass_canon = load_prepass_headers(DECISION_LOG_PATH)    # canonical TaskHeaders pre-passed

    # availability lookup keyed by the EXACT header string as in the grid
    person_to_row = {p:i for i,p in enumerate(people)}
    availability = {p: {task_headers[j]: avail[ri][j]
                        for j in range(len(task_headers))}
                    for p, ri in person_to_row.items()}

    # Represent components (from assigned_optimal.csv)
    class Comp:
        __slots__ = ("cid","week","wno","day","time","repeat","names","assignee","priority","manual","families")
    comps_norm: List[Comp] = []
    for r in comps:
        c = Comp()
        c.cid      = trim(r.get("ComponentId",""))
        c.week     = trim(r.get("Week",""))
        c.wno      = week_num(c.week)
        c.day      = trim(r.get("Day",""))
        # time is not in assigned_optimal.csv explicitly; we infer from Names when matching headers
        c.time     = ""  # kept for completeness; not used directly
        try: c.repeat = int(trim(r.get("Repeat","") or "1"))
        except: c.repeat = 1
        c.names    = [trim(x) for x in (r.get("Names","") or "").split("|")]
        c.assignee = trim(r.get("Assigned To",""))
        c.priority = (trim(r.get("Priority","")).upper() == "YES")
        c.manual   = bool(manual_flags.get(c.cid, False))
        fams: Set[str] = set()
        for nm in c.names:
            fams.add(cooldown_key.get(nm, nm))
        c.families = fams
        comps_norm.append(c)

    manual_cids: Set[str] = {c.cid for c in comps_norm if c.manual}

    # index by person/week
    by_person_week: Dict[Tuple[str,int], List[Comp]] = defaultdict(list)
    for c in comps_norm:
        by_person_week[(c.assignee, c.wno)].append(c)

    # determine cooldown fallbacks on CURRENT week comps
    cooldown_tag_for_comp: Dict[str, str] = {}  # cid -> "", "CooldownPRI" or "CooldownNON"
    for (person, w) in list(by_person_week.keys()):
        prev = by_person_week.get((person, w-1), [])
        cur  = by_person_week.get((person, w), [])
        if not prev or not cur:
            continue
        for b in cur:
            tag = ""
            for a in prev:
                if not (b.families & a.families):
                    continue
                # ignore both-manual
                if a.manual and b.manual:
                    continue
                # classify
                if a.priority or b.priority:
                    tag = "CooldownPRI"
                    break
                else:
                    tag = "CooldownNON"
            if tag:
                cooldown_tag_for_comp[b.cid] = tag

    # per-task collection + per-day counts (named days only)
    per_task_rows: List[Tuple[str,str,str,str,str,str,str,int]] = []  # (hdr_exact, hdr_canon, person, taskname, week, day, cid, repeat)
    tasks_on_named_day: Dict[Tuple[str,str,str], int] = defaultdict(int)  # (person, week, day) -> count

    for c in comps_norm:
        for nm in c.names:
            # We don't have Time in assigned_optimal.csv; it's embedded in header.
            # key_to_header keys on (week, day, name, repeat), where 'name' = task name
            hdr_exact = key_to_header.get((c.week, c.day, nm, c.repeat))
            if not hdr_exact:
                continue
            wk, dy, tm, _nm, rp, hdr_canon = parse_task_header(hdr_exact)
            per_task_rows.append((hdr_exact, hdr_canon, c.assignee, nm, c.week, c.day, c.cid, rp))
            if c.day and c.day.lower() != "all":
                tasks_on_named_day[(c.assignee, c.week, c.day)] += 1

    # For quick sibling lookup per task instance:
    # group key = (week, day, time, name) -> {repeat -> header}
    def find_sibling_header(curr_header: str) -> List[str]:
        wk, dy, tm, nm, rp, _canon = parse_task_header(curr_header)
        repeats = group_to_repeats.get((wk, dy, tm, nm), {})
        # return all other repeats' headers (there may be >2 repeats)
        return [h for r,h in repeats.items() if r != rp]

    # compose output
    out: List[Dict[str,str]] = []
    for hdr_exact, hdr_canon, person, nm, week, day, cid, rp in per_task_rows:
        # availability after pre-pass (assignee's availability on the exact column)
        avail_ok = (availability.get(person, {}).get(hdr_exact, 0) == 1)

        # named-day-only checks after pre-pass
        if day and day.lower() != "all":
            # number of distinct named days this week for the person
            num_days = len({d for (p,w,d),cnt in tasks_on_named_day.items() if p==person and w==week})
            one_day_ok = (num_days <= 1)
            one_day_str = "YES" if one_day_ok else "NO"
            # fallback: OneTaskDay only if exactly 1 task on that named day
            one_task_day = (tasks_on_named_day[(person, week, day)] == 1)
        else:
            one_day_str = "N/A"
            one_task_day = False

        # exclusions after pre-pass: pairwise on same (person, week, day) — only for named days
        excl_ok = True
        if day and day.lower() != "all":
            same_day_names = [nm2 for (_h2, _hc2, p2, nm2, w2, d2, _cid2, _rp2)
                              in per_task_rows if p2==person and w2==week and d2==day]
            for i in range(len(same_day_names)):
                for j in range(i+1, len(same_day_names)):
                    a, b = same_day_names[i], same_day_names[j]
                    if b in exclusions.get(a, set()):
                        excl_ok = False
                        break
                if not excl_ok: break

        # cooldown tag for this component (current week), after pre-pass
        cd_tag = cooldown_tag_for_comp.get(cid, "")

        # provenance flags (base)
        prepass_moved = ("YES" if hdr_canon in prepass_canon else "NO")
        manual_assignment = ("YES" if cid in manual_cids else "NO")

        # --- NEW: override/augment based on names present in Task ID tails ---
        # If the assignee is embedded in this Task ID => force ManualAssignment=YES
        if header_has_person(hdr_exact, person):
            manual_assignment = "YES"

        # If the assignee appears in ANY sibling Task ID (other repeat of same task header) => PrePassMoved=YES
        sib_headers = find_sibling_header(hdr_exact)
        if any(header_has_person(hh, person) for hh in sib_headers):
            prepass_moved = "YES"
        # --------------------------------------------------------------------

        ignore_availability = "YES" if nm in ignore_avail else "NO"

        # build fallback cell
        fb_parts = []
        if one_task_day: fb_parts.append("OneTaskDay")
        if cd_tag:       fb_parts.append(cd_tag)
        fb = ",".join(fb_parts)

        out.append({
            "Task ID": hdr_exact,                 # keep the exact header as seen in the grid
            "Assignee": person,
            "Fallback": fb,
            "PrePassMoved": prepass_moved,
            "ManualAssignment": manual_assignment,
            "IgnoreAvailability": ignore_availability,
            "OneDayPerWeek_OK": one_day_str,
            "Availability_OK": "YES" if avail_ok else "NO",
            "Exclusions_OK": "YES" if excl_ok else "NO",
            "PrevWeekCooldown_OK": "NO" if cd_tag else "YES",
        })

    # --- EXACT GAS ORDERING + BLANK ROWS BETWEEN WEEKS ---

    def _empty_row():
        return {
            "Task ID": "",
            "Assignee": "",
            "Fallback": "",
            "PrePassMoved": "",
            "ManualAssignment": "",
            "IgnoreAvailability": "",
            "OneDayPerWeek_OK": "",
            "Availability_OK": "",
            "Exclusions_OK": "",
            "PrevWeekCooldown_OK": "",
        }

    # Map last-produced row per Task ID (GAS "last write wins" behavior)
    rows_by_header = {}
    for r in out:
        rows_by_header[r["Task ID"]] = r

    final_rows = []
    prev_week = None

    for h in task_headers:
        wk, dy, tm, nm, rp, _canon = parse_task_header(h)

        # insert a blank separator row when week changes (but not before the first)
        if prev_week is not None and wk != prev_week:
            final_rows.append(_empty_row())

        if h in rows_by_header:
            final_rows.append(rows_by_header[h])
        else:
            banned_manual_assignee = header_assignee(h) if nm in banned_tasks else ""
            manual_flag = "YES" if banned_manual_assignee else "NO"
            # placeholder row so position matches original column
            final_rows.append({
                "Task ID": h,
                "Assignee": banned_manual_assignee,
                "Fallback": "",
                "PrePassMoved": "NO",
                "ManualAssignment": manual_flag,
                "IgnoreAvailability": "YES" if nm in ignore_avail else "NO",
                "OneDayPerWeek_OK": "N/A",
                "Availability_OK": "",
                "Exclusions_OK": "",
                "PrevWeekCooldown_OK": "",
            })

        prev_week = wk  # update after handling this header

    # Do NOT sort—this preserves the sheet’s left-to-right order with week breaks.
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "Task ID", "Assignee", "Fallback",
            "PrePassMoved", "ManualAssignment", "IgnoreAvailability",
            "OneDayPerWeek_OK", "Availability_OK", "Exclusions_OK", "PrevWeekCooldown_OK"
        ])
        w.writeheader()
        w.writerows(final_rows)

    print(f"Wrote per-task assignments → {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
