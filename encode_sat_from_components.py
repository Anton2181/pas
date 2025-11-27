#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tango task encoder.

The script reads ``components_all.csv`` and ``backend.csv`` and emits three
artifacts:

* ``schedule.opb`` – PB model consumed by SAT4J.
* ``varmap.json`` – labels for diagnostics/plots/consumer scripts.
* ``stats.txt`` – human-readable recap of the options that shaped the model.

Hard constraints:

* exactly-one assignment per component
* at most one task per (person, week, family) once sibling tokens overlap
* two named days per week are forbidden unless one side is fully manual
* manual “Both” links transfer candidates between repeat pairs
* same-day task exclusions and optional banned person pairs

Soft objective (highest → lowest tier):

1. cooldown ladders, repeat limits, and streaks (priority / non-priority)
2. manual-day coverage nudges and "Both" fallback penalties
3. deprioritized same-day pairs
4. preferred-pair misses
5. across-horizon fairness ladders
6. optional priority-coverage encouragement

Auto softening detects under-staffed families and can skip selected penalties
for them so their scarcity does not dominate the solve.
"""

from __future__ import annotations
import argparse, copy, csv, io, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable
from collections import defaultdict, deque

SCRIPT_DIR = Path(__file__).resolve().parent

# =============== CONFIG (flags + weights together) ====================
DEFAULT_CONFIG = {
    # Debug / relax selectors
    "DEBUG_RELAX": True,
    "DEBUG_ALLOW_UNASSIGNED": False,
    "W_HARD": 1_000_000_000_000_000_000_000,  # ≥ W1

    # Minimum-effort encouragement
    "EFFORT_FLOOR_TARGET": 6,
    "EFFORT_FLOOR_HARD": True,

    # Cooldown options (prev-week; separate from repeat limits below)
    "PRIORITY_COOLDOWN_HARD": False,
    "NONPRIORITY_COOLDOWN_HARD": False,

    # Geometric scaling bases
    "COOLDOWN_GEO": 3,
    "REPEAT_OVER_GEO": 2,

    # Repeat limits (per person × family across the full horizon)
    "REPEAT_LIMIT": {
        "PRI": 1,
        "NON": 1,
    },
    "REPEAT_LIMIT_HARD": {
        "PRI": False,
        "NON": False,
    },


    # Priority coverage pressure
    "PRIORITY_COVERAGE_MODE": "global",

    "SUNDAY_TWO_DAY_SOFT": True,    # when True: allow Tue+Sun (etc) even if both AUTO; add soft penalty
    "TWO_DAY_SOFT_ALL": True,

    # Auto-softening: detect families with very few eligible people and skip
    # building the harshest cooldown/repeat penalties for them.
    "AUTO_SOFTEN": {
        "ENABLED": False,
        "MIN_UNIQUE_CANDIDATES": 3,
        "MAX_SLOTS_PER_PERSON": 1.5,
        "RELAX_COOLDOWN": True,
        "RELAX_REPEAT": True,
    },

    "FAIR_MEAN_MULTIPLIER": 1.0,
    "FAIR_OVER_START_DELTA": 0,

    # Optional weight ladder (ordered list where each entry is RATIO× the next)
    "WEIGHT_LADDER": {
        # Strongest → weakest ladder. Enable to derive numeric weights from this
        # ordering with ``RATIO`` (each rung is ``RATIO``× the next). Inline
        # notes explain the intent, when the weight triggers, and a concrete
        # example of a violation that would pay the cost.
        #   * W_DEBUG_UNASSIGNED_PRIORITY / _NON_PRIORITY – debug-only drop
        #     selectors; activate when a task is intentionally left unassigned
        #     under debug relax mode; e.g., marking a hard-to-place component as
        #     dropped. Priority tasks can be weighted separately from non-priority
        #     tasks.
        #   * W4 – soft cost for using the “Both” expansion to assign a manual pair;
        #     activates when a person is auto-selected for a “Both” link; e.g.,
        #     filling both halves of a manual repeat in one step.
        #   * W4_DPR – same-day deprioritized pair penalty; activates when a person
        #     takes two tasks forming a deprioritized pair on the same day; e.g.,
        #     working two incompatible tasks on Saturday.
        #   * W_EFFORT_FLOOR – enforces/encourages ≥target effort for eligible
        #     people; activates when an eligible person could reach the target but
        #     does not; e.g., someone capable of 8 effort only gets 6.
        #   * W_PRIORITY_MISS – ensures eligible priority specialists get at least
        #     one priority task; activates when a top-eligible person receives zero;
        #     e.g., an expert never assigned any priority work that week.
        #   * W_TWO_DAY_SOFT – general named-day softening; activates when two named
        #     days are both auto-assigned under soft mode; e.g., taking Monday and
        #     Wednesday when not fully manual.
        #   * W1_COOLDOWN_INTRA – sibling-proximity guard for priority families;
        #     triggers when a priority person appears twice in the same family
        #     during sibling-linked weeks; e.g., covering both tokens of a split
        #     family in the same window.
        #   * W2_COOLDOWN_INTRA – sibling-proximity guard for non-priority
        #     families; triggers when a non-priority person serves both halves of a
        #     sibling pair too closely; e.g., covering linked weeks for Family Z.
        #   * T1C – pushes broad coverage of top-priority families; activates when
        #     top-priority capacity goes unused; e.g., skipping a top slot to keep
        #     someone idle.
        #   * T2C – secondary priority coverage; activates when secondary priority
        #     families are left empty after top coverage; e.g., leaving a second-tier
        #     slot open.
        #   * W1_REPEAT – penalizes exceeding the per-family priority repeat cap;
        #     triggers once a priority person crosses their allowed count; e.g., a
        #     second assignment to Family X when the limit is 1.
        #   * W1_STREAK – discourages back-to-back weeks in the same priority
        #     family; activates when a priority person repeats across adjacent weeks;
        #     e.g., Alice serves Family X in weeks 10 and 11.
        #   * W1_COOLDOWN – counts steps inside the priority cooldown ladder;
        #     activates on short gaps after a priority service; e.g., serving the
        #     same family again just one week later.
        #   * W2_REPEAT – penalizes exceeding the non-priority repeat cap; triggers
        #     when a non-priority person goes past their family limit; e.g., a third
        #     assignment where only two were allowed.
        #   * W2_STREAK – discourages back-to-back weeks for non-priority families;
        #     activates on consecutive non-priority assignments; e.g., Bob handles
        #     Family Y in weeks 12 and 13.
        #   * W2_COOLDOWN – counts steps in the non-priority cooldown ladder;
        #     activates on short gaps after a non-priority service; e.g., returning
        #     to a family after a single-week break.
        #   * W_SUNDAY_TWO_DAY – softens named-day bans involving Sunday; activates
        #     when a person takes two named days including Sunday under softened
        #     mode; e.g., Tuesday+Sunday combination.
        #   * W_AUTO_DAY – discourages single-task AUTO days on non-Sundays;
        #     activates when AUTO appears on a weekday with <2 tasks.
        #   * W_AUTO_DAY_SUNDAY – same as W_AUTO_DAY but scoped to Sunday,
        #     letting you tune weekend behavior separately.
        #   * W3 – “fill to two” nudger on manual-only days; activates when a day is
        #     short-staffed; e.g., only one manual assignment on a day that expects
        #     two.
        #   * W5 – preferred-pair miss; activates when a feasible preferred pair is
        #     not scheduled; e.g., two people who like to partner are assigned
        #     separately.
        #   * W6_UNDER – fairness ladder for under-loaded people; activates when a
        #     person receives less than peers; e.g., far fewer assignments than the
        #     median.
        #   * W6_OVER – fairness ladder for over-loaded people; activates when a
        #     person receives more than peers; e.g., significantly above-average
        #     assignment counts.
        "ENABLED": True,
        "ORDER": [
            "W_DEBUG_UNASSIGNED_PRIORITY",
            "W_DEBUG_UNASSIGNED_NON_PRIORITY",
            "W4",
            "W4_DPR",
            "W_EFFORT_FLOOR",
            "W_PRIORITY_MISS",
            "W_TWO_DAY_SOFT",
            "W1_COOLDOWN_INTRA",
            "W2_COOLDOWN_INTRA",
            "T1C",
            "T2C",
            "W1_REPEAT",
            "W1_STREAK",
            "W1_COOLDOWN",
            "W2_REPEAT",
            "W2_STREAK",
            "W2_COOLDOWN",
            "W_SUNDAY_TWO_DAY",
            "W_AUTO_DAY",
            "W_AUTO_DAY_SUNDAY",
            "W3",
            "W5",
            "W6_UNDER",
            "W6_OVER",
        ],
        "RATIO": 10,
        # Optional anchor for the strongest rung. If omitted, it defaults to
        # ``RATIO ** (len(ORDER) - 1)`` so the weakest rung bottoms out at ~1.
        "TOP": None,
    },

    # Weights (strict ×1000 scaling between major tiers)
    "WEIGHTS": {
        # --- Debug helper (other weights are derived from the ladder) ---
        "W_DEBUG_UNASSIGNED_PRIORITY": 1_000_000_000_000_000_000,
        "W_DEBUG_UNASSIGNED_NON_PRIORITY": 1_000_000_000_000_000_000,

        # Availability-aware fairness scaling (Tier-6 helper)
        "FAIRNESS_AVAILABILITY": {
            "ENABLED": True,
            "REFERENCE": "auto",  # "auto" or "all"
            "MIN_RATIO": 0.35,
            "MAX_RATIO": 1.85,
            "POWER": 0.75,
        },

    },

    # Banned unordered person pairs
    "BANNED_SIBLING_PAIRS": [
        ["Yulia Talybova", "Marius Latinis"],
        ["Sylwia Woźniak", "Marius Latinis"],
    ],
    "BANNED_SAME_DAY_PAIRS": [
        ["Sylwia Woźniak", "Jan Mężyński"],
        ["Sylwia Woźniak", "Pawel Wypijewski"],
        ["Antoni Domanowski", "Pawel Wypijewski"],
        ["Antoni Domanowski", "Jan Mężyński"],
        ["Nina Andrzejczyk", "Pawel Wypijewski"],
        ["Nina Andrzejczyk", "Jan Mężyński"],
        ["Nina Andrzejczyk", "Anna Sidorowicz"],
        ["Pawel Wypijewski", "Anna Sidorowicz"],
    ],
}

CONFIG = copy.deepcopy(DEFAULT_CONFIG)


def deep_update(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` (in-place)."""

    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _greedy_effort_floor_probe(
    *,
    target: int,
    eligible_people: Iterable[str],
    comps: List[CompRow],
    candidates: Dict[str, List[str]],
    original_manual: Dict[str, bool],
    blocked: Dict[str, Set[str]] | None = None,
) -> tuple[bool, dict]:
    """Attempt to cover the effort floor using a conservative greedy heuristic.

    The probe respects per-week family exclusivity, forbids more than one AUTO
    named-day per person/week (mirroring the hard two-day constraint), and
    limits people to one task per named day. If the probe cannot find a
    covering assignment, we treat the floor as infeasible rather than risk an
    UNSAT model.
    """

    remaining = {p: target for p in eligible_people}
    if not remaining or target <= 0:
        return True, {"assignments": {}, "covered_effort": {}, "remaining_need": remaining}

    # Track usage to mirror the strongest per-person hard rules.
    family_week_used: Dict[str, Set[tuple[int, str]]] = defaultdict(set)
    auto_days_by_week: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    assignments: Dict[str, List[str]] = defaultdict(list)

    comp_meta: List[tuple[int, CompRow]] = []
    for r in comps:
        effort_units = max(1, int(math.ceil(getattr(r, "total_effort", max(1, int(r.task_count))))))
        comp_meta.append((effort_units, r))

    # Highest-effort first to reduce waste; stable tie-breaker on cid for tests.
    comp_meta.sort(key=lambda t: (-t[0], t[1].cid))

    for effort_units, r in comp_meta:
        if all(need <= 0 for need in remaining.values()):
            break
        if effort_units <= 0:
            continue

        fams = tuple(r.sibling_key) if r.sibling_key else (r.cid,)
        fams = tuple(trim(f) for f in fams if trim(f))
        week = int(getattr(r, "week_num", 0) or 0)
        day = trim(getattr(r, "day", ""))
        is_auto = not original_manual.get(r.cid, False)

        # Assign to the person with the highest remaining need that can legally take it.
        for person in sorted(remaining, key=lambda p: remaining[p], reverse=True):
            if remaining[person] <= 0:
                continue
            if blocked and r.cid in blocked.get(person, set()):
                continue
            if person not in candidates.get(r.cid, []):
                continue
            if any((week, fam) in family_week_used[person] for fam in fams):
                continue
            if is_auto:
                seen = auto_days_by_week[person].get(week, set())
                if seen and day and day not in seen:
                    continue

            # Take the assignment for this person.
            remaining[person] -= effort_units
            assignments[person].append(r.cid)
            family_week_used[person].update((week, fam) for fam in fams)
            if is_auto:
                auto_days_by_week[person][week].add(day or "__UNNAMED__")
            break

    covered = {p: target - max(need, 0) for p, need in remaining.items()}
    success = all(need <= 0 for need in remaining.values())
    return success, {
        "assignments": assignments,
        "covered_effort": covered,
        "remaining_need": remaining,
    }


def _apply_weight_ladder(cfg: dict) -> None:
    ladder_cfg = cfg.get("WEIGHT_LADDER") or {}
    order: List[str] = ladder_cfg.get("ORDER") or []
    if not ladder_cfg.get("ENABLED") or not order:
        return

    ratio = int(ladder_cfg.get("RATIO", 100))
    if ratio <= 1:
        raise ValueError("WEIGHT_LADDER.RATIO must be >1")

    # Ensure every known weight gets a derived value by appending any missing
    # defaults after a custom ORDER prefix.
    default_order = [w for w in DEFAULT_CONFIG.get("WEIGHT_LADDER", {}).get("ORDER", []) if w not in order]
    order = [*order, *default_order]
    cfg.setdefault("WEIGHT_LADDER", {})["ORDER"] = order

    weights = copy.deepcopy(cfg.get("WEIGHTS", {}))
    top = ladder_cfg.get("TOP")
    anchor = int(top) if top is not None else None
    if anchor is None:
        anchor = int(ratio ** max(len(order) - 1, 0))

    current = anchor
    for name in order:
        weights[name] = current
        current = max(1, current // ratio)

    cfg["WEIGHTS"] = weights


def build_config(overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        deep_update(cfg, overrides)
    _apply_weight_ladder(cfg)
    return cfg

# =====================================================================

# -------------------- CLI (paths only) --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components",       default="components_all.csv", type=Path)
    ap.add_argument("--backend",          default="backend.csv",       type=Path)
    ap.add_argument("--out",              default="schedule.opb",      type=Path)
    ap.add_argument("--map",              default="varmap.json",       type=Path)
    ap.add_argument("--stats",            default="stats.txt",         type=Path)
    ap.add_argument("--family-registry", default="family_registry.json", type=Path,
                    help="JSON registry of canonical family tokens, aliases, and component membership")
    ap.add_argument("--config", type=Path, help="Optional JSON file with CONFIG overrides")
    return ap.parse_args()

# -------------------- Helpers --------------------
def trim(s: str) -> str:
    return (s or "").strip()

def to_int(s: str, default: int) -> int:
    try:
        return int(trim(s) or str(default))
    except Exception:
        return default


def resolve_data_path(path: Path) -> Path:
    """Locate a data file relative to CWD or the script directory.

    This preserves the prior right-click/run UX in IDEs by falling back to the
    repository directory if the current working directory does not contain the
    default CSVs.
    """

    if path.exists():
        return path
    if not path.is_absolute():
        alt = SCRIPT_DIR / path
        if alt.exists():
            return alt
    return path


def to_float(s: str, default: float) -> float:
    try:
        return float((trim(s) or str(default)).replace(",", "."))
    except Exception:
        return default

def parse_week_num(week_label: str) -> int:
    digits = "".join(ch for ch in week_label if ch.isdigit())
    return int(digits) if digits else 0

def split_pipe(s: str) -> List[str]:
    s = trim(s)
    return [x.strip() for x in s.split("|") if x.strip()] if s else []

def split_csv_people(s: str) -> List[str]:
    s = trim(s)
    return [x.strip() for x in s.split(",") if x.strip()] if s else []

def split_sibling_key(s: str) -> List[str]:
    s = trim(s)
    if not s:
        return []
    return [x.strip() for x in s.split("||") if x.strip()]

@dataclass
class CompRow:
    cid: str
    week_label: str
    week_num: int
    day: str
    repeat: int
    repeat_max: int
    task_count: int
    total_effort: float
    names: List[str]
    priority: bool
    assigned_flag: bool
    assigned_to: List[str]
    candidates_all: List[str]
    candidates_role: List[str]
    sibling_key: Tuple[str, ...]  # canonical family tokens used for sibling/cooldown grouping
    # Flags for two-tier priority spread
    is_top: bool = field(default=False)     # Top Priority (column AE)
    is_second: bool = field(default=False)  # Second Priority (column AF)


@dataclass
class AutoSoftener:
    """Tracks families with scarce staffing and knows which penalties to skip."""

    enabled: bool
    min_unique: int
    max_slots_ratio: float
    relax_cooldown: bool
    relax_repeat: bool
    notes: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def analyze(self, comps: List[CompRow], candidates: Dict[str, List[str]]) -> None:
        if not self.enabled:
            self.notes = {}
            return

        fam_people: Dict[str, Set[str]] = defaultdict(set)
        fam_slots: Dict[str, int] = defaultdict(int)

        for r in comps:
            fams = tuple(r.sibling_key) if r.sibling_key else (r.cid,)
            fams = tuple(trim(f) for f in fams if trim(f))
            for fam in fams:
                fam_slots[fam] += 1
                fam_people[fam].update(candidates.get(r.cid, []))

        scarce: Dict[str, Dict[str, str]] = {}
        for fam, slots in fam_slots.items():
            uniq = len(fam_people[fam])
            ratio = float("inf") if uniq == 0 else slots / max(1, uniq)
            reasons: List[str] = []
            if uniq == 0:
                reasons.append("no_candidates")
            if uniq <= self.min_unique:
                reasons.append(f"unique<= {self.min_unique}")
            if ratio >= self.max_slots_ratio:
                reasons.append(f"slots/person>= {self.max_slots_ratio}")
            if reasons:
                scarce[fam] = {
                    "slots": str(slots),
                    "unique_people": str(uniq),
                    "slots_per_person": (
                        "inf" if ratio == float("inf") else f"{ratio:.2f}"
                    ),
                    "reasons": ", ".join(reasons),
                }

        self.notes = scarce

    def should_skip(self, kind: str, fam: str) -> bool:
        if not self.enabled:
            return False
        fam_trim = trim(fam)
        if not fam_trim or fam_trim not in self.notes:
            return False
        if kind == "cooldown":
            return self.relax_cooldown
        if kind == "repeat":
            return self.relax_repeat
        return False


@dataclass
class FamilyRegistry:
    """Loads/stores canonical family labels so humans can rename them."""

    path: Path
    mapping: Dict[str, str] = field(default_factory=dict)
    components: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def _write_json(self) -> None:
        payload = {
            "families": [
                {
                    "canonical": fam,
                    "alias": trim(self.mapping.get(fam, "")),
                    "components": [
                        {
                            "id": cid,
                            "tasks": sorted(tasks),
                        }
                        for cid, tasks in sorted(self.components.get(fam, {}).items())
                    ],
                }
                for fam in self.order
            ]
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load(self) -> None:
        self.mapping = {}
        self.components = {}
        self.order = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_json()
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"families": []}
        for entry in data.get("families", []):
            fam = trim(entry.get("canonical", ""))
            alias = trim(entry.get("alias", ""))
            comps: Dict[str, Set[str]] = {}
            for comp in entry.get("components", []) or []:
                cid = trim(comp.get("id", ""))
                if not cid:
                    continue
                tasks = {trim(t) for t in (comp.get("tasks") or []) if trim(t)}
                comps[cid] = tasks
            if fam:
                self.order.append(fam)
                self.mapping[fam] = alias
                self.components[fam] = comps

    def sync(self, families_in_order: Iterable[str], fam_components: Dict[str, Dict[str, Set[str]]] | None = None) -> None:
        fam_components = fam_components or {}
        seen_new: Set[str] = set()
        for fam in families_in_order:
            fam_trim = trim(fam)
            if not fam_trim:
                continue
            if fam_trim not in self.order:
                self.order.append(fam_trim)
            if fam_trim not in self.mapping:
                self.mapping[fam_trim] = ""
            if fam_trim in seen_new:
                continue
            seen_new.add(fam_trim)
        for fam, comps in fam_components.items():
            fam_trim = trim(fam)
            if not fam_trim:
                continue
            merged = {k: set(v) for k, v in self.components.get(fam_trim, {}).items()}
            for cid, tasks in comps.items():
                if not cid:
                    continue
                merged.setdefault(cid, set()).update({t for t in tasks if t})
            self.components[fam_trim] = merged

        self._write_json()

    def label(self, fam: str) -> str:
        alias = trim(self.mapping.get(fam, ""))
        return alias or fam

def read_csv_matrix(path: Path) -> List[List[str]]:
    actual = resolve_data_path(path)
    txt = actual.read_text(encoding="utf-8-sig", errors="replace")
    return [list(row) for row in csv.reader(io.StringIO(txt))]

def load_components(path: Path) -> Tuple[List[CompRow], Set[str], bool]:
    path = resolve_data_path(path)
    rows: List[CompRow] = []
    people: Set[str] = set()
    used_siblingkey = False
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        hdr = [trim(h) for h in (rdr.fieldnames or [])]
        have_sibling_key = "siblingkey" in {h.lower() for h in hdr}
        for rr in rdr:
            cid = trim(rr.get("ComponentId",""))
            if not cid:
                continue
            week_label = trim(rr.get("Week",""))
            day = trim(rr.get("Day",""))
            week_num = parse_week_num(week_label)
            repeat     = to_int(rr.get("Repeat",""), 1)
            repeat_max = to_int(rr.get("RepeatMax",""), 1)
            task_count = to_int(rr.get("Task Count",""), repeat_max)

            names         = split_pipe(rr.get("Names",""))
            prio          = trim(rr.get("Priority","")).upper() == "YES"
            assigned_flag = trim(rr.get("Assigned?","")).upper() == "YES"
            assigned_to   = split_csv_people(rr.get("Assigned To",""))
            cand_role     = split_csv_people(rr.get("Role-Filtered Candidates",""))
            cand_all      = split_csv_people(rr.get("Candidates",""))
            candidates_role = cand_role if cand_role else cand_all
            total_effort  = max(0.0, to_float(rr.get("Total Effort", ""), task_count))

            people.update(assigned_to)
            people.update(candidates_role)

            sib = tuple(split_sibling_key(rr.get("SiblingKey",""))) if have_sibling_key else tuple()
            used_siblingkey |= bool(sib)

            rows.append(CompRow(
                cid=cid, week_label=week_label, week_num=week_num, day=day,
                repeat=repeat, repeat_max=repeat_max, task_count=task_count, total_effort=total_effort,
                names=names, priority=prio, assigned_flag=assigned_flag,
                assigned_to=assigned_to, candidates_all=cand_all, candidates_role=candidates_role,
                sibling_key=sib
            ))
    return rows, people, used_siblingkey

# --------------------- Backend parsing (roles/exclusions/pairs) ---------------------
def load_backend_roles_and_maps(backend_matrix: List[List[str]]):
    hdr = [trim(h) for h in (backend_matrix[0] if backend_matrix else [])]
    def find_col(name: str) -> int | None:
        lname = name.strip().lower()
        for i, h in enumerate(hdr):
            if h.strip().lower() == lname:
                return i
        return None

    col_name = 0
    col_both = 3
    col_ex_a = 17
    col_ex_b = 18

    c_pp1 = find_col("Preferred Pair 1")
    c_pp2 = find_col("Preferred Pair 2")

    c_dpr1   = find_col("DeprioritizedPairs1")
    c_dpr2   = find_col("DeprioritizedPairs2")

    # Columns for Top/Second priority task names (by header or fallback to AE/AF)
    c_top = find_col("Top Priority")
    c_second = find_col("Second Priority")
    if c_top is None and len(hdr) > 30:   # AE (0-based index ~ 30)
        c_top = 30
    if c_second is None and len(hdr) > 31:  # AF (0-based index ~ 31)
        c_second = 31

    both_people: Set[str] = set()
    exclusions: Dict[str, Set[str]] = {}
    preferred: Dict[str, Set[str]] = defaultdict(set)
    deprioritized: Dict[str, Set[str]] = defaultdict(set)
    top_priority_tasks: Set[str] = set()
    second_priority_tasks: Set[str] = set()

    # "Both" roles
    for r in range(1, len(backend_matrix)):
        row = backend_matrix[r]
        name = trim(row[col_name]) if len(row) > col_name else ""
        if not name:
            continue
        both = trim(row[col_both]).upper() in ("1","TRUE","YES") if len(row) > col_both else False
        if both:
            both_people.add(name)

    # Pairwise exclusions (task-task)
    for r in range(1, len(backend_matrix)):
        row = backend_matrix[r]
        a = trim(row[col_ex_a]) if len(row) > col_ex_a else ""
        b = trim(row[col_ex_b]) if len(row) > col_ex_b else ""
        if a and b:
            exclusions.setdefault(a, set()).add(b)
            exclusions.setdefault(b, set()).add(a)

    # Cooldown family graph (columns AB:AC)
    adj = defaultdict(set)
    for r in range(1, len(backend_matrix)):
        row = backend_matrix[r]
        a = trim(row[27]) if len(row) > 27 else ""
        b = trim(row[28]) if len(row) > 28 else ""
        if a and b:
            adj[a].add(b)
            adj[b].add(a)

    # Preferred pairs (unordered)
    if c_pp1 is not None and c_pp2 is not None:
        for r in range(1, len(backend_matrix)):
            row = backend_matrix[r]
            a = trim(row[c_pp1]) if len(row) > c_pp1 else ""
            b = trim(row[c_pp2]) if len(row) > c_pp2 else ""
            if a and b:
                preferred[a].add(b)
                preferred[b].add(a)
    # Deprioritized (task-task) pairs: legal, but soft-penalized if same person does both same day
    if c_dpr1 is not None and c_dpr2 is not None:
        for r in range(1, len(backend_matrix)):
            row = backend_matrix[r]
            a = trim(row[c_dpr1]) if len(row) > c_dpr1 else ""
            b = trim(row[c_dpr2]) if len(row) > c_dpr2 else ""
            if a and b:
                deprioritized[a].add(b)
                deprioritized[b].add(a)  # treat as unordered; penalize either orientation



    # Collect Top/Second priority task names (cells can repeat; sets dedupe)
    if c_top is not None:
        for r in range(1, len(backend_matrix)):
            row = backend_matrix[r]
            if len(row) > c_top:
                t = trim(row[c_top])
                if t:
                    top_priority_tasks.add(t)
    if c_second is not None:
        for r in range(1, len(backend_matrix)):
            row = backend_matrix[r]
            if len(row) > c_second:
                t = trim(row[c_second])
                if t:
                    second_priority_tasks.add(t)

    # Canonical representative per connected component
    cooldown_key: Dict[str, str] = {}
    seen: Set[str] = set()
    for start in list(adj.keys()):
        if start in seen:
            continue
        dq, comp = deque([start]), []
        seen.add(start)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        canon = sorted(comp)[0]
        for n in comp:
            cooldown_key[n] = canon

    return both_people, cooldown_key, exclusions, preferred, top_priority_tasks, second_priority_tasks, deprioritized


def canonical_families(task_names: Iterable[str], cooldown_key: Dict[str, str]) -> Tuple[str, ...]:
    fams = {trim(cooldown_key.get(t, t)) for t in (task_names or [])}
    fams = {f for f in fams if f}
    return tuple(sorted(fams)) if fams else tuple()

# --------------------- PB writer and helpers ---------------------
class PBWriter:
    def __init__(self, debug_relax: bool, W_HARD: int):
        self.constraints: List[str] = []
        self.vars: Set[str] = set()
        self.objective_terms: List[Tuple[int, str]] = []
        self.debug_relax = debug_relax
        self.W_HARD = W_HARD
        self._cur = 1
        self._selmap: Dict[str, str] = {}
        self.debug_rows: List[Dict[str, str]] = []

    def new_var(self) -> str:
        name = f"x{self._cur}"
        self._cur += 1
        self.vars.add(name)
        return name

    def selector(self, label: str) -> str:
        if label in self._selmap:
            return self._selmap[label]
        v = self.new_var()
        self._selmap[label] = v
        return v

    @staticmethod
    def _fmt_term(coef: int, xname: str, first: bool) -> str:
        sgn_num = f"{'+' if coef >= 0 else '-'}{abs(coef)}"
        return ("" if first else " ") + f"{sgn_num} {xname}"

    def _emit_ge(self, terms: List[Tuple[int, str]], rhs: int):
        for _, x in terms:
            self.vars.add(x)
        body = "".join(self._fmt_term(c, x, i == 0) for i, (c, x) in enumerate(terms))
        self.constraints.append(f"{body} >= {rhs} ;")

    def add_objective_terms(self, pairs: List[Tuple[int, str]]):
        for w, x in pairs:
            if w == 0:
                continue
            self.vars.add(x)
            self.objective_terms.append((w, x))

    def add_ge(self, terms: List[Tuple[int, str]], rhs: int,
               *, relax_label: str | None = None, M: int = 0, info: Dict[str, str] | None = None):
        if not terms:
            if self.debug_relax and relax_label is not None:
                s = self.selector(relax_label)
                if M <= 0:
                    M = max(rhs, 0) if rhs > 0 else 1
                self._emit_ge([(M, s)], rhs)
                self.add_objective_terms([(self.W_HARD, s)])
                if info is not None:
                    row = dict(info); row["selector"] = s; row["label"] = relax_label
                    row["form"] = "GE(empty)"; row["M"] = str(M); self.debug_rows.append(row)
            else:
                if rhs <= 0:
                    self.constraints.append("0 >= 0 ;")
                else:
                    d = self.new_var(); self._emit_ge([(1, d)], rhs + 1)
            return

        if self.debug_relax and relax_label is not None:
            s = self.selector(relax_label)
            if M <= 0:
                M = max(rhs, 0) + sum(abs(c) for c, _ in terms) + 1
            self._emit_ge(terms + [(M, s)], rhs)
            self.add_objective_terms([(self.W_HARD, s)])
            if info is not None:
                row = dict(info); row["selector"] = s; row["label"] = relax_label
                row["form"] = "GE"; row["M"] = str(M); self.debug_rows.append(row)
        else:
            self._emit_ge(terms, rhs)

    def add_le(self, terms: List[Tuple[int, str]], rhs: int,
               *, relax_label: str | None = None, M: int = 0, info: Dict[str, str] | None = None):
        if self.debug_relax and relax_label is not None:
            s = self.selector(relax_label)
            if M <= 0:
                M = (sum(abs(c) for c, _ in terms) if terms else 0) + 1
            ge_terms = [(-c, x) for c, x in terms] + [(M, s)]
            self.add_ge(ge_terms, -rhs, relax_label=None)
            self.add_objective_terms([(self.W_HARD, s)])
            if info is not None:
                row = dict(info); row["selector"] = s; row["label"] = relax_label
                row["form"] = "LE"; row["M"] = str(M); self.debug_rows.append(row)
        else:
            if not terms:
                if rhs >= 0:
                    self.constraints.append("0 >= 0 ;")
                else:
                    d = self.new_var(); self._emit_ge([(1, d)], 1)
            else:
                self._emit_ge([(-c, x) for c, x in terms], -rhs)

    def add_eq(self, terms: List[Tuple[int, str]], rhs: int,
               *, relax_label: str | None = None, M: int = 0, info: Dict[str, str] | None = None):
        if self.debug_relax and relax_label is not None:
            if M <= 0:
                M = (sum(abs(c) for c, _ in terms) if terms else 0) + abs(rhs) + 1
            self.add_ge(terms, rhs, relax_label=relax_label, M=M, info=info)
            self.add_le(terms, rhs, relax_label=relax_label, M=M, info=info)
        else:
            if not terms:
                if rhs == 0:
                    self.constraints.append("0 >= 0 ;")
                else:
                    d = self.new_var()
                    self._emit_ge([(1, d)], abs(rhs) + 1)
                    self._emit_ge([(-1, d)], -abs(rhs) - 1)
            else:
                self._emit_ge(terms, rhs)
                self._emit_ge([(-c, x) for c, x in terms], -rhs)

    def set_objective(self, weighted_vars: List[Tuple[int, str]]):
        self.add_objective_terms(weighted_vars)

    def dump(self, path: Path):
        if not self.vars:
            self.vars.add("x1")
        lines: List[str] = []
        lines.append(f"* #variable= {len(self.vars)} #constraint= {len(self.constraints)}")
        if self.objective_terms:
            obj = "".join(self._fmt_term(w, x, i == 0) for i, (w, x) in enumerate(self.objective_terms))
            lines.append(f"min: {obj} ;")
        else:
            d = f"x{len(self.vars)+1}"
            lines[0] = f"* #variable= {len(self.vars)+1} #constraint= {len(self.constraints)}"
            lines.append(f"min: +1 {d} ;")
        lines.extend(self.constraints)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------- Indicator helpers ----------
def make_or(pb: PBWriter, xs) -> str | None:
    if not xs:
        return None
    cleaned: List[str] = []
    for item in xs:
        if isinstance(item, str):
            cleaned.append(item)
        elif isinstance(item, (tuple, list)) and item and isinstance(item[0], str):
            cleaned.append(item[0])
        elif item is None:
            continue
        else:
            raise TypeError(f"make_or received non-variable item: {item!r}")
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    A = pb.new_var()
    for x in cleaned:
        pb.add_le([(1, x), (-1, A)], 0)
    pb.add_ge([(1, x) for x in cleaned] + [(-1, A)], 0)
    return A

def make_and(pb: PBWriter, a: str, b: str) -> str:
    z = pb.new_var()
    pb.add_le([(1, z), (-1, a)], 0)
    pb.add_le([(1, z), (-1, b)], 0)
    pb.add_le([(1, a), (1, b), (-1, z)], 1)
    return z

def make_and_not(pb: PBWriter, d: str, a: str | None) -> str:
    if a is None:
        return d
    y = pb.new_var()
    pb.add_le([(1, y), (-1, d)], 0)
    pb.add_le([(1, y), (1, a)], 1)
    pb.add_le([(1, d), (-1, a), (-1, y)], 0)
    return y


def _find_duplicates(weighted_vars: List[Tuple[int, str]]) -> Set[Tuple[int, str]]:
    seen: Set[Tuple[int, str]] = set()
    duplicates: Set[Tuple[int, str]] = set()
    for pair in weighted_vars:
        if pair in seen:
            duplicates.add(pair)
        else:
            seen.add(pair)
    return duplicates

# -------------------- Encoder --------------------
def _encode(args):
    DEBUG_RELAX = bool(CONFIG["DEBUG_RELAX"])
    DEBUG_ALLOW_UNASSIGNED = bool(CONFIG["DEBUG_ALLOW_UNASSIGNED"])
    W_HARD = int(CONFIG["W_HARD"])

    PRIORITY_COOLDOWN_HARD = bool(CONFIG["PRIORITY_COOLDOWN_HARD"])
    NONPRIORITY_COOLDOWN_HARD = bool(CONFIG["NONPRIORITY_COOLDOWN_HARD"])

    LIMIT_PRI = int(CONFIG["REPEAT_LIMIT"]["PRI"])
    LIMIT_NON = int(CONFIG["REPEAT_LIMIT"]["NON"])
    LIMIT_HARD_PRI = bool(CONFIG["REPEAT_LIMIT_HARD"]["PRI"])
    LIMIT_HARD_NON = bool(CONFIG["REPEAT_LIMIT_HARD"]["NON"])

    COOLDOWN_GEO = int(CONFIG["COOLDOWN_GEO"])
    REPEAT_OVER_GEO = int(CONFIG["REPEAT_OVER_GEO"])

    PRIORITY_COVERAGE_MODE = str(CONFIG["PRIORITY_COVERAGE_MODE"]).lower()
    W = CONFIG["WEIGHTS"]
    W_DEBUG_UNASSIGNED_PRIORITY = int(
        W.get("W_DEBUG_UNASSIGNED_PRIORITY", W.get("W_DEBUG_UNASSIGNED", 0))
    )
    W_DEBUG_UNASSIGNED_NON_PRIORITY = int(
        W.get("W_DEBUG_UNASSIGNED_NON_PRIORITY", W.get("W_DEBUG_UNASSIGNED", 0))
    )

    def weight(name: str, *, default: int | None = None) -> int:
        if name in W:
            return int(W[name])
        if default is not None:
            return int(default)
        raise KeyError(f"Missing weight '{name}' in CONFIG['WEIGHTS']")

    W1_COOLDOWN = weight("W1_COOLDOWN")
    W1_REPEAT = weight("W1_REPEAT")
    W2_COOLDOWN = weight("W2_COOLDOWN")
    W2_REPEAT = weight("W2_REPEAT")
    W1_COOLDOWN_INTRA = weight("W1_COOLDOWN_INTRA", default=W1_COOLDOWN)
    W2_COOLDOWN_INTRA = weight("W2_COOLDOWN_INTRA", default=W2_COOLDOWN)
    W1_STREAK = weight("W1_STREAK")
    W2_STREAK = weight("W2_STREAK")

    W3, W4, W5 = weight("W3"), weight("W4"), weight("W5")
    W4_DPR = weight("W4_DPR", default=W4)
    W_AUTO_DAY = weight("W_AUTO_DAY")
    W_AUTO_DAY_SUNDAY = weight("W_AUTO_DAY_SUNDAY", default=W_AUTO_DAY)

    W6_OVER, W6_UNDER, T1C = weight("W6_OVER"), weight("W6_UNDER"), weight("T1C")
    T2C = weight("T2C", default=max(1, T1C // 10))
    W_PRIORITY_MISS = weight("W_PRIORITY_MISS", default=0)
    SUNDAY_TWO_DAY_SOFT = bool(CONFIG.get("SUNDAY_TWO_DAY_SOFT", False))
    TWO_DAY_SOFT_ALL = bool(CONFIG.get("TWO_DAY_SOFT_ALL", False))
    FAIR_MEAN_MULTIPLIER = float(CONFIG.get("FAIR_MEAN_MULTIPLIER", 1.0))
    FAIR_OVER_START_DELTA = int(CONFIG.get("FAIR_OVER_START_DELTA", 0))
    W_TWODAY_GENERIC = int(W.get("W_TWO_DAY_SOFT", 100_000_000_000))  # non-Sunday pairs
    W_TWODAY_SUNDAY = int(W.get("W_SUNDAY_TWO_DAY", W_TWODAY_GENERIC))  # Sunday-inclusive pairs

    comps, universe_people, used_siblingkey = load_components(args.components)
    backend = read_csv_matrix(args.backend)
    (both_people, cooldown_key, exclusions, preferred_pairs,
     top_priority_tasks, second_priority_tasks, deprioritized_pairs_map) = load_backend_roles_and_maps(backend)

    # If SiblingKey wasn't present, synthesize from backend cooldown families.
    if not used_siblingkey:
        for r in comps:
            r.sibling_key = canonical_families(r.names, cooldown_key)

    # Tag two-tier priority by task names; keep existing 'priority' meaning as TOP
    for r in comps:
        r.is_top = any(n in top_priority_tasks for n in r.names)
        r.is_second = any(n in second_priority_tasks for n in r.names)
        # Preserve prior logic that 'priority' drives cooldown/repeat as TOP-only:
        r.priority = bool(r.is_top)

    fam_reg_path = resolve_data_path(Path(getattr(args, "family_registry", Path("family_registry.json"))))
    family_registry = FamilyRegistry(fam_reg_path)
    family_registry.load()

    family_components: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def ordered_family_tokens() -> List[str]:
        order: List[str] = []
        seen: Set[str] = set()
        for comp in comps:
            fams = comp.sibling_key if comp.sibling_key else (comp.cid,)
            for fam in fams:
                fam_trim = trim(fam)
                if not fam_trim:
                    continue
                family_components[fam_trim][comp.cid].update(comp.names)
                if fam_trim in seen:
                    continue
                seen.add(fam_trim)
                order.append(fam_trim)
        return order

    family_tokens = ordered_family_tokens()
    family_registry.sync(family_tokens, family_components)
    family_labels = {fam: family_registry.label(fam) for fam in family_tokens}

    def fam_label(name: str) -> str:
        return family_labels.get(name, family_registry.label(name))

    # Banned pairs config → frozensets
    BANNED_SIBLING_PAIRS: Set[frozenset] = {frozenset(p) for p in CONFIG["BANNED_SIBLING_PAIRS"]}
    BANNED_SAME_DAY_PAIRS: Set[frozenset] = {frozenset(p) for p in CONFIG["BANNED_SAME_DAY_PAIRS"]}

    # Universes and indexes
    people     = sorted(p for p in universe_people if p)
    days_all   = sorted({r.day for r in comps})
    days_named = [d for d in days_all if d]
    weeks      = sorted({r.week_num for r in comps})

    # Index by week/day and by week
    by_week: Dict[int, List[CompRow]] = defaultdict(list)
    by_week_day: Dict[Tuple[int, str], List[CompRow]] = defaultdict(list)
    comp_by_cid: Dict[str, CompRow] = {}
    for r in comps:
        by_week[r.week_num].append(r)
        by_week_day[(r.week_num, r.day)].append(r)
        comp_by_cid[r.cid] = r

    # Names snapshot for exclusions
    comp_names: Dict[str, Set[str]] = {r.cid: set(r.names) for r in comps}

    # Snapshot original manual flags (from extractor)
    original_manual: Dict[str, bool] = {r.cid: bool(r.assigned_flag) for r in comps}
    manual_assignee: Dict[str, str] = {
        r.cid: r.assigned_to[0]
        for r in comps
        if r.assigned_flag and r.assigned_to
    }

    # Sibling groups by literal Names (used for preferred pairs / manual links)
    sib_groups_names: Dict[Tuple[str, str, Tuple[str, ...]], List[CompRow]] = defaultdict(list)
    for r in comps:
        sib_groups_names[(r.week_label, r.day, tuple(sorted(r.names)))].append(r)

    # Sibling groups (family-based) PER FAMILY TOKEN
    sib_groups_fam: Dict[Tuple[str, str, str], List[CompRow]] = defaultdict(list)
    for r in comps:
        fams = list(r.sibling_key) if r.sibling_key else ["__NONE__"]
        for fam in fams:
            fam = trim(fam)
            if fam:
                sib_groups_fam[(r.week_label, r.day, fam)].append(r)

    # Candidate sets (role-based + Both expansions) and manual flags
    base_role: Dict[str, Set[str]] = {r.cid: set([p for p in r.candidates_role if p]) for r in comps}
    base_all:  Dict[str, Set[str]] = {r.cid: set([p for p in r.candidates_all  if p]) for r in comps}
    expanded_role: Dict[str, Set[str]] = {r.cid: set(base_role[r.cid]) for r in comps}
    both_penalty_mark: Set[Tuple[str,str]] = set()

    for r in comps:
        add_both = (base_all[r.cid] - base_role[r.cid]) & both_people
        if add_both:
            for p in add_both:
                expanded_role[r.cid].add(p)
                both_penalty_mark.add((r.cid, p))

    # Sibling handling for manual “Both” (names-based pairing)
    sibling_move_links: List[Tuple[str,str,str]] = []
    for key, items in sib_groups_names.items():
        if len(items) != 2:
            continue
        A, B = sorted(items, key=lambda rr: rr.repeat)

        def activate_move(src: CompRow, dst: CompRow):
            if not src.assigned_to:
                return
            # Skip links when the destination component is already manual – there is
            # no room to "move" the person, and the link would only introduce
            # unsatisfiable exactly-one constraints for the original assignees.
            if dst.assigned_flag and dst.assigned_to:
                return
            person = src.assigned_to[0]
            if person not in both_people:
                return
            expanded_role[src.cid].update(expanded_role[dst.cid])
            expanded_role[dst.cid].update(expanded_role[src.cid])
            expanded_role[src.cid].add(person)
            expanded_role[dst.cid].add(person)
            both_penalty_mark.add((dst.cid, person))
            sibling_move_links.append((src.cid, dst.cid, person))

        if A.assigned_flag and A.assigned_to: activate_move(A, B)
        if B.assigned_flag and B.assigned_to: activate_move(B, A)

    move_people_by_cid: Dict[str, Set[str]] = defaultdict(set)
    for a, b, p in sibling_move_links:
        move_people_by_cid[a].add(p); move_people_by_cid[b].add(p)

    cand: Dict[str, List[str]] = {}
    is_manual: Dict[str, bool] = {}
    for r in comps:
        if r.cid in move_people_by_cid:
            cset = set(expanded_role[r.cid]); cset.update(r.assigned_to); cset.update(move_people_by_cid[r.cid])
            cand[r.cid] = sorted(cset) if cset else (r.assigned_to or [])
            is_manual[r.cid] = (len(cand[r.cid]) == 1)
        elif r.assigned_flag and r.assigned_to:
            cand[r.cid] = [r.assigned_to[0]]
            is_manual[r.cid] = True
        else:
            cset = expanded_role[r.cid] if expanded_role[r.cid] else base_role[r.cid]
            cand[r.cid] = sorted(cset)
            is_manual[r.cid] = (len(cand[r.cid]) == 1)

    auto_soften_cfg = CONFIG.get("AUTO_SOFTEN", {})
    softener = AutoSoftener(
        enabled=bool(auto_soften_cfg.get("ENABLED", False)),
        min_unique=int(auto_soften_cfg.get("MIN_UNIQUE_CANDIDATES", 2)),
        max_slots_ratio=float(auto_soften_cfg.get("MAX_SLOTS_PER_PERSON", 2.0)),
        relax_cooldown=bool(auto_soften_cfg.get("RELAX_COOLDOWN", True)),
        relax_repeat=bool(auto_soften_cfg.get("RELAX_REPEAT", True)),
    )
    softener.analyze(comps, cand)

    # -------------------- Prebuild x-variables --------------------
    pb = PBWriter(debug_relax=DEBUG_RELAX, W_HARD=W_HARD)

    def debug_relax_label(label: str, *, allow_relax: bool = True) -> str | None:
        return label if (DEBUG_RELAX and allow_relax) else None

    x_index: Dict[Tuple[str, str], str] = {}
    x_to_label: Dict[str, str] = {}
    for r in comps:
        for p in cand[r.cid]:
            xname = pb.new_var()
            x_index[(r.cid, p)] = xname
            x_to_label[xname] = f"x::{r.cid}::{p}"

    def xv(cid: str, person: str) -> str:
        return x_index[(cid, person)]

    # -------------------- Precompute day-level indicators --------------------
    person_day_X_all: Dict[Tuple[str,int,str], List[Tuple[str,int]]] = defaultdict(list)
    person_day_X_auto: Dict[Tuple[str,int,str], List[Tuple[str,int]]] = defaultdict(list)

    for (w, d), rows in by_week_day.items():
        for r in rows:
            tc = max(1, int(r.task_count))
            manual = is_manual[r.cid]
            for p in cand[r.cid]:
                x = xv(r.cid, p)
                person_day_X_all[(p, w, d)].append((x, tc))
                if not manual:
                    person_day_X_auto[(p, w, d)].append((x, tc))

    D_map: Dict[Tuple[str,int,str], str] = {}
    A_map: Dict[Tuple[str,int,str], str | None] = {}
    Y_map: Dict[Tuple[str,int,str], str | None] = {}
    for key, X_all in person_day_X_all.items():
        D = make_or(pb, [x for x, _ in X_all])
        D_map[key] = D
        X_auto = person_day_X_auto.get(key, [])
        A = make_or(pb, [x for x, _ in X_auto]) if X_auto else None
        A_map[key] = A
        Y_map[key] = make_and_not(pb, D, A) if D is not None else None

    # -------------------- Precompute same-day exclusion pairs --------------------
    conflict_pairs_by_day: Dict[Tuple[int,str], List[Tuple[str,str]]] = defaultdict(list)
    for (w, d), rows in by_week_day.items():
        ids = [r.cid for r in rows]
        for i in range(len(ids)):
            a = ids[i]; A = comp_names[a]
            for j in range(i + 1, len(ids)):
                b = ids[j]; B = comp_names[b]
                if any(exclusions.get(t, set()) & B for t in A):
                    conflict_pairs_by_day[(w,d)].append((a,b))

    # -------------------- Deprioritized pairs (same (week,day), soft) --------------------
    deprio_pairs_by_day: Dict[Tuple[int,str], List[Tuple[str,str]]] = defaultdict(list)
    if deprioritized_pairs_map:
        for (w, d), rows in by_week_day.items():
            ids = [r.cid for r in rows]
            for i in range(len(ids)):
                a = ids[i]; A = comp_names[a]   # set of task names on component a
                for j in range(i + 1, len(ids)):
                    b = ids[j]; B = comp_names[b]
                    # If ANY name-name combo is flagged as deprioritized, mark this component pair
                    hit = False
                    for tA in A:
                        if hit: break
                        for tB in B:
                            if (tA in deprioritized_pairs_map and tB in deprioritized_pairs_map[tA]):
                                hit = True; break
                            if (tB in deprioritized_pairs_map and tA in deprioritized_pairs_map[tB]):
                                hit = True; break
                    if hit:
                        deprio_pairs_by_day[(w, d)].append((a, b))


    # -------------------- Exactly-one per component (hard) --------------------
    component_drop_vars: Dict[str, Tuple[str, bool]] = {}
    for r in comps:
        X = [xv(r.cid, p) for p in cand[r.cid]]
        manual_orig = bool(original_manual.get(r.cid, False))
        allow_drop = DEBUG_ALLOW_UNASSIGNED and ((not manual_orig) or (not X))
        if allow_drop:
            drop_var = pb.new_var()
            component_drop_vars[r.cid] = (drop_var, manual_orig)
            x_to_label[drop_var] = f"drop::{r.cid}"
            label = f"exactly_one_or_drop::{r.cid}"
            terms = [(1, v) for v in X] + [(1, drop_var)]
            pb.add_eq(terms, 1,
                      relax_label=(label if DEBUG_RELAX else None),
                      info={"kind":"exactly_one_or_drop","cid":r.cid,"drop":drop_var})
        else:
            if not X:
                label = f"exactly_one_empty::{r.cid}"
                pb.add_eq([], 1, relax_label=label, info={"kind":"exactly_one_empty","cid":r.cid})
            else:
                label = f"exactly_one::{r.cid}"
                pb.add_eq([(1, v) for v in X], 1,
                          relax_label=(label if DEBUG_RELAX else None),
                          info={"kind":"exactly_one","cid":r.cid})

    # -------------------- Sibling move links (manual-Both unfreeze) --------------------
    for (A, B, P) in sibling_move_links:
        pb.add_eq([(1, xv(A, P)), (1, xv(B, P))], 1,
                  relax_label=(f"both_move_link::{A}::{B}::{P}" if DEBUG_RELAX else None),
                  info={"kind":"both_move_link","A":A,"B":B,"person":P})

    penalties: List[Tuple[int, str]] = []
    if DEBUG_ALLOW_UNASSIGNED:
        for cid, (drop_var, manual_orig) in component_drop_vars.items():
            if manual_orig:
                continue
            r = comp_by_cid[cid]
            weight = (
                W_DEBUG_UNASSIGNED_PRIORITY
                if r.priority
                else W_DEBUG_UNASSIGNED_NON_PRIORITY
            )
            if weight > 0:
                penalties.append((weight, drop_var))
    two_day_soft_vars: Dict[str, str] = {}

    # -------------------- Sibling anti-dup (names-based exact match) --------------------
    for key, items in sib_groups_names.items():
        if len(items) <= 1: continue
        cids = [r.cid for r in items]
        group_people = set().union(*(cand.get(cid, []) for cid in cids))
        for p in group_people:
            terms = [(1, xv(cid, p)) for cid in cids if p in cand.get(cid, [])]
            if len(terms) > 1:
                label = f"sibling_no_double::names::{'+'.join(cids)}::{p}"
                pb.add_le(terms, 1,
                          relax_label=(label if DEBUG_RELAX else None),
                          M=len(terms),
                          info={"kind":"sibling_no_double_names","group_cids":";".join(cids),"person":p})

    # -------------------- Sibling anti-dup (family-based PER FAMILY TOKEN) --------------------
    for key, items in sib_groups_fam.items():
        if len(items) <= 1: continue
        cids = [r.cid for r in items]
        group_people = set().union(*(cand.get(cid, []) for cid in cids))
        for p in group_people:
            terms = [(1, xv(cid, p)) for cid in cids if p in cand.get(cid, [])]
            if len(terms) > 1:
                label = f"sibling_no_double::family::{'+'.join(cids)}::{p}"
                pb.add_le(terms, 1,
                          relax_label=(label if DEBUG_RELAX else None),
                          M=len(terms),
                          info={"kind":"sibling_no_double_family","group_cids":";".join(cids),"person":p})
        # Banned sibling pairs inside this family-token group
        if BANNED_SIBLING_PAIRS:
            for i in range(len(cids)):
                A = cids[i]; candsA = set(cand.get(A, []))
                for j in range(i + 1, len(cids)):
                    B = cids[j]; candsB = set(cand.get(B, []))
                    if not candsA or not candsB:
                        continue
                    for pair in BANNED_SIBLING_PAIRS:
                        if len(pair) != 2: continue
                        u, v = tuple(pair)
                        if (u in candsA) and (v in candsB):
                            pb.add_le([(1, xv(A, u)), (1, xv(B, v))], 1,
                                      relax_label=(f"banned_pair::{A}::{B}::{u}|{v}" if DEBUG_RELAX else None),
                                      M=2,
                                      info={"kind":"banned_pair","cidA":A,"cidB":B,"u":u,"v":v})
                        if (v in candsA) and (u in candsB):
                            pb.add_le([(1, xv(A, v)), (1, xv(B, u))], 1,
                                      relax_label=(f"banned_pair::{A}::{B}::{v}|{u}" if DEBUG_RELAX else None),
                                      M=2,
                                      info={"kind":"banned_pair","cidA":A,"cidB":B,"u":v,"v":u})

    # -------------------- Same-day banned pairs (named days only) --------------------
    if CONFIG["BANNED_SAME_DAY_PAIRS"]:
        for w in weeks:
            for d in days_named:
                for pair in BANNED_SAME_DAY_PAIRS:
                    if len(pair) != 2: continue
                    u, v = tuple(pair)
                    Du = D_map.get((u, w, d))
                    Dv = D_map.get((v, w, d))
                    if not Du or not Dv:
                        continue
                    label = f"banned_same_day::{d}::W{w}::{u}|{v}"
                    pb.add_le([(1, Du), (1, Dv)], 1,
                              relax_label=(label if DEBUG_RELAX else None),
                              M=2,
                              info={"kind":"banned_same_day","week":str(w),"day":d,"u":u,"v":v})

    # -------------------- Two named days per week: manual vs soft --------------------
    for p in people:
        for w in weeks:
            day_info = [(d, D_map.get((p, w, d)), Y_map.get((p, w, d)), A_map.get((p, w, d))) for d in days_named]
            day_info = [(d, D, Y, A) for (d, D, Y, A) in day_info if D and Y]  # require indicators

            for i in range(len(day_info)):
                d1, D1, Y1, A1 = day_info[i]
                for j in range(i + 1, len(day_info)):
                    d2, D2, Y2, A2 = day_info[j]

                    pair_has_sunday = ("Sunday" in (d1, d2))
                    weight = W_TWODAY_SUNDAY if pair_has_sunday else W_TWODAY_GENERIC

                    # Two days taken
                    B12 = make_and(pb, D1, D2)
                    # "Both manual" indicator = Y1 & Y2  (recall Y = D & not A)
                    M12 = make_and(pb, Y1, Y2)
                    # Soft condition: two days chosen AND NOT(both manual)
                    V_soft = make_and_not(pb, B12, M12)

                    if TWO_DAY_SOFT_ALL:
                        # allow any two named days, but penalize if not both manual
                        penalties.append((weight, V_soft))
                        tag = "sunday_two_day_soft" if pair_has_sunday else "two_day_soft"
                        two_day_soft_vars[V_soft] = f"{tag}::W{w}::{p}::{d1}+{d2}"
                        continue

                    # Legacy: only Sunday-inclusive pairs are softened; others remain HARD
                    if SUNDAY_TWO_DAY_SOFT and pair_has_sunday:
                        penalties.append((weight, V_soft))
                        two_day_soft_vars[V_soft] = f"sunday_two_day_soft::W{w}::{p}::{d1}+{d2}"
                        continue

                    # HARD: two named days require at least one side to be fully manual
                    pb.add_le(
                        [(1, D1), (1, D2), (-1, Y1), (-1, Y2)],
                        1,
                        relax_label=(f"two_days_need_manual::W{w}::{p}::{d1}+{d2}" if DEBUG_RELAX else None),
                        M=3,
                        info={"kind": "two_days", "week": str(w), "person": p, "d1": d1, "d2": d2}
                    )

    # -------------------- Exclusions (same (week,day)) --------------------
    for p in people:
        for (w, d), pairs in conflict_pairs_by_day.items():
            for (a, b) in pairs:
                if p not in cand.get(a, []) or p not in cand.get(b, []):
                    continue

                # If both components are already fixed to their manual assignee, and that
                # assignee is this person, allow the conflicting manual pair to stand.
                if original_manual.get(a, False) and original_manual.get(b, False):
                    ma = manual_assignee.get(a)
                    mb = manual_assignee.get(b)
                    if ma and mb and ma == mb == p:
                        continue

                pb.add_le([(1, xv(a, p)), (1, xv(b, p))], 1,
                          relax_label=(f"exclusion::W{w}::{d}::{p}::{a}::{b}" if DEBUG_RELAX else None),
                          M=1,
                          info={"kind":"exclusion","week":str(w),"day":d,"person":p,"a":a,"b":b})

    # -------------------- AUTO-day soft penalties (weighted by Task Count) --------------------
    auto_day_min_vars: Dict[str, str] = {}
    auto_day_min_sunday_vars: Dict[str, str] = {}
    for key, X_all in person_day_X_all.items():
        p, w, d = key
        A = A_map.get(key)
        if not X_all or A is None:
            continue

        # Skip unnamed days entirely; the AUTO-day minimum applies only to named days.
        if not trim(d):
            continue

        v_short = pb.new_var()
        pb.add_ge([(tc, xi) for (xi, tc) in X_all] + [(-2, A), (2, v_short)], 0)

        is_sunday = trim(d).lower() == "sunday"
        if is_sunday:
            penalties.append((W_AUTO_DAY_SUNDAY, v_short))
            auto_day_min_sunday_vars[v_short] = f"auto_day_under::sunday::person={p}::week={w}::day={d or 'UNNAMED'}"
        else:
            penalties.append((W_AUTO_DAY, v_short))
            auto_day_min_vars[v_short] = f"auto_day_under::weekday::person={p}::week={w}::day={d or 'UNNAMED'}"

    # -------------------- Prev-week cooldown (aggregated per family) --------------------
    fam_index: Dict[Tuple[int, str, str], Dict[str, List[str]]] = defaultdict(lambda: {"any":[], "pri":[], "auto":[]})
    all_fams: Set[str] = set()
    for r in comps:
        fams = list(r.sibling_key) if r.sibling_key else [r.cid]
        fams = [trim(f) for f in fams if trim(f)]
        for fam in fams:
            all_fams.add(fam)
            for p in cand.get(r.cid, []):
                xv_ = xv(r.cid, p)
                bucket = fam_index[(r.week_num, fam, p)]
                bucket["any"].append(xv_)
                if r.priority: bucket["pri"].append(xv_)
                if not original_manual.get(r.cid, False): bucket["auto"].append(xv_)

    def fam_indicators(week: int, fam: str, person: str):
        bucket = fam_index.get((week, trim(fam), person))
        if not bucket:
            return None, None, None
        Any = make_or(pb, bucket["any"])
        Pri = make_or(pb, bucket["pri"]) if bucket["pri"] else None
        Auto = make_or(pb, bucket["auto"]) if bucket["auto"] else None
        return Any, Pri, Auto

    vprev_pri_by_pfam_week: Dict[Tuple[str, str, int], str] = {}
    vprev_non_by_pfam_week: Dict[Tuple[str, str, int], str] = {}

    cooldown_viols_pri: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    cooldown_viols_non: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    # Richer gate info for debugging/inspection
    cooldown_gate_info: Dict[str, Dict[str, str | int]] = {}

    for p in people:
        for w in weeks:
            prev_w = w - 1
            if prev_w not in by_week:
                continue
            for fam in all_fams:
                fam_trim = trim(fam)
                if not fam_trim:
                    continue
                if softener.should_skip("cooldown", fam_trim):
                    continue
                P_any, P_pri, P_auto = fam_indicators(prev_w, fam_trim, p)
                C_any, C_pri, C_auto = fam_indicators(w, fam_trim, p)
                if P_any is None or C_any is None:
                    continue

                # Both weeks have something in this family for this person
                T_both_weeks = make_and(pb, P_any, C_any)

                # Require that at least one of the two weeks is AUTO (gate)
                AutoEither = make_or(pb, [P_auto, C_auto])
                if AutoEither is None:
                    # No AUTO on either side => cooldown cannot fire; skip entirely
                    continue

                # Priority presence (either week)
                PriEither = make_or(pb, [P_pri, C_pri])

                # ----- PRIORITY cooldown -----
                if PriEither is not None:
                    gate = make_and(pb, T_both_weeks, AutoEither)
                    V_pri = make_and(pb, gate, PriEither)
                    if PRIORITY_COOLDOWN_HARD:
                        pb.add_le(
                            [(1, V_pri)],
                            0,
                            relax_label=debug_relax_label(
                                f"cooldown_prev_hard_PRI::fam={fam_trim}::W{w}::{p}", allow_relax=False
                            ),
                            M=1,
                            info={"kind": "cooldown_prev_hard_PRI", "week": str(w), "person": p, "family": fam_trim},
                        )
                    else:
                        cooldown_viols_pri[(p, fam_trim)].append(V_pri)
                        cooldown_gate_info[V_pri] = {
                            "person": p,
                            "family": fam_trim,
                            "family_label": fam_label(fam_trim),
                            "week": w,
                            "gate": "either",
                        }

                    vprev_pri_by_pfam_week[(p, fam_trim, w)] = V_pri

                # ----- NON-PRIORITY cooldown -----
                # Non-priority triggers when both weeks assigned & AUTO present, but not priority on either.
                T_non = T_both_weeks  # already AND(P_any, C_any)
                if PriEither is None:
                    V_non = make_and(pb, T_non, AutoEither)
                else:
                    # AND(T_non, AutoEither, NOT(PriEither))
                    not_pri = make_and_not(pb, T_non, PriEither)  # gives T_non if PriEither=0 else 0
                    V_non = make_and(pb, not_pri, AutoEither)

                if NONPRIORITY_COOLDOWN_HARD:
                    pb.add_le(
                        [(1, V_non)],
                        0,
                        relax_label=debug_relax_label(
                            f"cooldown_prev_hard_NON::fam={fam_trim}::W{w}::{p}", allow_relax=False
                        ),
                        M=1,
                        info={"kind": "cooldown_prev_hard_NON", "week": str(w), "person": p, "family": fam_trim},
                    )
                else:
                    cooldown_viols_non[(p, fam_trim)].append(V_non)
                    cooldown_gate_info[V_non] = {
                        "person": p,
                        "family": fam_trim,
                        "family_label": fam_label(fam_trim),
                        "week": w,
                        "gate": "either",
                    }

                vprev_non_by_pfam_week[(p, fam_trim, w)] = V_non

    # Repeat discouragers: STREAK only (GAP removed)
    vprev_streak_vars: Dict[str, str] = {}
    def add_streak_penalties(by_map: Dict[Tuple[str, str, int], str], weight_streak: int, tag: str):
        groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
        for (person, fam, w), var in by_map.items():
            groups[(person, trim(fam))].append((w, var))
        for (person, fam), items in groups.items():
            items.sort(key=lambda t: t[0])
            for i in range(len(items)-1):
                w1, v1 = items[i]
                w2, v2 = items[i+1]
                if w2 == w1 + 1:
                    z = make_and(pb, v1, v2)
                    penalties.append((weight_streak, z))
                    vprev_streak_vars[z] = f"vprev::REPEAT::STREAK::{tag}::fam={fam}::W{w1}->{w2}::{person}"

    add_streak_penalties(vprev_pri_by_pfam_week, W1_STREAK, "PRI")
    add_streak_penalties(vprev_non_by_pfam_week, W2_STREAK, "NON")

    # Geometric scaling ladders for cooldown violation counts
    cooldown_pri_ladder_vars: Dict[str, str] = {}
    cooldown_non_ladder_vars: Dict[str, str] = {}

    for (person, fam), viols in cooldown_viols_pri.items():
        U = len(viols)
        if U == 0:
            continue
        sum_terms = [(1, v) for v in viols]
        for t in range(1, U + 1):
            b_t = pb.new_var()
            pb.add_le(sum_terms + [(-U, b_t)], t - 1)
            penalties.append((W1_COOLDOWN * (CONFIG["COOLDOWN_GEO"] ** (t - 1)), b_t))
            cooldown_pri_ladder_vars[b_t] = f"cooldown_geo::PRI::person={person}::family={fam}::t={t}::{U}"

    for (person, fam), viols in cooldown_viols_non.items():
        U = len(viols)
        if U == 0:
            continue
        sum_terms = [(1, v) for v in viols]
        for t in range(1, U + 1):
            b_t = pb.new_var()
            pb.add_le(sum_terms + [(-U, b_t)], t - 1)
            penalties.append((W2_COOLDOWN * (CONFIG["COOLDOWN_GEO"] ** (t - 1)), b_t))
            cooldown_non_ladder_vars[b_t] = f"cooldown_geo::NON::person={person}::family={fam}::t={t}::{U}"

    # -------------------- INTRA-week cooldown (same week, per family) --------------------
    # Build day-granular family buckets: (week, family, person, day) -> any/pri/auto x-vars
    fam_day_index: Dict[Tuple[int, str, str, str], Dict[str, List[str]]] = \
        defaultdict(lambda: {"any": [], "pri": [], "auto": []})

    for r in comps:
        fams = list(r.sibling_key) if r.sibling_key else [r.cid]
        fams = [trim(f) for f in fams if trim(f)]
        for fam in fams:
            for p in cand.get(r.cid, []):
                xvar = xv(r.cid, p)
                bucket = fam_day_index[(r.week_num, fam, p, r.day)]
                bucket["any"].append(xvar)
                if r.priority:
                    bucket["pri"].append(xvar)
                if not original_manual.get(r.cid, False):
                    bucket["auto"].append(xvar)

    # Geometric ladders for "days used in same week within same family" above 1 day
    for p in people:
        for w in weeks:
            for fam in all_fams:
                fam_trim = trim(fam)
                if not fam_trim:
                    continue
                if softener.should_skip("cooldown", fam_trim):
                    continue
                day_terms_any: List[str] = []
                day_terms_pri: List[str] = []
                day_terms_auto: List[str] = []

                for d in days_named:
                    b = fam_day_index.get((w, fam_trim, p, d))
                    if not b:
                        continue
                    D_any = make_or(pb, b["any"])
                    if not D_any:
                        continue
                    D_pri = make_or(pb, b["pri"]) if b["pri"] else None
                    D_auto = make_or(pb, b["auto"]) if b["auto"] else None

                    day_terms_any.append(D_any)
                    if D_pri is not None:  day_terms_pri.append(D_pri)
                    if D_auto is not None: day_terms_auto.append(D_auto)

                # Need at least two distinct days in this (w,fam,p) to consider cooldown
                if len(day_terms_any) < 2:
                    continue

                # Gates: at least one of those days must be AUTO
                AutoAny = make_or(pb, day_terms_auto)
                if AutoAny is None:
                    continue

                PriAny = make_or(pb, day_terms_pri)  # None => strictly non-priority across those days
                U = len(day_terms_any)
                sum_terms = [(1, z) for z in day_terms_any]

                # Ladder over day-count: t = 2..U (each extra day beyond the first)
                for t in range(2, U + 1):
                    b_t = pb.new_var()
                    # sum(day_used) - U * b_t <= t - 1   <=>   sum(day_used) <= (t - 1) + U * b_t
                    pb.add_le(sum_terms + [(-U, b_t)], t - 1)

                    # Gate by having at least one AUTO day in the set
                    gated = make_and(pb, b_t, AutoAny)

                    # Weight split: if ANY day is priority, count under PRI; else NON
                    step = t - 2  # first extra day uses base^(0)
                    if PriAny is not None:
                        z = make_and(pb, gated, PriAny)
                        penalties.append((W1_COOLDOWN_INTRA * (COOLDOWN_GEO ** step), z))
                        cooldown_pri_ladder_vars[z] = (
                            f"cooldown_geo::PRI::INTRA_WEEK::person={p}::family={fam_trim}::days_used={t}::W{w}"
                        )
                        cooldown_gate_info[z] = {
                            "person": p,
                            "family": fam_trim,
                            "family_label": fam_label(fam_trim),
                            "week": w,
                            "gate": "intra_week_auto",
                        }
                    else:
                        z = gated  # no need to AND with not-priority; PriAny is None already
                        penalties.append((W2_COOLDOWN_INTRA * (COOLDOWN_GEO ** step), z))
                        cooldown_non_ladder_vars[z] = (
                            f"cooldown_geo::NON::INTRA_WEEK::person={p}::family={fam_trim}::days_used={t}::W{w}"
                        )
                        cooldown_gate_info[z] = {
                            "person": p,
                            "family": fam_trim,
                            "family_label": fam_label(fam_trim),
                            "week": w,
                            "gate": "intra_week_auto",
                        }

    # ---------- Tier 3 ----------
    for key, X_all in person_day_X_all.items():
        p, w, d = key
        A = A_map.get(key)
        Y = Y_map.get(key)
        if not X_all or Y is None or A is not None or len(X_all) < 2:
            continue
        v_fill = pb.new_var()
        pb.add_ge([(1, xi) for (xi, _tc) in X_all] + [(-2, Y), (1, v_fill)], 0)
        penalties.append((W3, v_fill))

    # ---------- Tier 4 ----------
    both_fallback_vars: Dict[str, str] = {}
    for (cid, person) in both_penalty_mark:
        if (cid, person) not in x_index:
            continue
        xvar = xv(cid, person)
        penalties.append((W4, xvar))
        both_fallback_vars[xvar] = f"both_fallback::{cid}::{person}"

    # -------------------- Deprioritized pair penalties --------------------
    deprioritized_pair_vars: Dict[str, str] = {}
    if deprio_pairs_by_day:
        for p in people:
            for (w, d), pairs in deprio_pairs_by_day.items():
                for (a, b) in pairs:
                    if p not in cand.get(a, []) or p not in cand.get(b, []):
                        continue
                    z = make_and(pb, xv(a, p), xv(b, p))
                    penalties.append((W4_DPR, z))
                    deprioritized_pair_vars[z] = (
                        f"deprioritized_pair::{d}::W{w}::{p}::{a}::{b}"
                    )


    # ---------- Tier 5 ----------
    preferred_miss_vars: Dict[str, str] = {}
    for key, items in sib_groups_names.items():
        if len(items) != 2:
            continue
        A, B = sorted(items, key=lambda rr: rr.repeat)
        candsA = cand.get(A.cid, []); candsB = cand.get(B.cid, [])
        if not candsA or not candsB:
            continue

        disj: List[str] = []
        unordered_seen: Set[frozenset] = set()
        group_people = set(candsA) | set(candsB)
        for u in group_people:
            for v in preferred_pairs.get(u, set()):
                if u == v or v not in group_people:
                    continue
                pair = frozenset((u, v))
                has_orient = False
                if (u in candsA) and (v in candsB):
                    disj.append(make_and(pb, xv(A.cid, u), xv(B.cid, v))); has_orient = True
                if (v in candsA) and (u in candsB):
                    disj.append(make_and(pb, xv(A.cid, v), xv(B.cid, u))); has_orient = True
                if has_orient:
                    unordered_seen.add(pair)

        if not disj:
            continue
        PrefOK = make_or(pb, disj)
        z = pb.new_var()  # 1 - PrefOK
        pb.add_eq([(1, z), (1, PrefOK)], 1)
        K = len(unordered_seen)
        penalties.append((W5 * K, z))
        group_label = f"{A.week_label}/{A.day}/" + "+".join(sorted(A.names))
        preferred_miss_vars[z] = f"preferred_pair_missed::group={group_label}::K={K}"

    # ---------- Tier 6 (UPDATED): Across-horizon fairness (global, non-linear) ----------
    total_tc = sum(max(1, int(r.task_count)) for r in comps)
    base_people = max(1, len(people))
    # base mean (ceil)
    mean_base = (total_tc + base_people - 1) // base_people
    # scaled mean (ceil again, >=1)
    mean_scaled = max(1, int((mean_base * FAIR_MEAN_MULTIPLIER + 0.9999999)))
    # final target for building ladders
    mean_target = max(1, mean_scaled + FAIR_OVER_START_DELTA)
    fairness_avail_cfg = CONFIG["WEIGHTS"].get("FAIRNESS_AVAILABILITY", {})
    fairness_avail_enabled = bool(fairness_avail_cfg.get("ENABLED"))
    fairness_reference = fairness_avail_cfg.get("REFERENCE", "auto").lower()
    fairness_min_ratio = float(fairness_avail_cfg.get("MIN_RATIO", 0.25))
    fairness_max_ratio = float(fairness_avail_cfg.get("MAX_RATIO", 2.0))
    fairness_power = float(fairness_avail_cfg.get("POWER", 1.0))

    candidate_effort_total: Dict[str, int] = defaultdict(int)
    candidate_effort_auto: Dict[str, int] = defaultdict(int)
    component_effort_units: Dict[str, int] = {}
    # Track per-person/day totals so the effort-floor eligibility can respect the
    # AUTO-day rule (needs ≥2 task_count on any day that includes AUTO).
    person_day_taskcount: Dict[Tuple[str, int, str], int] = defaultdict(int)
    person_day_has_auto: Dict[Tuple[str, int, str], bool] = defaultdict(bool)
    for r in comps:
        tc = max(1, int(r.task_count))
        is_manual_flag = bool(original_manual.get(r.cid, False))
        for p in cand.get(r.cid, []):
            candidate_effort_total[p] += tc
            if not is_manual_flag:
                candidate_effort_auto[p] += tc
            day_key = (p, r.week_num, trim(getattr(r, "day", "")))
            person_day_taskcount[day_key] += tc
            if not is_manual_flag:
                person_day_has_auto[day_key] = True

        effort_units = max(1, int(math.ceil(getattr(r, "total_effort", tc))))
        component_effort_units[r.cid] = effort_units

    person_effort_terms: Dict[str, List[Tuple[int, str]]] = {}
    person_effort_caps: Dict[str, int] = {}
    person_effort_terms_effort: Dict[str, List[Tuple[int, str]]] = {}
    person_effort_caps_effort: Dict[str, int] = {}
    effort_floor_attainable: Dict[str, int] = {}
    effort_floor_auto_blocked: Dict[str, Dict[Tuple[int, str], Dict[str, int]]] = {}
    effort_floor_blocked_cids: Dict[str, Set[str]] = defaultdict(set)
    for p in people:
        terms_p: List[Tuple[int, str]] = []
        terms_effort: List[Tuple[int, str]] = []
        U_p = 0
        U_effort = 0
        attainable_groups: Dict[Tuple[str, str], int] = {}
        for r in comps:
            if p in cand.get(r.cid, []):
                tc = max(1, int(r.task_count))
                terms_p.append((tc, xv(r.cid, p)))
                U_p += tc

                effort_units = max(1, int(math.ceil(getattr(r, "total_effort", tc))))
                # Respect AUTO-day minimums when building effort-floor capacity.
                week_key = int(getattr(r, "week_num", 0) or 0)
                day_key = trim(getattr(r, "day", ""))
                pd_key = (p, week_key, day_key)
                auto_for_day = person_day_has_auto.get(pd_key, False)
                day_capacity = person_day_taskcount.get(pd_key, 0)
                auto_blocked = (auto_for_day and day_capacity < 2 and not original_manual.get(r.cid, False))
                if auto_blocked:
                    blocked_key = f"W{week_key}:{day_key or 'UNSPEC'}"
                    effort_floor_auto_blocked.setdefault(p, {})[blocked_key] = {
                        "total_task_count": day_capacity,
                        "component": r.cid,
                        "effort_units": effort_units,
                        "week": week_key,
                        "day": day_key,
                    }

                terms_effort.append((effort_units, xv(r.cid, p)))
                U_effort += effort_units

                week_key = trim(getattr(r, "week", ""))
                fam_tokens = tuple(r.sibling_key) if r.sibling_key else (r.cid,)
                fam_tokens = tuple(trim(f) for f in fam_tokens if trim(f))
                if not fam_tokens:
                    fam_tokens = (r.cid,)
                for fam in fam_tokens:
                    gkey = (week_key, fam)
                    attainable_groups[gkey] = max(attainable_groups.get(gkey, 0), effort_units)
        if terms_p and U_p > 0:
            person_effort_terms[p] = terms_p
            person_effort_caps[p] = U_p
        if terms_effort and U_effort > 0:
            person_effort_terms_effort[p] = terms_effort
            person_effort_caps_effort[p] = U_effort
            effort_floor_attainable[p] = sum(attainable_groups.values())

    if fairness_reference not in {"auto", "all"}:
        fairness_reference = "auto"
    fairness_counts_map = candidate_effort_auto if fairness_reference == "auto" else candidate_effort_total
    fairness_counts_mean = max(
        1.0,
        (sum(fairness_counts_map.get(p, 0) for p in people) / len(people)) if people else 1.0,
    )

    fairness_targets: Dict[str, int] = {}
    fairness_target_notes: Dict[str, Dict[str, float | int]] = {}

    def log_thresholds_from(start: int, U: int, base: int) -> List[int]:
        """Monotone increasing thresholds starting from 'start', doubling by 'base' (>=2), capped at U."""
        if start > U:
            return []
        ts = [max(1, start)]
        t = max(start + 1, start * base)
        while t <= U:
            ts.append(t)
            t = max(t + 1, t * base)
        return ts

    for p, terms_p in person_effort_terms.items():
        U_p = person_effort_caps.get(p, 0)
        if fairness_avail_enabled:
            raw_slots = fairness_counts_map.get(p, 0)
            ratio = raw_slots / fairness_counts_mean if fairness_counts_mean > 0 else 1.0
            ratio = max(fairness_min_ratio, min(fairness_max_ratio, ratio))
            scale = ratio ** fairness_power
            person_target = max(1, int(math.ceil(mean_target * scale - 1e-9)))
        else:
            raw_slots = fairness_counts_map.get(p, 0)
            ratio = 1.0
            scale = 1.0
            person_target = mean_target

        fairness_targets[p] = person_target
        fairness_target_notes[p] = {
            "raw_slots": raw_slots,
            "ratio": ratio,
            "scale": scale,
            "target": person_target,
        }

        # -------- OVER-load ladder
        over_thresholds = log_thresholds_from(person_target + 1, U_p, REPEAT_OVER_GEO)
        for idx, t in enumerate(over_thresholds, start=1):
            b_over = pb.new_var()
            pb.add_le(terms_p + [(-U_p, b_over)], t - 1)
            penalties.append((W6_OVER * (REPEAT_OVER_GEO ** (idx - 1)), b_over))

        # -------- UNDER-load ladder
        under_candidates = sorted({
            max(0, (person_target * 1) // 2),
            max(0, (person_target * 3) // 4),
            max(0, person_target - 1),
        })
        under_thresholds = [t for t in under_candidates if 0 < t <= person_target - 1]
        for idx, t in enumerate(under_thresholds, start=1):
            b_under = pb.new_var()
            # S_p + t * b_under >= t
            pb.add_ge(terms_p + [(t, b_under)], t)
            penalties.append((W6_UNDER * (REPEAT_OVER_GEO ** (idx - 1)), b_under))

    # ---------- Effort floor (eligible people only) ----------
    effort_floor_vars: Dict[str, str] = {}
    effort_floor_notes: Dict[str, int | str] = {}
    effort_floor_feasible = True
    effort_floor_hard_applied = False
    EFFORT_FLOOR_TARGET = int(CONFIG.get("EFFORT_FLOOR_TARGET", 0) or 0)
    EFFORT_FLOOR_HARD = bool(CONFIG.get("EFFORT_FLOOR_HARD", False))
    W_EFFORT_FLOOR = int(CONFIG.get("WEIGHTS", {}).get("W_EFFORT_FLOOR", 0))
    effort_floor_eligible: List[str] = []
    eligible_terms: List[Tuple[str, List[Tuple[int, str]], int]] = []
    if EFFORT_FLOOR_TARGET > 0:
        for p, terms_p in person_effort_terms_effort.items():
            U_p = person_effort_caps_effort.get(p, 0)
            attainable = effort_floor_attainable.get(p, 0)
            if U_p < EFFORT_FLOOR_TARGET or attainable < EFFORT_FLOOR_TARGET:
                continue
            effort_floor_eligible.append(p)
            eligible_terms.append((p, terms_p, U_p))

        eligible_set = set(effort_floor_eligible)
        effort_floor_supply_effort = sum(
            component_effort_units.get(cid, 0)
            for cid in component_effort_units
            if eligible_set.intersection(cand.get(cid, []))
        )
        effort_floor_supply_capped = sum(
            min(component_effort_units.get(cid, 0), EFFORT_FLOOR_TARGET)
            for cid in component_effort_units
            if eligible_set.intersection(cand.get(cid, []))
        )
        effort_floor_demand = EFFORT_FLOOR_TARGET * len(effort_floor_eligible)
        effort_floor_notes = {
            "demand": effort_floor_demand,
            "supply_effort": effort_floor_supply_effort,
            "supply_capped": effort_floor_supply_capped,
            "attainable_caps": {p: effort_floor_attainable.get(p, 0) for p in people},
        }
        if effort_floor_auto_blocked:
            effort_floor_notes["auto_day_blocked"] = effort_floor_auto_blocked
        if not effort_floor_eligible:
            effort_floor_feasible = False
            effort_floor_notes.setdefault("reason", "no_eligible_people")
        if effort_floor_eligible:
            effort_floor_notes["eligible_attainable_min"] = min(
                effort_floor_attainable.get(p, 0) for p in effort_floor_eligible
            )
            effort_floor_notes["eligible_attainable_max"] = max(
                effort_floor_attainable.get(p, 0) for p in effort_floor_eligible
            )
        if len(effort_floor_eligible) < len(person_effort_terms_effort):
            missing = {
                p: effort_floor_attainable.get(p, 0)
                for p in person_effort_terms_effort
                if p not in effort_floor_eligible
            }
            effort_floor_notes["ineligible_by_attainable"] = missing
        if effort_floor_demand > effort_floor_supply_effort:
            effort_floor_feasible = False
            effort_floor_notes["reason"] = "insufficient_global_effort"
        elif effort_floor_demand > effort_floor_supply_capped:
            effort_floor_feasible = False
            effort_floor_notes["reason"] = "insufficient_slot_capacity"

        if effort_floor_feasible and effort_floor_eligible:
            probe_ok, probe_notes = _greedy_effort_floor_probe(
                target=EFFORT_FLOOR_TARGET,
                eligible_people=effort_floor_eligible,
                comps=comps,
                candidates=cand,
                original_manual=original_manual,
                blocked=effort_floor_blocked_cids,
            )
            effort_floor_notes["feasibility_probe"] = probe_notes
            if not probe_ok:
                effort_floor_feasible = False
                effort_floor_notes["reason"] = "feasibility_probe_failed"

        if effort_floor_feasible:
            for p, terms_p, U_p in eligible_terms:
                under_floor = pb.new_var()
                pb.add_ge(
                    terms_p + [(EFFORT_FLOOR_TARGET, under_floor)],
                    EFFORT_FLOOR_TARGET,
                    info={"kind": "effort_floor_soft", "person": p, "target": EFFORT_FLOOR_TARGET, "cap": U_p},
                )
                if W_EFFORT_FLOOR != 0:
                    penalties.append((W_EFFORT_FLOOR, under_floor))
                    effort_floor_vars[under_floor] = (
                        f"effort_floor_under::person={p}::target={EFFORT_FLOOR_TARGET}::capacity={U_p}"
                    )

                if EFFORT_FLOOR_HARD:
                    pb.add_ge(
                        terms_p,
                        EFFORT_FLOOR_TARGET,
                        relax_label=debug_relax_label(f"effort_floor_hard::{p}", allow_relax=False),
                        info={"kind": "effort_floor_hard", "person": p, "target": EFFORT_FLOOR_TARGET, "cap": U_p},
                    )
                    effort_floor_hard_applied = True

    # ---------- Per-person × family repeat limits (geometric overage) ----------
    repeat_limit_pri_vars: Dict[str, str] = {}
    repeat_limit_non_vars: Dict[str, str] = {}

    # Collect family tokens per component; if none, fall back to the component id
    comp_fams: Dict[str, Tuple[str, ...]] = {}
    fam_tokens: Set[str] = set()
    for r in comps:
        fams = tuple(r.sibling_key) if r.sibling_key else (r.cid,)
        fams = tuple(trim(f) for f in fams if trim(f))
        comp_fams[r.cid] = fams
        fam_tokens.update(fams)

    for p in people:
        for fam in sorted(fam_tokens):
            fam_trim = trim(fam)
            if not fam_trim:
                continue
            if softener.should_skip("repeat", fam_trim):
                continue
            # All PRIORITY assignments for (person p, family fam) — manual + auto
            pri_terms_pf = [
                (1, xv(r.cid, p))
                for r in comps
                if r.priority and (p in cand.get(r.cid, [])) and (fam_trim in comp_fams.get(r.cid, ()))
            ]

            # Indicator that (p,fam) has at least one AUTO PRIORITY anywhere in this family (for gating)
            auto_pri_vars_pf = [
                xv(r.cid, p)
                for r in comps
                if r.priority
                   and (p in cand.get(r.cid, []))
                   and (fam_trim in comp_fams.get(r.cid, ()))
                   and (not original_manual.get(r.cid, False))
            ]
            AutoPriAny_pf = make_or(pb, auto_pri_vars_pf) if auto_pri_vars_pf else None

            # All NON-PRIORITY assignments for (person p, family fam)
            non_terms_pf = [
                (1, xv(r.cid, p))
                for r in comps
                if (not r.priority) and (p in cand.get(r.cid, [])) and (fam_trim in comp_fams.get(r.cid, ()))
            ]

            # ----- PRIORITY -----
            if pri_terms_pf:
                if LIMIT_HARD_PRI:
                    pb.add_le(
                        pri_terms_pf,
                        LIMIT_PRI,
                        relax_label=debug_relax_label(
                            f"repeat_limit_PRI_hard::{p}::fam={fam_trim}", allow_relax=False
                        ),
                        M=len(pri_terms_pf),
                        info={
                            "kind": "repeat_limit_PRI_hard",
                            "person": p,
                            "family": fam_trim,
                            "limit": str(LIMIT_PRI),
                        },
                    )
                else:
                    U = len(pri_terms_pf)
                    for t in range(1, U + 1):
                        b_t = pb.new_var()
                        pb.add_le(pri_terms_pf + [(-U, b_t)], t - 1)
                        # Soft over-limit penalties only when there is at least one AUTO-PRI in this family
                        if t > LIMIT_PRI and AutoPriAny_pf is not None:
                            over = t - LIMIT_PRI
                            g_t = make_and(pb, b_t, AutoPriAny_pf)
                            penalties.append((W1_REPEAT * (REPEAT_OVER_GEO ** (over - 1)), g_t))
                            repeat_limit_pri_vars[g_t] = (
                                f"repeat_over_geo::PRI::person={p}::family={fam_trim}::t={t}::limit={LIMIT_PRI}"
                            )

            # ----- NON-PRIORITY -----
            if non_terms_pf:
                auto_non_vars_pf = [
                    xv(r.cid, p)
                    for r in comps
                    if (not r.priority)
                       and (p in cand.get(r.cid, []))
                       and (fam_trim in comp_fams.get(r.cid, ()))
                       and (not original_manual.get(r.cid, False))
                ]
                AutoNonAny_pf = make_or(pb, auto_non_vars_pf) if auto_non_vars_pf else None

                if LIMIT_HARD_NON:
                    pb.add_le(
                        non_terms_pf,
                        LIMIT_NON,
                        relax_label=debug_relax_label(
                            f"repeat_limit_NON_hard::{p}::fam={fam_trim}", allow_relax=False
                        ),
                        M=len(non_terms_pf),
                        info={
                            "kind": "repeat_limit_NON_hard",
                            "person": p,
                            "family": fam_trim,
                            "limit": str(LIMIT_NON),
                        },
                    )
                else:
                    U = len(non_terms_pf)
                    for t in range(1, U + 1):
                        b_t = pb.new_var()
                        pb.add_le(non_terms_pf + [(-U, b_t)], t - 1)
                        # Soft over-limit penalties only when there is at least one AUTO-NON in this family
                        if t > LIMIT_NON and AutoNonAny_pf is not None:
                            over = t - LIMIT_NON
                            g_t = make_and(pb, b_t, AutoNonAny_pf)
                            penalties.append((W2_REPEAT * (REPEAT_OVER_GEO ** (over - 1)), g_t))
                            repeat_limit_non_vars[g_t] = (
                                f"repeat_over_geo::NON::person={p}::family={fam_trim}::t={t}::limit={LIMIT_NON}"
                            )

    # ---------- Priority coverage (GLOBAL / FAMILY), two tiers ----------
    priority_coverage_vars_top: Dict[str, str] = {}
    priority_coverage_vars_second: Dict[str, str] = {}
    priority_required_vars: Dict[str, str] = {}

    top_any_by_person: Dict[str, str] = {}
    top_miss_by_person: Dict[str, str] = {}
    second_any_by_person: Dict[str, str] = {}
    for p in people:
        x_top = [xv(r.cid, p) for r in comps if r.is_top and p in cand.get(r.cid, [])]
        if x_top:
            TopAny = make_or(pb, x_top)
            z_miss = pb.new_var()
            pb.add_eq([(1, z_miss), (1, TopAny)], 1)
            top_any_by_person[p] = TopAny
            top_miss_by_person[p] = z_miss
            priority_required_vars[z_miss] = f"priority_required::tier=TOP::person={p}"
        else:
            x_second = [xv(r.cid, p) for r in comps if r.is_second and p in cand.get(r.cid, [])]
            if not x_second:
                continue
            SecondAny = make_or(pb, x_second)
            second_any_by_person[p] = SecondAny

    if PRIORITY_COVERAGE_MODE == "global":
        # TOP (independent)
        top_weight = W_PRIORITY_MISS if W_PRIORITY_MISS > 0 else T1C
        for p in sorted(top_miss_by_person):
            z = top_miss_by_person[p]
            penalties.append((top_weight, z))
            priority_coverage_vars_top[z] = f"priority_coverage_TOP::GLOBAL::person={p}"

        # SECOND (only when no TOP eligibility at all)
        second_weight = W_PRIORITY_MISS if W_PRIORITY_MISS > 0 else T2C
        for p in sorted(second_any_by_person):
            SecondAny = second_any_by_person[p]
            z2 = pb.new_var()
            pb.add_eq([(1, z2), (1, SecondAny)], 1)
            penalties.append((second_weight, z2))
            priority_coverage_vars_second[z2] = f"priority_coverage_SECOND::GLOBAL::person={p}"
            priority_required_vars[z2] = f"priority_required::tier=SECOND::person={p}"

    else:  # family mode
        fam_tokens2: Set[str] = set()
        comp_fams2: Dict[str, Tuple[str, ...]] = {}
        for r in comps:
            fams = tuple(r.sibling_key) if r.sibling_key else (r.cid,)
            comp_fams2[r.cid] = fams
            fam_tokens2.update([trim(f) for f in fams if trim(f)])

        # TOP (independent)
        for p in people:
            for fam in sorted(fam_tokens2):
                x_top_pf = [xv(r.cid, p)
                            for r in comps
                            if r.is_top and (fam in comp_fams2.get(r.cid, ())) and (p in cand.get(r.cid, []))]
                if x_top_pf:
                    TopAny_pf = make_or(pb, x_top_pf)
                    z = pb.new_var()
                    pb.add_eq([(1, z), (1, TopAny_pf)], 1)
                    penalties.append((W_PRIORITY_MISS if W_PRIORITY_MISS > 0 else T1C, z))
                    priority_coverage_vars_top[z] = f"priority_coverage_TOP::FAMILY::person={p}::family={fam}"
                else:
                    x_second_pf = [xv(r.cid, p)
                                   for r in comps
                                   if r.is_second and (fam in comp_fams2.get(r.cid, ())) and (p in cand.get(r.cid, []))]
                    if not x_second_pf:
                        continue  # not eligible for second in this family
                    SecondAny_pf = make_or(pb, x_second_pf)
                    z2 = pb.new_var()
                    pb.add_eq([(1, z2), (1, SecondAny_pf)], 1)
                    penalties.append((W_PRIORITY_MISS if W_PRIORITY_MISS > 0 else T2C, z2))
                    priority_coverage_vars_second[z2] = f"priority_coverage_SECOND::FAMILY::person={p}::family={fam}"
                    priority_required_vars[z2] = f"priority_required::tier=SECOND::person={p}::family={fam}"

    # Objective and dump
    duplicate_penalties = _find_duplicates(penalties)
    if duplicate_penalties:
        dup_descriptions = ", ".join(sorted({f"{w}:{v}" for w, v in duplicate_penalties}))
        raise ValueError(f"Duplicate penalties detected: {dup_descriptions}")

    pb.set_objective(penalties)
    pb.dump(args.out)

    # Map / debug
    selectors_by_var = {v: k for k, v in pb._selmap.items()}
    # Back-compat key "priority_coverage_vars" preserved to point to TOP
    priority_coverage_top_alias = dict(priority_coverage_vars_top)

    auto_soften_notes = {
        fam: {**meta, "family_label": fam_label(fam)}
        for fam, meta in softener.notes.items()
    }

    penalty_weights = {v: int(w) for w, v in penalties}
    if CONFIG.get("DEBUG_RELAX"):
        W_HARD = int(CONFIG.get("W_HARD", 0))
        for var in selectors_by_var:
            penalty_weights.setdefault(var, W_HARD)

    Path(args.map).write_text(json.dumps({
        "x_to_label": x_to_label,
        "q_vars":                    {},
        "selectors":                 pb._selmap,
        "selectors_by_var":          selectors_by_var,
        "penalty_weights":           penalty_weights,
        "manual_components":         {r.cid: bool(is_manual.get(r.cid, False)) for r in comps},
        "manual_components_original":{r.cid: bool(original_manual.get(r.cid, False)) for r in comps},
        "both_fallback_vars":        both_fallback_vars,
        "vprev_streak_vars":         vprev_streak_vars,
        "vprev_nonconsec_vars":      {},
        "preferred_miss_vars":       preferred_miss_vars,
        # Coverage maps
        "priority_coverage_vars":           priority_coverage_top_alias,   # kept for older consumers
        "priority_coverage_vars_top":       priority_coverage_vars_top,
        "priority_coverage_vars_second":    priority_coverage_vars_second,
        "priority_required_vars":           priority_required_vars,
        # NOTE: PRI repeat-over exports gated vars (only penalize when auto-PRI exists within family)
        "repeat_limit_pri_vars":     repeat_limit_pri_vars,
        "repeat_limit_non_vars":     repeat_limit_non_vars,
        "cooldown_pri_ladder_vars":  cooldown_pri_ladder_vars,
        "cooldown_non_ladder_vars":  cooldown_non_ladder_vars,
        # Gate visibility for cooldowns (lets you confirm AUTO was required)
        "component_drop_vars":       {cid: var for cid, (var, _) in component_drop_vars.items()},
        "component_drop_manual":     {cid: manual for cid, (_, manual) in component_drop_vars.items() if manual},
        "two_day_soft_vars": two_day_soft_vars,
        # back-compat alias:
        "sunday_two_day_vars": two_day_soft_vars,
        "auto_day_min_vars": auto_day_min_vars,
        "auto_day_min_sunday_vars": auto_day_min_sunday_vars,
        "cooldown_gate_info": cooldown_gate_info,
        "family_registry_path": str(getattr(args, "family_registry", "family_registry.json")),
        "family_labels": family_labels,
        "deprioritized_pair_vars": deprioritized_pair_vars,
        "effort_floor_vars": effort_floor_vars,
        "effort_floor_target": EFFORT_FLOOR_TARGET,
        "effort_floor_hard": EFFORT_FLOOR_HARD,
        "effort_floor_feasible": effort_floor_feasible,
        "effort_floor_hard_applied": effort_floor_hard_applied,
        "effort_floor_notes": effort_floor_notes,
        "effort_floor_eligible": sorted(effort_floor_eligible),
        "effort_floor_attainable": effort_floor_attainable,
        "auto_soften_families": auto_soften_notes,
        "fairness_targets": fairness_targets,
        "fairness_availability": fairness_target_notes,
        "config": CONFIG

    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats / README-ish summary
    # Count “gated pairs” (number of cooldown nodes that had an AUTO gate present)
    gated_pairs_count = len(cooldown_gate_info)
    twoday_soft_count = len(two_day_soft_vars)
    auto_day_var_count = len(auto_day_min_vars) + len(auto_day_min_sunday_vars)
    stats = [
        f"People: {len(people)}",
        f"Components: {len(comps)}",
        f"Debug-relax: {'ON' if DEBUG_RELAX else 'OFF'} (W_HARD={W_HARD})",
        f"Allow unassigned components: {'ON' if DEBUG_ALLOW_UNASSIGNED else 'OFF'} "
        f"(W_DEBUG_UNASSIGNED_PRIORITY={W_DEBUG_UNASSIGNED_PRIORITY}, "
        f"W_DEBUG_UNASSIGNED_NON_PRIORITY={W_DEBUG_UNASSIGNED_NON_PRIORITY}, "
        f"drop_vars={len(component_drop_vars)})",
    ]
    ladder_cfg = CONFIG.get("WEIGHT_LADDER", {})
    ladder_order = ladder_cfg.get("ORDER") or []
    if ladder_order:
        anchor_desc = ladder_cfg.get("TOP")
        if anchor_desc is None:
            anchor_desc = ladder_cfg.get("RATIO", 100) ** max(len(ladder_order) - 1, 0)
        stats.append(
            "Weight ladder: "
            f"ratio={ladder_cfg.get('RATIO', 100)}, anchor={anchor_desc}, "
            f"order={', '.join(ladder_order)}"
        )
    else:
        stats.append("Weight ladder: disabled (explicit WEIGHTS in use).")
    if EFFORT_FLOOR_TARGET > 0:
        feas_note = "ON" if effort_floor_feasible else "SKIPPED (insufficient supply)"
        stats.append(
            "Effort floor: "
            f"target={EFFORT_FLOOR_TARGET}, hard={'ON' if EFFORT_FLOOR_HARD else 'OFF'}, "
            f"feasible={feas_note}, eligible_people={len(effort_floor_eligible)}, "
            f"demand={effort_floor_notes.get('demand', 0)}, "
            f"supply_effort={effort_floor_notes.get('supply_effort', 0)}, "
            f"supply_capped={effort_floor_notes.get('supply_capped', 0)}, "
            f"attainable_min={effort_floor_notes.get('eligible_attainable_min', 0)}, "
            f"attainable_max={effort_floor_notes.get('eligible_attainable_max', 0)}"
        )
        reason = effort_floor_notes.get("reason")
        if reason:
            stats[-1] += f", reason={reason}"
    stats.extend([
        "Day rules:",
        f"  • AUTO present on (person,week,day) ⇒ soft min to 2 tasks (weekday W={W_AUTO_DAY}, Sunday W={W_AUTO_DAY_SUNDAY}); total selectors={auto_day_var_count}.",
        "  • Same-day task exclusions respected; banned same-day pairs enforced (named days only).",
        "  • Two named days in a week require at least one side to be fully manual.",
        "Sibling rules:",
        "  • Names-based pair groups (preferred-pair scoring).",
        "  • Family-based groups per FAMILY TOKEN: at-most-one per person; banned sibling pairs respected.",
        "Cooldown (prev-week, family-aggregated; applies only if ≥1 side AUTO):",
        f"  • PRIORITY: log-ladder soft W1 with base {COOLDOWN_GEO}; streak W1_STREAK={W1_STREAK}.",
        f"  • NON-PRIORITY: log-ladder soft W2 with base {COOLDOWN_GEO}; streak W2_STREAK={W2_STREAK}.",
        "  • GAP penalties removed.",
        f"  • Gated pairs (AUTO present on prev or curr week): {gated_pairs_count}",
        f"  • Two-day soft (both AUTO) counted pairs: {twoday_soft_count}",
        "Repeat limits (per person × family across horizon):",
        f"  • PRIORITY limit={LIMIT_PRI} (log-ladder soft at W1, base {REPEAT_OVER_GEO} above limit, gated on auto-PRI within family).",
        f"  • NON-PRIORITY limit={LIMIT_NON} (log-ladder soft at W2, base {REPEAT_OVER_GEO} above limit, gated on auto-NON within family).",
        f"Tier-3 manual-only 1-task day discourage: W3={W3}.",
        f"Tier-4 ‘Both’ fallback per-use: W4={W4}.",
        f"Tier-5 preferred-pair missed: W5={W5}.",
        f"Tier-6 across-horizon total-effort fairness (log-ladder): W6_OVER={W6_OVER}, W6_UNDER={W6_UNDER}.",
        f"Tier-6 fairness mean: base={mean_base}, scaled={mean_scaled}, target(with delta)={mean_target} "
        f"(FAIR_MEAN_MULTIPLIER={FAIR_MEAN_MULTIPLIER}, FAIR_OVER_START_DELTA={FAIR_OVER_START_DELTA})",
        f"Priority coverage ({PRIORITY_COVERAGE_MODE.upper()}): TOP weight={W_PRIORITY_MISS if W_PRIORITY_MISS > 0 else T1C}, "
        f"T2C={T2C} (SECOND; ignored if TOP already).",
        f"SiblingKey source: {'Extractor SiblingKey' if used_siblingkey else 'Synthesized from backend cooldown graph (fallback)'}",
        f"#vars (approx): {len(pb.vars)}  |  #constraints: {len(pb.constraints)}  |  obj terms: {len(pb.objective_terms)}",
    ])
    stats.append(
        f"Family registry: {getattr(args, 'family_registry', 'family_registry.json')} (tracked {len(family_labels)} families)"
    )
    if fairness_avail_enabled:
        stats.append(
            "Fairness availability scaling: "
            f"ref={fairness_reference}, power={fairness_power}, ratio_floor={fairness_min_ratio}, cap={fairness_max_ratio}, "
            f"mean_slots={fairness_counts_mean:.2f}"
        )
        if fairness_targets:
            lowest = sorted(fairness_targets.items(), key=lambda kv: kv[1])[:3]
            highest = sorted(fairness_targets.items(), key=lambda kv: kv[1])[-3:]
            if lowest:
                stats.append("  • Lowest targets: " + ", ".join(f"{name}:{target}" for name, target in lowest))
            if highest:
                stats.append("  • Highest targets: " + ", ".join(f"{name}:{target}" for name, target in highest))
    if softener.enabled:
        if softener.notes:
            stats.append("Auto-soften: skipped cooldown/repeat penalties for these scarce families:")
            for fam in sorted(softener.notes):
                meta = softener.notes[fam]
                label = fam_label(fam)
                disp = label if label == fam else f"{label} ({fam})"
                stats.append(
                    f"  • {disp}: slots={meta['slots']}, unique_people={meta['unique_people']}, "
                    f"slots/person={meta['slots_per_person']}, reasons={meta['reasons']}"
                )
        else:
            stats.append("Auto-soften: enabled but no families crossed the scarcity thresholds.")
    else:
        stats.append("Auto-soften: disabled via CONFIG.")

    if EFFORT_FLOOR_TARGET > 0:
        stats.append(
            f"Effort floor: target={EFFORT_FLOOR_TARGET}, weight={W_EFFORT_FLOOR}, "
            f"hard={'ON' if EFFORT_FLOOR_HARD else 'OFF'}, eligible_people={len(effort_floor_eligible)}"
        )
        if effort_floor_eligible:
            stats.append("  • Eligible for effort floor: " + ", ".join(sorted(effort_floor_eligible)))
    dprio_count = len(deprioritized_pair_vars)
    stats.append(f"Deprioritized pair soft hits (potential vars): {dprio_count}")

    Path(args.stats).write_text("\n".join(stats) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} | {args.map} | {args.stats}"
          f"{' | debug artifacts' if DEBUG_RELAX else ''}")


def encode_with_args(args, overrides: dict | None = None) -> None:
    config_data = build_config(overrides or {})
    global CONFIG
    prev_config = CONFIG
    CONFIG = config_data
    try:
        _encode(args)
    finally:
        CONFIG = prev_config


def run_encoder(
    *,
    components: Path,
    backend: Path,
    out: Path,
    map_path: Path,
    stats_path: Path,
    family_registry: Path | None = None,
    overrides: dict | None = None,
) -> None:
    args = argparse.Namespace(
        components=components,
        backend=backend,
        out=out,
        map=map_path,
        stats=stats_path,
        family_registry=family_registry or Path("family_registry.json"),
        config=None,
    )
    encode_with_args(args, overrides=overrides)


def main() -> None:
    args = parse_args()
    overrides: dict | None = None
    if args.config:
        cfg_path = resolve_data_path(Path(args.config))
        overrides = json.loads(cfg_path.read_text(encoding="utf-8"))
    encode_with_args(args, overrides=overrides)


if __name__ == "__main__":
    main()
