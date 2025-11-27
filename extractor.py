#!/usr/bin/env python3
"""
Manual-first + Repeat-pool unification + Pre-pass role swap (handles assigned & unassigned)
+ Component pool mirroring + Component splitting + Singleton promotion for components_all.csv.

NOW WITH:
- Repeat clamp to max 2 per (Week, Day, TaskName), with exception if ≥3 distinct manual-locked people.
- Priority Assignments (Backend F:G) respected as a post-role filter.
- For fully-assigned components, candidate lists collapse to exactly the assignees (1:1).
- Alert if the same person is assigned to multiple repeats of the same task on the same day.
- FIX: `RepeatMax` equals the number of surviving repeats for that task on that week/day (after clamping/skips).

ADDED:
- Header-assignee parsing (6th semicolon-separated field) and early application.
- De-duplication: same person cannot take multiple repeats of SAME TaskName on SAME Week/Day.
- NEW: `SiblingKey` column (union of cooldown families across task names).

ROLE PARITY (correct):
- A task is role-based only if that (Week, Day, TaskName) has ≥2 repeats, **and**
  the specific repeat is 1 (leader) or 2 (follower). Repeats ≥3 are neutral.
- Singletons (K=1) are neutral even if label says "1".

STRICT UNIFORMITY FOR COMPONENTS:
- When forming components, **all tasks must share the same role bucket**:
  leader-only, follower-only, or neutral-only. Any mix is disallowed.

NO GROUP SIZE CAP:
- Components will include **all** legal, role-uniform, complimentary tasks available
  (subject to exclusions and no duplicate TaskName inside the same component).

REPEAT NORMALIZATION:
- After clamping, renumber surviving repeats for each (Week, Day, TaskName) to 1..K.

PRIORITY TIERS (NEW):
- Extractor marks priority as T1 (Top) or T2 (Second) based on Backend columns:
  Top Priority = AE, Second Priority = AF (or matching header names).
- CSV outputs now have Priority ∈ {"T1","T2","NO"}.
"""

from __future__ import annotations
import csv, io, ssl, urllib.request, urllib.parse
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
import certifi
from collections import defaultdict

# ---------------------------- CONFIG ---------------------------------

DOC_ID = "1s1hdDGjMQTjT1P5zO3xMX__hM1V-5Y9rEGt8uUg5_B0"
TASK_ASSIGNMENT_GID = "1625318510"  # Task Assignment
BACKEND_GID = "1060258331"          # Backend

# Local cache
CACHE_TASKS_CSV = Path("task_assignment.csv")
CACHE_BACKEND_CSV = Path("backend.csv")

# Outputs
OUTPUT_DECISION_LOG = Path("decision_log.csv")
OUTPUT_UNASSIGNED_ONLY = Path("remaining_unassigned_only.csv")
OUTPUT_COMPONENTS_ALL = Path("components_all.csv")

FORCE_REFRESH = False

# Grid layout
HEADER_ROW_1BASED = 36
NAMES_START_1BASED = 37
FIRST_TASK_COL_1B = 2  # B
NAME_COL_1B = 1       # A

EXCEPTION_REPEAT_LIMIT = 3

# ---------------------------- I/O ------------------------------------

def export_csv_url(doc_id: str, gid: str) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{doc_id}/export"
    q = urllib.parse.urlencode({"format": "csv", "gid": gid})
    return f"{base}?{q}"

def download_if_needed(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TaskAssigner/1.6)"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return dest

def read_csv_matrix(path: Path) -> List[List[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    rdr = csv.reader(io.StringIO(text))
    return [list(row) for row in rdr]

def trim(s: str) -> str:
    return (s or "").strip()

# ------------------------ Task Grid Parse -----------------------------

def parse_task_grid(matrix: List[List[str]]):
    header_idx = HEADER_ROW_1BASED - 1
    people_start_idx = NAMES_START_1BASED - 1
    first_task_col_idx = FIRST_TASK_COL_1B - 1
    name_col_idx = NAME_COL_1B - 1

    header_row = matrix[header_idx] if header_idx < len(matrix) else []
    tasks: List[str] = []
    c = first_task_col_idx
    while c < len(header_row):
        h = trim(header_row[c])
        if not h: break
        tasks.append(h)
        c += 1

    people: List[str] = []
    r = people_start_idx
    while r < len(matrix):
        row = matrix[r] if r < len(matrix) else []
        name = trim(row[name_col_idx] if name_col_idx < len(row) else "")
        if not name: break
        people.append(name)
        r += 1

    avail: List[List[int]] = []
    for i, row_idx in enumerate(range(people_start_idx, people_start_idx + len(people))):
        row = matrix[row_idx] if row_idx < len(matrix) else []
        row_av: List[int] = []
        for j, col_idx in enumerate(range(first_task_col_idx, first_task_col_idx + len(tasks))):
            cell = trim(row[col_idx] if col_idx < len(row) else "")
            v = 1 if cell in ("1", "TRUE", "true", "Yes", "YES") else 0
            row_av.append(v)
        avail.append(row_av)

    return tasks, people, avail

# ------------------------ Backend Parse -------------------------------

class BackendConfig:
    def __init__(self):
        self.comp_adj: Dict[str, Set[str]] = defaultdict(set)
        self.priority_assign: Dict[str, List[str]] = defaultdict(list)  # ordered
        self.exclusions: List[Tuple[str, str]] = []
        self.exceptions: Set[str] = set()
        self.preferred_map: Dict[str, Set[str]] = defaultdict(set)
        self.banned_tasks: Set[str] = set()
        self.cooldown_pairs: List[Tuple[str, str]] = []
        self.cooldown_key: Dict[str, str] = {}
        self.all_eligible_tasks: Set[str] = set()
        self.strict_priority_tasks: Set[str] = set()  # kept for back-compat (now top ∪ second)
        self.roles_map: Dict[str, Dict[str, bool]] = {}
        self.effort_map: Dict[str, float] = {}
        # NEW: explicit priority tiers
        self.top_priority_tasks: Set[str] = set()     # T1
        self.second_priority_tasks: Set[str] = set()  # T2

def parse_backend(matrix: List[List[str]]) -> BackendConfig:
    cfg = BackendConfig()
    last_row = len(matrix)

    # Try to detect header row for name-based column lookup (fallback to fixed positions)
    header = matrix[0] if matrix else []

    def col_by_name(target: str) -> Optional[int]:
        target_l = target.strip().lower()
        for idx0, name in enumerate(header):
            if trim(name).lower() == target_l:
                return idx0 + 1  # convert to 1-based for our helper
        return None

    def col(row: List[str], idx1b: int) -> str:
        i0 = idx1b - 1
        return trim(row[i0]) if i0 < len(row) else ""

    # Static/fallback indices (1-based)
    NAME_COL = 1     # A
    ROLE_LEADER = 2  # B
    ROLE_FOLLOWER = 3# C
    ROLE_BOTH = 4    # D
    EFF_TASK = 9     # I
    EFF_VAL  = 10    # J
    COMPL_A  = 14    # N
    COMPL_B  = 15    # O
    PRIO_TASK = 6    # F
    PRIO_PERSON = 7  # G
    EXCL_A = 18      # R
    EXCL_B = 19      # S
    EXCEPT = 21      # U
    EVERYONE = 12    # L
    BANNED = 26      # Z
    COOLDOWN_A = 28  # AB
    COOLDOWN_B = 29  # AC
    STRICT_PRIORITY = 31  # AE
    TOP_PRIORITY_COL = col_by_name("Top Priority")
    SECOND_PRIORITY_COL = col_by_name("Second Priority")
    if TOP_PRIORITY_COL is None:
        TOP_PRIORITY_COL = 31  # AE
    if SECOND_PRIORITY_COL is None:
        SECOND_PRIORITY_COL = 32  # AF

    # Roles A:D
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; name = col(row, NAME_COL)
        if not name: continue
        leader = col(row, ROLE_LEADER) in ("1","TRUE","true","Yes","YES")
        follower = col(row, ROLE_FOLLOWER) in ("1","TRUE","true","Yes","YES")
        both    = col(row, ROLE_BOTH) in ("1","TRUE","true","Yes","YES")
        cfg.roles_map[name] = {"leader": leader, "follower": follower, "both": both}

    # Effort I:J
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; tname = col(row, EFF_TASK); eff = col(row, EFF_VAL)
        if not tname: continue
        try: v = float((eff or "1").replace(",", "."))
        except: v = 1.0
        cfg.effort_map[tname] = v if v > 0 else 1.0

    # Complimentary N:O
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; a = col(row, COMPL_A); b = col(row, COMPL_B)
        if a and b: cfg.comp_adj[a].add(b); cfg.comp_adj[b].add(a)

    # Priority assignments (task → ordered people), F:G
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; t = col(row, PRIO_TASK); p = col(row, PRIO_PERSON)
        if t and p: cfg.priority_assign[t].append(p)

    # Exclusions R:S
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; a = col(row, EXCL_A); b = col(row, EXCL_B)
        if a and b: cfg.exclusions.append((a, b))

    # Exceptions U
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; v = col(row, EXCEPT)
        if v: cfg.exceptions.add(v)

    # Everyone-eligible L
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; t = col(row, EVERYONE)
        if t: cfg.all_eligible_tasks.add(t)

    # Banned Z
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; t = col(row, BANNED)
        if t: cfg.banned_tasks.add(t)

    # Cooldown AB:AC → connected components canonicalization
    adj: Dict[str, Set[str]] = defaultdict(set)
    for r in range(2, last_row + 1):
        row = matrix[r - 1]; a = col(row, COOLDOWN_A); b = col(row, COOLDOWN_B)
        if a and b: adj[a].add(b); adj[b].add(a)
    seen = set()
    for start in list(adj.keys()):
        if start in seen: continue
        stack=[start]; comp=[]
        seen.add(start)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        canon = sorted(comp)[0]
        for name in comp: cfg.cooldown_key[name] = canon

    # Priority tiers AE (Top) & AF (Second) — or by header names
    for r in range(2, last_row + 1):
        row = matrix[r - 1]
        t_top = col(row, TOP_PRIORITY_COL)
        t_sec = col(row, SECOND_PRIORITY_COL)
        if t_top: cfg.top_priority_tasks.add(t_top)
        if t_sec: cfg.second_priority_tasks.add(t_sec)

    # Back-compat "strict" = union of both tiers
    cfg.strict_priority_tasks = set(cfg.top_priority_tasks) | set(cfg.second_priority_tasks)
    return cfg

# --------------------- Family canonicalization helpers ----------------

def canonical_family_for(name: str, cooldown_key: Dict[str, str]) -> str:
    n = trim(name);
    if not n: return ""
    canon = trim(cooldown_key.get(n, n))
    return canon or n

def families_for_names(names: List[str], cooldown_key: Dict[str, str]) -> List[str]:
    fams = {canonical_family_for(n, cooldown_key) for n in names if trim(n)}
    fams.discard(""); return sorted(fams)

# ------------------------ Model types --------------------------------

class Task:
    def __init__(self, idx: int, header: str, col: int, effort: float):
        self.idx = idx
        self.header = header
        self.col = col
        (self.week, self.day, self.time, self.name,
         self.repeat, self.group_key, self.header_assignee) = parse_task_header(header)
        self.available: List[int] = []
        self.assigned_to: Optional[int] = None
        self.skipped: bool = False
        self.banned: bool = False
        self.effort: float = effort
        self._orig_avail: List[int] = []
        self.manual_lock: bool = False
        self.strict_priority: bool = False
        self.everyone_eligible: bool = False
        self.component_id: Optional[str] = None

def parse_task_header(header: str):
    parts = [trim(x) for x in str(header).split(";")]
    week = parts[0] if len(parts) > 0 else ""
    day  = parts[1] if len(parts) > 1 else ""
    time = parts[2] if len(parts) > 2 else ""
    name = parts[3] if len(parts) > 3 else ""
    rep_raw = parts[4] if len(parts) > 4 else ""
    rep_num = 1
    if rep_raw:
        digits = "".join(ch for ch in rep_raw if ch.isdigit())
        rep_num = int(digits) if digits else 1
    assignee = parts[5] if len(parts) > 5 else ""  # header-assignee
    group_key = f"{week} | {day} | {name}"
    return week, day, time, name, rep_num, group_key, assignee

# ------------------------ Role helpers (fixed) ------------------------

def required_role_if_role_based(
    t: Task,
    repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]]
) -> Optional[str]:
    """Return leader/follower only for repeats 1 or 2 when the task is role-based."""
    k = (t.week, t.day, t.name)
    k_reps = repeats_per_wdn.get(k, set())
    if len(k_reps) < 2:         # singleton: neutral
        return None
    if t.repeat == 1:           # leader slot
        return "leader"
    if t.repeat == 2:           # follower slot
        return "follower"
    return None                 # ≥3: neutral

def role_ok_strict(roles_map: Dict[str, Dict[str, bool]], person: str, need: Optional[str]) -> bool:
    if not need:
        return True
    r = roles_map.get(person, {"leader": False, "follower": False, "both": False})
    # NEW: if person has Both, they satisfy any role-based need
    if r.get("both", False):
        return True
    return bool(r.get(need, False))

# ------------------------ Components & Parity -------------------------

def _is_complimentary_pair(a: str, b: str, comp_adj: Dict[str, Set[str]]) -> bool:
    return (b in comp_adj.get(a, set())) or (a in comp_adj.get(b, set()))

def _conflicts(a: str, b: str, exclusion_pairs: Set[FrozenSet[str]]) -> bool:
    return frozenset((a, b)) in exclusion_pairs

def _role_bucket_for_task_index(
    idx: int,
    tasks: List[Task],
    repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]]
) -> int:
    """
    1 -> leader bucket, 2 -> follower bucket, 0 -> neutral.
    Neutral when the TaskName is singleton or the specific repeat is ≥3.
    """
    t = tasks[idx]
    reps = repeats_per_wdn.get((t.week, t.day, t.name), set())
    if len(reps) < 2:
        return 0
    if t.repeat == 1:
        return 1
    if t.repeat == 2:
        return 2
    return 0  # ≥3 neutral

def _group_uniform_bucket(current_idxs: List[int],
                          tasks: List[Task],
                          repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]]) -> Optional[int]:
    """Return the group's bucket if all members share the same bucket, else None."""
    if not current_idxs:
        return None
    b0 = _role_bucket_for_task_index(current_idxs[0], tasks, repeats_per_wdn)
    for i in current_idxs[1:]:
        if _role_bucket_for_task_index(i, tasks, repeats_per_wdn) != b0:
            return None
    return b0

def _component_accepts_uniform(
    current_idxs: List[int],
    cand_idx: int,
    tasks: List[Task],
    repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]]
) -> bool:
    """STRICT: enforce that all nodes in a component share the same role bucket."""
    cand_bucket = _role_bucket_for_task_index(cand_idx, tasks, repeats_per_wdn)
    group_bucket = _group_uniform_bucket(current_idxs, tasks, repeats_per_wdn)
    if group_bucket is None:
        return False
    return cand_bucket == group_bucket

def build_components(
    tasks: List[Task],
    comp_adj: Dict[str, Set[str]],
    exclusion_pairs: Set[FrozenSet[str]],
    repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]],
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """
    Build components per (Week, Day) by greedy expansion:
      - Seed with a task; determine its role bucket (leader, follower, neutral).
      - Add ANY neighbor that:
          * is complimentary with at least one current member,
          * does not create an exclusion conflict with any current member,
          * does not duplicate a TaskName already inside the component,
          * matches the component's role bucket (strict uniformity).
      - Keep adding until saturation (no size cap).
      - Only emit components with size ≥ 2.
      - Tasks can appear in at most one component.
    """
    by_wd: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for t in tasks:
        if t.skipped or t.banned:
            continue
        by_wd[(t.week, t.day)].append(t.idx)

    comp_map: Dict[str, List[int]] = {}
    idx_to_comp: Dict[int, str] = {}
    cid_counter = 1

    def safe_can_add(current_idxs: List[int], cand_idx: int) -> bool:
        cand = tasks[cand_idx]
        current_names = [tasks[i].name for i in current_idxs]
        if cand.name in current_names:
            return False
        # must be complimentary with at least ONE member
        if not any(_is_complimentary_pair(tasks[i].name, cand.name, comp_adj) for i in current_idxs):
            return False
        # must not conflict with ANY member
        if any(_conflicts(tasks[i].name, cand.name, exclusion_pairs) for i in current_idxs):
            return False
        # must match role bucket
        if not _component_accepts_uniform(current_idxs, cand_idx, tasks, repeats_per_wdn):
            return False
        return True

    for (wk, dy), idxs in by_wd.items():
        if not idxs:
            continue

        leftovers = list(sorted(idxs, key=lambda k: (tasks[k].repeat, k)))

        # Precompute adjacency (complimentary & not excluded)
        adj: Dict[int, Set[int]] = defaultdict(set)
        for i in leftovers:
            ni = tasks[i].name
            for j in leftovers:
                if j <= i: continue
                nj = tasks[j].name
                if _conflicts(ni, nj, exclusion_pairs):
                    continue
                if _is_complimentary_pair(ni, nj, comp_adj):
                    adj[i].add(j); adj[j].add(i)

        used: Set[int] = set()
        for s in leftovers:
            if s in used:
                continue

            # Build a maximal uniform component starting from seed s
            seed_bucket = _role_bucket_for_task_index(s, tasks, repeats_per_wdn)
            group = [s]
            used.add(s)

            # frontier-based greedy expansion until no more candidates
            changed = True
            while changed:
                changed = False
                # candidates are neighbors of the current group not yet used and not already inside
                neighbor_pool = set()
                for u in group:
                    neighbor_pool |= adj.get(u, set())
                neighbor_pool -= set(group)
                neighbor_pool -= used

                # filter to those that pass all safety & uniformity checks
                candidates = [v for v in neighbor_pool
                              if _role_bucket_for_task_index(v, tasks, repeats_per_wdn) == seed_bucket
                              and safe_can_add(group, v)]

                if not candidates:
                    break

                # ranking: prefer nodes with higher connections to current group, then degree
                def score(v: int):
                    conn_to_group = sum(1 for u in group if v in adj.get(u, set()))
                    deg_total = len(adj.get(v, set()))
                    # slight bias for earlier columns for determinism
                    return (conn_to_group, deg_total, -tasks[v].col)

                candidates.sort(key=score, reverse=True)

                # try to add as many as possible this round (no cap)
                added_any = False
                for v in candidates:
                    if v in used or v in group:
                        continue
                    if safe_can_add(group, v):
                        group.append(v)
                        used.add(v)
                        added_any = True
                changed = added_any

            # Only emit uniform components with size ≥ 2
            if len(group) >= 2 and _group_uniform_bucket(group, tasks, repeats_per_wdn) is not None:
                cid = f"C{cid_counter}"; cid_counter += 1
                comp_map[cid] = group
                for i in group:
                    idx_to_comp[i] = cid
            # singles remain for singleton promotion

    for i, cid in idx_to_comp.items():
        tasks[i].component_id = cid

    return comp_map, idx_to_comp

# ------------------------ Decision log --------------------------------

DECISION_FIELDS = ["Step","Phase","TaskHeader","GroupKey","Repeat","AssignedTo","Status","Note"]

class DecisionLogger:
    def __init__(self):
        self.rows = (); self.step = 0
        self.rows = []
    def log(self, phase: str, task: Optional[Task], assigned_to: str, status: str, note: str = ""):
        self.step += 1
        self.rows.append({
            "Step": self.step, "Phase": phase,
            "TaskHeader": (task.header if task else ""),
            "GroupKey": (task.group_key if task else ""),
            "Repeat": (task.repeat if task else ""),
            "AssignedTo": assigned_to, "Status": status, "Note": note
        })
    def write_csv(self, out: Path):
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
            w.writeheader()
            for r in self.rows: w.writerow({k: r.get(k, "") for k in DECISION_FIELDS})


def exclusion_violation(
    day_assign: Dict[str, Dict[int, Set[str]]],
    excl_map: Dict[str, Set[str]],
    task: "Task",
    m_idx: int,
) -> bool:
    """Return True if assigning task to m_idx would violate exclusions on that day.

    One-candidate pools override exclusions so they can be promoted to manual assignments.
    """
    if len(getattr(task, "available", [])) == 1:
        return False
    dkey = f"{task.week}|{task.day}"
    already = day_assign[dkey].get(m_idx, set())
    for n in already:
        if n in excl_map.get(task.name, set()):
            return True
    return False

def apply_header_assignments(
    tasks: List["Task"],
    members: List[str],
    violates_exclusions,
    record_day_assign,
    logger: "DecisionLogger",
):
    for t in tasks:
        assignee_name = trim(getattr(t, "header_assignee", ""))
        if not assignee_name:
            continue
        try:
            m_idx = members.index(assignee_name)
        except ValueError:
            logger.log("manual", t, assignee_name, "Header assignee not found", ""); continue
        if (m_idx not in getattr(t, "available", [])) and (not getattr(t, "everyone_eligible", False)):
            logger.log("manual", t, assignee_name, "Header assignee not in pool", ""); continue
        if violates_exclusions(t, m_idx):
            logger.log("manual", t, assignee_name, "Header assignment blocked by exclusion", ""); continue
        t.assigned_to = m_idx; record_day_assign(t, m_idx)
        logger.log("manual", t, assignee_name, "Assigned (header)", "Applied before clamp")

# ------------------------ Helpers -------------------------------------

def _existing_component_ids(comp_map: Dict[str, List[int]], tasks: List[Task]) -> Set[str]:
    ids: Set[str] = set(comp_map.keys())
    ids |= {t.component_id for t in tasks if getattr(t, "component_id", None)}
    return ids

def next_component_id_global(comp_map: Dict[str, List[int]], tasks: List[Task]) -> str:
    ids = _existing_component_ids(comp_map, tasks)
    max_n = 0
    for cid in ids:
        if isinstance(cid, str) and cid.startswith("C"):
            num = cid[1:]
            if num.isdigit(): max_n = max(max_n, int(num))
    n = max_n + 1
    while f"C{n}" in ids: n += 1
    return f"C{n}"

def names_from_idxs(members: List[str], idxs: List[int]) -> List[str]:
    return [members[i] for i in idxs if 0 <= i < len(members)]

def filter_candidates_by_role_strict(cands: List[str], need: Optional[str], roles_map: Dict[str, Dict[str, bool]]) -> List[str]:
    if not need: return list(cands)
    return [p for p in cands if role_ok_strict(roles_map, p, need)]

def component_tasks_by_name_same_wd_repeat(tasks: List[Task], week: str, day: str, repeat: int) -> Dict[str, Task]:
    return {t.name: t for t in tasks if t.week == week and t.day == day and t.repeat == repeat and not t.skipped}

# ------------------------ Propagate/split ------------------------------

def propagate_or_split_components(
    tasks: List[Task],
    comp_map: Dict[str, List[int]],
    members: List[str],
    cfg: BackendConfig,
    logger: DecisionLogger,
    record_day_assign,
    violates_exclusions,
    repeats_per_wdn: Dict[Tuple[str,str,str], Set[int]],
):
    for comp_id in list(comp_map.keys()):
        idxs = comp_map.get(comp_id, [])
        if not idxs: continue

        assigned_by_person: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            t = tasks[i]
            if t.assigned_to is not None and not t.skipped and not t.banned:
                assigned_by_person[t.assigned_to].append(i)

        # Ensure component is uniform; if not (shouldn't happen), skip propagation for safety
        if _group_uniform_bucket(idxs, tasks, repeats_per_wdn) is None:
            continue

        active_nodes = [i for i in idxs if not tasks[i].banned and not tasks[i].skipped]

        # Solo candidate → auto-manual
        if not assigned_by_person:
            active_unassigned = [i for i in active_nodes if tasks[i].assigned_to is None]
            if active_unassigned:
                inter_names: Optional[Set[str]] = None
                for i in active_unassigned:
                    u = tasks[i]
                    pool_names = set(members) if u.everyone_eligible else {members[idx] for idx in u.available}
                    inter_names = pool_names if inter_names is None else (inter_names & pool_names)
                    if not inter_names: break
                inter_names = inter_names or set()

                rep = tasks[active_unassigned[0]]
                need_rep = required_role_if_role_based(rep, repeats_per_wdn)

                def role_ok_name(p: str, need: Optional[str]) -> bool:
                    if not need: return True
                    r = cfg.roles_map.get(p, {"leader": False, "follower": False, "both": False})
                    return bool(r.get(need, False))

                role_filtered = [p for p in sorted(inter_names) if role_ok_name(p, need_rep)]

                seen_pri: Set[str] = set(); prio_ordered: List[str] = []
                for i in active_unassigned:
                    for p in cfg.priority_assign.get(tasks[i].name, []):
                        if p not in seen_pri:
                            seen_pri.add(p)
                            prio_ordered.append(p)
                prio_filtered = [p for p in prio_ordered if p in set(role_filtered)] if prio_ordered else role_filtered

                if len(prio_filtered) == 1:
                    solo = prio_filtered[0]
                    solo_idx = members.index(solo) if solo in members else None
                    if solo_idx is not None:
                        legal_to_take: List[int] = []
                        for i in active_unassigned:
                            u = tasks[i]
                            if u.assigned_to is not None: continue
                            pool = (list(range(len(members))) if u.everyone_eligible else u.available)
                            if solo_idx not in pool: continue
                            need_u = required_role_if_role_based(u, repeats_per_wdn)
                            if not role_ok_strict(cfg.roles_map, solo, need_u): continue
                            if violates_exclusions(u, solo_idx): continue
                            legal_to_take.append(i)
                        if legal_to_take:
                            remaining = set(idxs)
                            for i in legal_to_take:
                                u = tasks[i]; u.assigned_to = solo_idx
                                record_day_assign(u, solo_idx)
                                logger.log("propagate", u, solo, "Assigned (solo-candidate)", f"Component {comp_id}")
                                remaining.discard(i)
                            new_cid = next_component_id_global(comp_map, tasks)
                            for i in legal_to_take: tasks[i].component_id = new_cid
                            comp_map[new_cid] = sorted(legal_to_take)
                            rem_list = sorted(list(remaining))
                            if rem_list: comp_map[comp_id] = rem_list
                            else: comp_map.pop(comp_id, None)
                            rep_task = tasks[legal_to_take[0]]
                            logger.log("propagate", rep_task, solo, "Component split (solo-candidate)", f"{comp_id} -> {new_cid} (+ remainder)")
                            continue

        if not assigned_by_person: continue

        remaining = set(idxs); new_components_to_add: List[Tuple[str, List[int]]] = []
        for pid, already_assigned_idxs in assigned_by_person.items():
            person = members[pid]
            legal_to_take: List[int] = []
            for i in idxs:
                if i in already_assigned_idxs: continue
                u = tasks[i]
                if u.assigned_to is not None or u.banned or u.skipped: continue
                if not _component_accepts_uniform(idxs, i, tasks, repeats_per_wdn): continue
                pool = (list(range(len(members))) if u.everyone_eligible else u.available)
                if pid not in pool: continue
                need_u = required_role_if_role_based(u, repeats_per_wdn)
                if not role_ok_strict(cfg.roles_map, person, need_u): continue
                if violates_exclusions(u, pid): continue
                legal_to_take.append(i)

            for i in legal_to_take:
                u = tasks[i]; u.assigned_to = pid
                record_day_assign(u, pid)
                logger.log("propagate", u, person, "Assigned (propagate)", f"Component {comp_id}")

            person_subset = sorted(set(already_assigned_idxs) | set(legal_to_take))
            if not person_subset: continue
            for i in person_subset: remaining.discard(i)
            new_cid = next_component_id_global(comp_map, tasks)
            new_components_to_add.append((new_cid, person_subset))
            for i in person_subset: tasks[i].component_id = new_cid
            rep = tasks[person_subset[0]]
            logger.log("propagate", rep, person, "Component split", f"{comp_id} -> {new_cid} (+ remainder)")

        rem_list = sorted(list(remaining))
        if rem_list: comp_map[comp_id] = rem_list
        else: comp_map.pop(comp_id, None)
        for new_cid, arr in new_components_to_add:
            comp_map[new_cid] = arr

    return comp_map

# ------------------------ Priority helper -----------------------------

def apply_priority_filter(task_names: List[str], role_filtered: List[str], cfg: BackendConfig) -> List[str]:
    ordered_priority: List[str] = []; seen: Set[str] = set()
    for t in task_names:
        for p in cfg.priority_assign.get(t, []):
            if p not in seen:
                seen.add(p)
                ordered_priority.append(p)
    if not ordered_priority: return role_filtered
    rf_set = set(role_filtered)
    keep = [p for p in ordered_priority if p in rf_set]
    return keep if keep else role_filtered

# ------------------------ Repeat clamp & normalize --------------------

def clamp_repeats_with_exceptions(
    tasks: List["Task"],
    roles_map: Dict[str, Dict[str, bool]],
    members: List[str],
    exceptions: Set[str],
    exception_limit: int,
    logger: "DecisionLogger",
):
    by_wdn: Dict[Tuple[str, str, str], List[Task]] = defaultdict(list)
    for t in tasks:
        if t.skipped or t.banned: continue
        by_wdn[(t.week, t.day, t.name)].append(t)

    def person_name_for(task: "Task") -> str | None:
        if task.assigned_to is not None: return members[task.assigned_to]
        if len(task.available) == 1: return members[task.available[0]]
        return None

    def person_role_ok(p: str, need: Optional[str]) -> bool:
        if not need: return True
        r = roles_map.get(p, {"leader": False, "follower": False, "both": False})
        return bool(r.get(need, False))

    for (wk, dy, nm), L in by_wdn.items():
        L = [u for u in L if not u.skipped and not u.banned]
        if not L: continue
        limit = exception_limit if nm in exceptions else 2
        if len(L) <= limit: continue

        manual = [u for u in L if (u.assigned_to is not None) or (len(u.available) == 1)]
        manual_people = set(filter(None, (person_name_for(u) for u in manual)))
        if len(manual_people) >= (limit + 1):
            logger.log("normalize", L[0], "", f"Kept > limit repeats",
                       f"{wk} {dy} {nm}: >= {limit+1} distinct manual assignees")
            continue

        L.sort(key=lambda u: u.repeat)
        pool_ge = [u for u in L if u.repeat > limit]
        pool_ge_manual = [u for u in pool_ge if (u.assigned_to is not None) or (len(u.available) == 1)]
        slot_map: Dict[int, Task] = {u.repeat: u for u in L if u.repeat <= limit}

        def need_for_slot(slot_rep: int) -> Optional[str]:
            if limit <= 1: return None
            return "leader" if (slot_rep % 2) == 1 else "follower"

        for slot in range(1, limit + 1):
            if slot in slot_map: continue
            need = need_for_slot(slot)
            picked = None
            for u in list(pool_ge_manual):
                p = person_name_for(u)
                if p and person_role_ok(p, need):
                    picked = u; pool_ge_manual.remove(u); pool_ge.remove(u); break
            if picked is None and pool_ge:
                picked = pool_ge.pop(0)
            if picked is not None:
                old = picked.repeat; picked.repeat = slot
                slot_map[slot] = picked
                logger.log("normalize", picked, person_name_for(picked) or "",
                           f"Promote rep{old}->rep{slot}", f"{wk} {dy} {nm} (limit={limit})")

        for u in L:
            if u.repeat > limit:
                u.skipped = True
                logger.log("normalize", u, "", f"Skip extra repeat (> {limit})", f"{wk} {dy} {nm} rep={u.repeat}")

def normalize_repeat_numbers(tasks: List["Task"], logger: "DecisionLogger"):
    by_wdn: Dict[Tuple[str, str, str], List[Task]] = defaultdict(list)
    for t in tasks:
        if t.skipped or t.banned: continue
        by_wdn[(t.week, t.day, t.name)].append(t)
    for (wk, dy, nm), L in by_wdn.items():
        alive = [u for u in L if not u.skipped and not u.banned]
        if not alive: continue
        alive.sort(key=lambda u: (u.repeat, u.col))
        for new_rep, u in enumerate(alive, start=1):
            if u.repeat != new_rep:
                logger.log("normalize", u, "", f"Renumber {u.repeat}->{new_rep}", f"{wk} {dy} {nm}")
                u.repeat = new_rep

# ------------------------ Core ----------------------------------------

def main():
    # Download
    task_url = export_csv_url(DOC_ID, TASK_ASSIGNMENT_GID)
    backend_url = export_csv_url(DOC_ID, BACKEND_GID)
    task_csv = download_if_needed(task_url, CACHE_TASKS_CSV, force=FORCE_REFRESH)
    backend_csv = download_if_needed(backend_url, CACHE_BACKEND_CSV, force=FORCE_REFRESH)

    task_mat = read_csv_matrix(task_csv)
    backend_mat = read_csv_matrix(backend_csv)

    headers, members, avail_matrix = parse_task_grid(task_mat)
    cfg = parse_backend(backend_mat)

    # Build tasks
    tasks: List[Task] = []
    for j, header in enumerate(headers):
        tname = parse_task_header(header)[3]
        eff = cfg.effort_map.get(tname, 1.0)
        t = Task(j, header, j, eff)
        tasks.append(t)

    # Availability + flags
    for j, t in enumerate(tasks):
        base_idxs = [i for i in range(len(members)) if avail_matrix[i][j] == 1]
        if t.name in cfg.all_eligible_tasks:
            base_idxs = list(range(len(members)))
            t.everyone_eligible = True
        t.available = base_idxs
        t._orig_avail = list(base_idxs)
        t.banned = (t.name in cfg.banned_tasks)
        # strict_priority kept for back-compat; now true if in T1 or T2
        t.strict_priority = (t.name in cfg.strict_priority_tasks)

    logger = DecisionLogger()

    exclusion_pairs: Set[FrozenSet[str]] = {frozenset((a, b)) for (a, b) in cfg.exclusions}

    # Exclusions/day-assign helpers
    excl_map: Dict[str, Set[str]] = defaultdict(set)
    for a, b in cfg.exclusions:
        excl_map[a].add(b); excl_map[b].add(a)

    def same_day_key(task: Task) -> str: return f"{task.week}|{task.day}"
    day_assign: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def violates_exclusions(task: Task, m_idx: int) -> bool:
        return exclusion_violation(day_assign, excl_map, task, m_idx)

    def record_day_assign(task: Task, m_idx: int):
        dkey = same_day_key(task); day_assign[dkey][m_idx].add(task.name)

    def unrecord_day_assign(task: Task, m_idx: int):
        dkey = same_day_key(task); s = day_assign[dkey].get(m_idx)
        if s and task.name in s: s.remove(task.name)

    # HEADER ASSIGNMENTS
    apply_header_assignments(tasks, members, violates_exclusions, record_day_assign, logger)

    # CLAMP + NORMALIZE
    clamp_repeats_with_exceptions(tasks, cfg.roles_map, members, cfg.exceptions, EXCEPTION_REPEAT_LIMIT, logger)
    normalize_repeat_numbers(tasks, logger)

    # Repeat set AFTER normalization
    repeats_per_wdn: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    for t in tasks:
        if t.skipped or t.banned: continue
        repeats_per_wdn[(t.week, t.day, t.name)].add(t.repeat)

    # MANUAL single-candidate
    for t in sorted([x for x in tasks if x.assigned_to is None and not x.skipped and not x.banned], key=lambda x: x.col):
        t.manual_lock = (len(t.available) == 1)
        if not t.manual_lock: continue
        m_idx = t.available[0]; person = members[m_idx]
        need = required_role_if_role_based(t, repeats_per_wdn)
        if role_ok_strict(cfg.roles_map, person, need) and not violates_exclusions(t, m_idx):
            t.assigned_to = m_idx; record_day_assign(t, m_idx)
            logger.log("manual", t, person, "Assigned (manual)", "")

    # Unify sibling pools (1 vs 2)
    def unify_repeat_pools(note_tag: str):
        by_wd: Dict[Tuple[str,str], Dict[int, Dict[str, Task]]] = defaultdict(lambda: defaultdict(dict))
        for u in tasks:
            if u.skipped: continue
            by_wd[(u.week,u.day)][u.repeat][u.name] = u
        for (wk,dy), rep_map in by_wd.items():
            for name in set(rep_map.get(1,{}).keys()) | set(rep_map.get(2,{}).keys()):
                t1 = rep_map.get(1,{}).get(name); t2 = rep_map.get(2,{}).get(name)
                if not t1 or not t2: continue
                pool = list(range(len(members))) if (t1.everyone_eligible or t2.everyone_eligible) else sorted(set(t1._orig_avail) | set(t2._orig_avail))
                t1.available = list(pool); t2.available = list(pool)
                t1.manual_lock = (len(t1.available) == 1 and t1.assigned_to is None)
                t2.manual_lock = (len(t2.available) == 1 and t2.assigned_to is None)
                logger.log("unify", t1, "", "Pools unified", f"{note_tag}: {wk} {dy} {name} → |pool|={len(pool)}")
    unify_repeat_pools("repeat-pool-unify (post-manual)")

    # PRE-PASS role swap (only if role-based & repeat in {1,2})
    for t in tasks:
        if t.skipped or t.banned or t.assigned_to is not None: continue
        t.manual_lock = (len(t.available) == 1)
        if not t.manual_lock: continue
        m_idx = t.available[0]; person = members[m_idx]
        need = required_role_if_role_based(t, repeats_per_wdn)
        if need and not role_ok_strict(cfg.roles_map, person, need):
            sib = None
            if len(repeats_per_wdn.get((t.week, t.day, t.name), set())) > 1:
                sib_rep = 2 if t.repeat == 1 else 1
                for u in tasks:
                    if (u.week, u.day, u.name, u.repeat) == (t.week, t.day, t.name, sib_rep):
                        sib = u; break
            if sib and sib.assigned_to is None and not sib.banned and not sib.skipped:
                sib_need = required_role_if_role_based(sib, repeats_per_wdn)
                if role_ok_strict(cfg.roles_map, person, sib_need) and not violates_exclusions(sib, m_idx):
                    sib.assigned_to = m_idx; record_day_assign(sib, m_idx)
                    logger.log("pre-pass", sib, person, f"Assigned (manual, moved) → {person}", "Role swap to sibling")

    for t in tasks:
        if t.skipped or t.banned or t.assigned_to is None: continue
        person_idx = t.assigned_to; person = members[person_idx]
        need = required_role_if_role_based(t, repeats_per_wdn)
        if need and not role_ok_strict(cfg.roles_map, person, need):
            sib = None
            if len(repeats_per_wdn.get((t.week, t.day, t.name), set())) > 1:
                sib_rep = 2 if t.repeat == 1 else 1
                for u in tasks:
                    if (u.week, u.day, u.name, u.repeat) == (t.week, t.day, t.name, sib_rep):
                        sib = u; break
            if sib and not sib.banned and not sib.skipped and sib.assigned_to is None:
                sib_need = required_role_if_role_based(sib, repeats_per_wdn)
                if role_ok_strict(cfg.roles_map, person, sib_need) and not violates_exclusions(sib, person_idx):
                    sib.assigned_to = person_idx; record_day_assign(sib, person_idx)
                    logger.log("pre-pass", sib, person, f"Assigned (manual, moved) → {person}", "Role swap from assigned")
                    unrecord_day_assign(t, person_idx); t.assigned_to = None

    # Enforce uniqueness per (Week, Day, TaskName)
    def enforce_unique_per_task_per_day():
        by_wdn: Dict[Tuple[str, str, str], List[Task]] = defaultdict(list)
        for t in tasks:
            if t.skipped or t.banned or t.assigned_to is None: continue
            by_wdn[(t.week, t.day, t.name)].append(t)
        for (wk, dy, nm), L in by_wdn.items():
            per_person: Dict[int, List[Task]] = defaultdict(list)
            for u in L: per_person[u.assigned_to].append(u)
            for pid, assigned_tasks in per_person.items():
                if len(assigned_tasks) <= 1: continue
                person = members[pid]
                def score(u: Task) -> Tuple[int,int]:
                    need = required_role_if_role_based(u, repeats_per_wdn)
                    can  = role_ok_strict(cfg.roles_map, person, need)
                    return (1 if can else 0, 1 if u.repeat == 1 else 0)
                keep = max(assigned_tasks, key=score)
                for u in assigned_tasks:
                    if u is keep: continue
                    unrecord_day_assign(u, pid); u.assigned_to = None
                    logger.log("dedupe", u, person, "Unassigned duplicate repeat", f"{wk} {dy} {nm} (kept rep {keep.repeat})")
    enforce_unique_per_task_per_day()

    # Recompute manual locks for leftovers
    for t in tasks:
        if t.assigned_to is None:
            t.manual_lock = (len(t.available) == 1)

    # BUILD COMPONENTS (strict uniformity, no size cap) & propagate/split
    comp_map, idx_to_comp = build_components(tasks, cfg.comp_adj, exclusion_pairs, repeats_per_wdn)
    comp_map = propagate_or_split_components(
        tasks, comp_map, members, cfg, logger,
        record_day_assign,
        violates_exclusions=lambda t, midx: (t.name in excl_map and any(n in excl_map[t.name] for n in day_assign[f"{t.week}|{t.day}"][midx])),
        repeats_per_wdn=repeats_per_wdn,
    )

    # Group-manual placement inside components (still respects uniformity via acceptance)
    for t in sorted([x for x in tasks if x.assigned_to is None and x.manual_lock and x.component_id], key=lambda x: x.col):
        if t.banned or t.skipped: continue
        m_idx = t.available[0]; person = members[m_idx]
        comp_id = t.component_id; idxs = comp_map.get(comp_id, [])[:]
        if _group_uniform_bucket(idxs, tasks, repeats_per_wdn) is None:
            continue
        legal: List[int] = []
        for i in idxs:
            u = tasks[i]
            if u.assigned_to is not None or u.banned or u.skipped: continue
            if not _component_accepts_uniform(idxs, i, tasks, repeats_per_wdn): continue
            pool = (list(range(len(members))) if u.everyone_eligible else u.available)
            if m_idx not in pool: continue
            need_u = required_role_if_role_based(u, repeats_per_wdn)
            if not role_ok_strict(cfg.roles_map, person, need_u): continue
            legal.append(i)
        if legal:
            for i in legal:
                u = tasks[i]; u.assigned_to = m_idx
                record_day_assign(u, m_idx)
                logger.log("manual", u, person, "Assigned (manual, group)", f"Component {comp_id}")
            remain = [i for i in idxs if i not in legal]
            if remain:
                comp_map[comp_id] = sorted(remain)
                for i in legal: tasks[i].component_id = None

    # Alerts: duplicate repeats per person
    alerts: List[str] = []
    by_wdn_alert: Dict[Tuple[str, str, str], List[Task]] = defaultdict(list)
    for t in tasks:
        if t.skipped or t.banned: continue
        by_wdn_alert[(t.week, t.day, t.name)].append(t)
    for (wk, dy, nm), L in by_wdn_alert.items():
        counts: Dict[int, int] = defaultdict(int)
        for u in L:
            if u.assigned_to is not None:
                counts[u.assigned_to] += 1
        for pid, cnt in counts.items():
            if cnt >= 2:
                person = members[pid]
                alerts.append(f"ALERT: {person} assigned to {cnt} repeats of '{nm}' on {wk} {dy}")
                logger.log("alert", L[0], person, "Duplicate repeats", f"{wk} {dy} {nm} x{cnt}")

    # Candidate pools for reporting
    all_member_names = list(members)
    task_candidates: Dict[int, Set[str]] = {}
    for t in tasks:
        if t.everyone_eligible:
            task_candidates[t.idx] = set(all_member_names)
        else:
            task_candidates[t.idx] = set(names_from_idxs(members, t.available))

    def task_total_effort(t: Task) -> float:
        return float(t.effort or 0)
    def component_total_effort(idxs: List[int]) -> float:
        return float(sum(tasks[i].effort for i in idxs if not tasks[i].banned and not tasks[i].skipped))

    # ---- NEW: Priority tier flags (T1/T2/NO) ----
    def priority_tier_task(t: Task) -> str:
        if t.name in cfg.top_priority_tasks:
            return "T1"
        if t.name in cfg.second_priority_tasks:
            return "T2"
        return "NO"

    def priority_tier_component(idxs: List[int]) -> str:
        has_t1 = any(tasks[i].name in cfg.top_priority_tasks for i in idxs)
        has_t2 = any(tasks[i].name in cfg.second_priority_tasks for i in idxs)
        if has_t1: return "T1"
        if has_t2: return "T2"
        return "NO"

    def component_priority_pool_ordered(task_names: List[str]) -> List[str]:
        seen: Set[str] = set(); out: List[str] = []
        for nm in task_names:
            for p in cfg.priority_assign.get(nm, []):
                if p not in seen:
                    seen.add(p); out.append(p)
        return out

    def component_candidate_intersection(idxs: List[int]) -> Tuple[List[str], List[str], List[str]]:
        active = [i for i in idxs if tasks[i].assigned_to is None and not tasks[i].banned and not tasks[i].skipped]
        if not active: return [], [], []
        inter: Optional[Set[str]] = None
        for i in active:
            pool = task_candidates[i]
            inter = pool if inter is None else (inter & pool)
        cands = sorted(inter or set())
        rep = tasks[active[0]]
        need = required_role_if_role_based(rep, repeats_per_wdn)
        role_filtered = filter_candidates_by_role_strict(cands, need, cfg.roles_map)
        task_names = [tasks[i].name for i in active]
        prio_ordered = component_priority_pool_ordered(task_names)
        if prio_ordered:
            rf_set = set(role_filtered)
            prio_filtered = [p for p in prio_ordered if p in rf_set]
            if prio_filtered:
                return cands, role_filtered, prio_filtered
        return cands, role_filtered, role_filtered

    # remaining_unassigned_only.csv
    unassigned_rows = []
    for comp_id, idxs in sorted(((cid, arr) for cid, arr in comp_map.items() if arr), key=lambda kv: kv[0]):
        all_assigned = all(tasks[i].assigned_to is not None or tasks[i].banned or tasks[i].skipped for i in idxs)
        if all_assigned: continue
        cands, role_filtered, prio_filtered = component_candidate_intersection(idxs)
        rep = tasks[idxs[0]]
        task_names = [tasks[i].name for i in idxs if not tasks[i].banned and not tasks[i].skipped]
        rmax = max((len(repeats_per_wdn.get((rep.week, rep.day, nm), set())) for nm in task_names), default=0)
        assignees = sorted({members[tasks[i].assigned_to] for i in idxs if tasks[i].assigned_to is not None})
        tot_eff = component_total_effort(idxs)
        task_count = sum(1 for i in idxs if not tasks[i].banned and not tasks[i].skipped)
        sibling_key = " || ".join(families_for_names(task_names, cfg.cooldown_key))
        unassigned_rows.append({
            "Kind":"Component","ComponentId":comp_id,"Week":rep.week,"Day":rep.day,"Repeat":rep.repeat,"RepeatMax":rmax,
            "Task Count":task_count,"Names":" | ".join([tasks[i].name for i in idxs]),"SiblingKey":sibling_key,
            "Priority":priority_tier_component(idxs),"Assigned?":"YES" if all_assigned else "NO",
            "Assigned To":", ".join(assignees),
            "Candidate Count":len(prio_filtered),"Candidates":", ".join(cands),
            "Role-Filtered Candidates":", ".join(prio_filtered),"Total Effort":f"{tot_eff:.2f}",
        })

    comp_task_idxs = set(i for idxs in comp_map.values() for i in idxs)
    for t in tasks:
        if t.idx in comp_task_idxs or t.banned or t.skipped or t.assigned_to is not None: continue
        cand_list = sorted(task_candidates[t.idx])
        need = required_role_if_role_based(t, repeats_per_wdn)
        role_filtered = filter_candidates_by_role_strict(cand_list, need, cfg.roles_map)
        prio_ordered = cfg.priority_assign.get(t.name, [])
        if prio_ordered:
            rf_set = set(role_filtered)
            prio_filtered = [p for p in prio_ordered if p in rf_set]
            if prio_filtered: role_filtered = prio_filtered
        tot_eff = task_total_effort(t)
        rmax = len(repeats_per_wdn.get((t.week, t.day, t.name), set()))
        sibling_key = " || ".join(families_for_names([t.name], cfg.cooldown_key))
        unassigned_rows.append({
            "Kind":"Task","ComponentId":"","Week":t.week,"Day":t.day,"Repeat":t.repeat,"RepeatMax":rmax,"Task Count":1,
            "Names":t.name,"SiblingKey":sibling_key,"Priority":priority_tier_task(t),
            "Assigned?":"NO","Assigned To":"",
            "Candidate Count":len(role_filtered),"Candidates":", ".join(cand_list),
            "Role-Filtered Candidates":", ".join(role_filtered),"Total Effort":f"{tot_eff:.2f}",
        })

    def sort_key(row):
        # T1 highest, then T2, then NO; break ties by smaller candidate pool first
        pri_rank = {"T1": 0, "T2": 1, "YES": 0, "NO": 2}  # accept legacy "YES"/"NO" if present
        return (pri_rank.get(row.get("Priority","NO"), 2), row.get("Candidate Count", 9999))
    unassigned_rows.sort(key=sort_key)

    # components_all.csv (promote singletons)
    comp_map_all: Dict[str, List[int]] = {k: list(v) for k, v in comp_map.items() if v}
    used_task_idxs = set(i for idxs in comp_map_all.values() for i in idxs)
    for t in tasks:
        if t.banned or t.skipped: continue
        if t.idx in used_task_idxs: continue
        cid = next_component_id_global(comp_map_all, tasks)
        comp_map_all[cid] = [t.idx]; used_task_idxs.add(t.idx); t.component_id = cid

    components_rows = []
    for comp_id, idxs in sorted(((cid, arr) for cid, arr in comp_map_all.items() if arr), key=lambda kv: kv[0]):
        rep = tasks[idxs[0]]
        task_names = [tasks[i].name for i in idxs if not tasks[i].banned and not tasks[i].skipped]
        rmax = max((len(repeats_per_wdn.get((rep.week, rep.day, nm), set())) for nm in task_names), default=0)
        all_assigned = all(tasks[i].assigned_to is not None or tasks[i].banned or tasks[i].skipped for i in idxs)
        assignees = sorted({members[tasks[i].assigned_to] for i in idxs if tasks[i].assigned_to is not None})
        tot_eff = float(sum(tasks[i].effort for i in idxs if not tasks[i].banned and not tasks[i].skipped))
        task_count = sum(1 for i in idxs if not tasks[i].banned and not tasks[i].skipped)
        if all_assigned:
            cands = role_filtered = prio_filtered = list(assignees)
        else:
            active = [k for k in idxs if tasks[k].assigned_to is None and not tasks[k].banned and not tasks[k].skipped]
            inter: Optional[Set[str]] = None
            for i in active:
                pool = set(names_from_idxs(members, tasks[i].available)) if not tasks[i].everyone_eligible else set(members)
                inter = pool if inter is None else (inter & pool)
            inter = inter or set()
            rep_need = required_role_if_role_based(rep, repeats_per_wdn)
            rf = filter_candidates_by_role_strict(sorted(inter), rep_need, cfg.roles_map)
            prio = []
            seen: Set[str] = set()
            for nm in task_names:
                for p in cfg.priority_assign.get(nm, []):
                    if p not in seen: seen.add(p); prio.append(p)
            keep = [p for p in prio if p in set(rf)]
            cands = sorted(inter); role_filtered = (keep if keep else rf); prio_filtered = role_filtered
        sibling_key = " || ".join(families_for_names(task_names, cfg.cooldown_key))
        components_rows.append({
            "Kind":"Component","ComponentId":comp_id,"Week":rep.week,"Day":rep.day,"Repeat":rep.repeat,"RepeatMax":rmax,
            "Task Count":task_count,"Names":" | ".join([tasks[i].name for i in idxs]),"SiblingKey":sibling_key,
            "Priority":priority_tier_component(idxs),
            "Assigned?":"YES" if all_assigned else "NO","Assigned To":", ".join(assignees),
            "Candidate Count":len(prio_filtered),"Candidates":", ".join(cands),
            "Role-Filtered Candidates":", ".join(prio_filtered),"Total Effort":f"{tot_eff:.2f}",
        })

    components_rows.sort(key=sort_key)

    # Write files
    logger.write_csv(OUTPUT_DECISION_LOG); print(f"Wrote: {OUTPUT_DECISION_LOG.resolve()}")
    with OUTPUT_UNASSIGNED_ONLY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Kind","ComponentId","Week","Day","Repeat","RepeatMax","Task Count","Names","SiblingKey",
            "Priority","Assigned?","Assigned To","Candidate Count","Candidates","Role-Filtered Candidates","Total Effort"
        ])
        w.writeheader(); [w.writerow(r) for r in unassigned_rows]
    print(f"Wrote: {OUTPUT_UNASSIGNED_ONLY.resolve()}")

    with OUTPUT_COMPONENTS_ALL.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Kind","ComponentId","Week","Day","Repeat","RepeatMax","Task Count","Names","SiblingKey",
            "Priority","Assigned?","Assigned To","Candidate Count","Candidates","Role-Filtered Candidates","Total Effort"
        ])
        w.writeheader(); [w.writerow(r) for r in components_rows]
    print(f"Wrote: {OUTPUT_COMPONENTS_ALL.resolve()}")

if __name__ == "__main__":
    main()
