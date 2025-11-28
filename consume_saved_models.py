#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consume saved SAT assignments (models) and produce:
  - assigned_optimal.csv
  - models_summary.csv
  - loads_by_person.csv
  - fairness_plots_bars.png
  - fairness_plots_lorenz.png
  - penalties_activated.csv (all weight-impacting decisions that were True)
  - component_penalties.csv (per-component penalty rollups for the best model)
  - cooldown_debug_by_pf.csv (debug: chosen weeks per (person,family), AUTO/MANUAL)
  - stdout: readable list of weight-impacting decisions per model.

Accepted model line formats:
  1) SAT4J-like: v x1 -x2 x3 -x4 ...
  2) Coef/var pairs: v +1 x1 -1 x2 +1 x3 ...

Loads:
  --metric=count|taskcount|effort
  - taskcount: column "Task Count" (>=1)
  - effort   : column "Total Effort"
  - count    : 1 per assigned component

Note: Analyzer intentionally excludes Tier-6 tiny per-unit load penalties.
"""

from __future__ import annotations
import argparse, csv, io, json, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# ----------------------------- CLI ------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--models', required=True, type=Path,
                    help='Text file with one or more lines containing "v ..." assignments')
    ap.add_argument('--varmap', required=True, type=Path,
                    help='varmap.json containing x_to_label and optional penalty maps')
    ap.add_argument('--components', required=True, type=Path,
                    help='components_all.csv (the pre-assignment components table)')
    ap.add_argument('--metric', choices=['count','taskcount','effort'], default='count',
                    help='Load metric used for fairness')
    ap.add_argument('--assigned-out', default='assigned_optimal.csv')
    ap.add_argument('--models-out', default='models_summary.csv')
    ap.add_argument('--loads-out', default='loads_by_person.csv')
    ap.add_argument('--plots-bars', default='fairness_plots_bars.png')
    ap.add_argument('--plots-lorenz', default='fairness_plots_lorenz.png')
    ap.add_argument('--penalties-out', default='penalties_activated.csv',
                    help='CSV listing all activated (True) penalty decisions in the best model')
    ap.add_argument('--component-penalties-out', default='component_penalties.csv',
                    help='CSV listing per-component penalty totals and details for the best model')
    ap.add_argument('--cooldown-debug-out', default='cooldown_debug_by_pf.csv',
                    help='CSV listing chosen weeks per (person,family) with AUTO/MANUAL flags')
    return ap.parse_args()

# ----------------------------- Parsing utils ---------------------------
VLINE_RE = re.compile(r'v\b', re.IGNORECASE)
VAR_RE   = re.compile(r'^-?x(\d+)$', re.IGNORECASE)
COEF_RE  = re.compile(r'^[+-]?\d+$')

def read_text_guess(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ('utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'cp1250', 'latin-1'):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode('latin-1', errors='replace')

def _clean_token(tok: str) -> str:
    return tok.strip().strip(',;')

def parse_models_from_text(text: str) -> List[List[str]]:
    models: List[List[str]] = []
    for rawline in text.splitlines():
        line = rawline.strip()
        if not line or 'v' not in line:
            continue
        if line.startswith('[') and line.endswith(']'):
            line = line[1:-1].strip()
        m = VLINE_RE.search(line)
        if not m:
            continue
        toks = line[m.end():].strip().split()

        true_vars: Set[str] = set()
        i = 0
        pairs_mode = (len(toks) >= 2 and COEF_RE.match(_clean_token(toks[0])) and
                      _clean_token(toks[1]).lstrip('+-').lower().startswith('x'))

        if pairs_mode:
            while i < len(toks):
                coef_tok = _clean_token(toks[i]); i += 1
                if i >= len(toks): break
                var_tok = _clean_token(toks[i]); i += 1
                var_core = var_tok.lstrip('+-').lower()
                if not VAR_RE.match(var_core): continue
                try:
                    coef = int(coef_tok)
                except ValueError:
                    lit = var_tok
                    if lit.startswith('-'):
                        true_vars.discard(lit[1:].lower())
                    else:
                        true_vars.add(lit.lower())
                    continue
                varname = var_core if var_core.startswith('x') else f'x{var_core}'
                if coef > 0: true_vars.add(varname)
                else: true_vars.discard(varname)
        else:
            while i < len(toks):
                lit = _clean_token(toks[i]); i += 1
                lit_low = lit.lower()
                if not lit_low: continue
                if lit_low.startswith('-'):
                    core = lit_low[1:]
                    if VAR_RE.match(core): true_vars.discard(core)
                else:
                    if VAR_RE.match(lit_low): true_vars.add(lit_low)

        def sort_key(s: str):
            num = s[1:]
            return (0, int(num)) if num.isdigit() else (1, s)
        ordered = sorted(true_vars, key=sort_key)
        if ordered: models.append(ordered)
    return models

# -------------------------- Var map & components -----------------------
def read_varmap(path: Path):
    vm = json.loads(path.read_text(encoding='utf-8'))
    x_to_label: Dict[str,str] = vm.get('x_to_label', {})
    manual_original = {cid for cid, flag in (vm.get('manual_components_original') or {}).items() if flag}
    penalty_weights = vm.get('penalty_weights', {}) or {}
    penalty_components = vm.get('penalty_components', {}) or {}

    # Optional maps (present in newer encoders)
    vprev_pri_vars     = vm.get('vprev_pri_vars', {})      # may be missing in current encoder
    vprev_non_vars     = vm.get('vprev_non_vars', {})      # may be missing in current encoder
    both_fallback      = vm.get('both_fallback_vars', {})
    q_vars             = vm.get('q_vars', {})
    vprev_streak       = vm.get('vprev_streak_vars', {})
    vprev_nonconsec    = vm.get('vprev_nonconsec_vars', {})  # encoder now leaves this empty
    preferred_miss     = vm.get('preferred_miss_vars', {})
    prio_cov_top = vm.get('priority_coverage_vars_top', {}) or {}
    prio_cov_second = vm.get('priority_coverage_vars_second', {}) or {}
    priority_coverage = vm.get('priority_coverage_vars', {}) or {**prio_cov_top, **prio_cov_second}
    effort_floor_vars  = vm.get('effort_floor_vars', {}) or {}

    # Geometric ladders + over-limit maps (present in the optimized encoder)
    cooldown_pri_ladder = vm.get('cooldown_pri_ladder_vars', {})
    cooldown_non_ladder = vm.get('cooldown_non_ladder_vars', {})
    repeat_over_pri     = vm.get('repeat_limit_pri_vars', {})
    repeat_over_non     = vm.get('repeat_limit_non_vars', {})

    two_day_soft = vm.get('two_day_soft_vars') or vm.get('sunday_two_day_vars') or {}

    debug_relax_by_var = vm.get('selectors_by_var', {}) or {}

    component_drop_by_cid = vm.get('component_drop_vars', {}) or {}
    component_drop_by_var = {}
    for cid, var in component_drop_by_cid.items():
        if not var:
            continue
        if cid in manual_original:
            continue
        label = x_to_label.get(var) or f"drop::{cid}"
        component_drop_by_var[var] = label

    auto_day_min_vars = vm.get('auto_day_min_vars', {}) or {}
    auto_day_min_sunday_vars = vm.get('auto_day_min_sunday_vars', {}) or {}

    penalty_maps = {
        'CooldownPRI': vm.get('vprev_pri_vars', {}),
        'CooldownNON': vm.get('vprev_non_vars', {}),
        'CooldownStreak': vm.get('vprev_streak_vars', {}),
        'CooldownNonConsec': vm.get('vprev_nonconsec_vars', {}),
        'CooldownGeoPRI': vm.get('cooldown_pri_ladder_vars', {}),
        'CooldownGeoNON': vm.get('cooldown_non_ladder_vars', {}),
        'RepeatOverPRI': vm.get('repeat_limit_pri_vars', {}),
        'RepeatOverNON': vm.get('repeat_limit_non_vars', {}),
        'BothFallback': vm.get('both_fallback_vars', {}),
        'PreferredMiss': vm.get('preferred_miss_vars', {}),
        'PriorityCoverage': priority_coverage,
        'OneTaskDay': vm.get('q_vars', {}),
        'TwoDaySoft': two_day_soft,
        'DeprioritizedPair': vm.get('deprioritized_pair_vars', {}),  # <-- NEW
        'EffortFloor': effort_floor_vars,
        'AutoDayMin': auto_day_min_vars,
        'AutoDayMinSunday': auto_day_min_sunday_vars,
        'DebugRelax': debug_relax_by_var,
        'DebugUnassigned': component_drop_by_var,
    }
    config = vm.get('config', {})
    return x_to_label, penalty_maps, config, penalty_weights, penalty_components


def load_components_info(path: Path):
    rows: List[Dict[str,str]] = []
    comp_info: Dict[str, Dict[str, float]] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(dict(r))
            cid = (r.get('ComponentId') or '').strip()
            if not cid:
                continue
            try:
                tc = float((r.get('Task Count') or '1').strip() or '1')
            except Exception:
                tc = 1.0
            try:
                eff = float((r.get('Total Effort') or '1').replace(',','.'))
            except Exception:
                eff = 1.0
            comp_info[cid] = {'taskcount': max(1.0, tc), 'effort': max(0.0, eff)}
    return rows, comp_info

def pick_weight_for_component(cid: str, metric: str, comp_info: Dict[str, Dict[str,float]]) -> float:
    d = comp_info.get(cid, {'taskcount':1.0,'effort':1.0})
    return 1.0 if metric == 'count' else (d['taskcount'] if metric == 'taskcount' else d['effort'])


DAY_ORDER = {
    day: idx
    for idx, day in enumerate(
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
    )
}


def _parse_week_num(raw: str | None) -> int:
    if not raw:
        return 0
    m = re.search(r"(\d+)", raw)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def select_direct_components(comps: List[str], comp_meta: Dict[str, Dict[str, str]]) -> List[str]:
    """Choose the components that *faced* a penalty.

    For multi-component penalties, the first chronological occurrence is treated as
    the baseline, and every later component (by week/day ordering) is counted as
    directly penalized. If no ordering information is available, fall back to
    attributing the penalty to every component present.
    """

    if not comps:
        return []
    if len(comps) == 1:
        return list(comps)

    def order_for(cid: str) -> Tuple[int, int, str]:
        meta = comp_meta.get(cid, {})
        week = _parse_week_num(meta.get("Week"))
        day_raw = (meta.get("Day") or "").strip().lower()
        day_rank = DAY_ORDER.get(day_raw, 0)
        return (week, day_rank, cid)

    ordered = [(order_for(cid), cid) for cid in comps]
    min_week_day = min((o[0], o[1]) for o, _ in ordered)
    direct = [cid for o, cid in ordered if (o[0], o[1]) > min_week_day]

    # If all items share the same ordering (no temporal info), attribute to all
    # components so the penalty remains visible in the direct columns.
    if not direct:
        return list(comps)
    return direct

# --- NEW: pull week number, SiblingKey families, and which cids were manual in the input ---
def extract_comp_meta(rows: List[Dict[str,str]]):
    def parse_week_num(week_label: str) -> int:
        s = (week_label or "")
        digits = "".join(ch for ch in s if s and ch.isdigit())
        return int(digits) if digits else 0

    comp_week: Dict[str,int] = {}
    comp_fams: Dict[str, List[str]] = {}
    manual_cids: Set[str] = set()

    for r in rows:
        cid = (r.get("ComponentId") or "").strip()
        if not cid: continue
        comp_week[cid] = parse_week_num(r.get("Week",""))
        sk = (r.get("SiblingKey") or "").strip()
        fams = [t.strip() for t in sk.split("||") if t.strip()] if sk else []
        comp_fams[cid] = fams if fams else [cid]  # fallback to per-cid family if no token
        if (r.get("Assigned?","").strip().upper() == "YES") and (r.get("Assigned To","").strip()):
            manual_cids.add(cid)
    return comp_week, comp_fams, manual_cids

# -------------------------- Decoding & fairness ------------------------
def decode_pairs(true_vars: List[str], x_to_label: Dict[str,str]) -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    for xv in true_vars:
        lbl = x_to_label.get(xv)
        if not lbl:
            continue
        parts = lbl.split("::")
        if len(parts) >= 3:
            cid, person = parts[1], parts[2]
            if cid and person:
                pairs.append((cid, person))
    return pairs

def compute_manual_loads(comp_rows: List[Dict[str,str]], comp_info: Dict[str, Dict[str,float]],
                        metric: str) -> Tuple[Set[str], Dict[str,float]]:
    manual_components: Set[str] = set()
    manual_loads: Dict[str,float] = {}

    for r in comp_rows:
        if (r.get("Assigned?","").strip().upper() == "YES") and (r.get("Assigned To","").strip()):
            cid = (r.get("ComponentId") or "").strip()
            person = (r.get("Assigned To") or "").strip()
            if cid:
                manual_components.add(cid)
            if cid and person:
                manual_loads[person] = manual_loads.get(person, 0.0) + pick_weight_for_component(cid, metric, comp_info)

    return manual_components, manual_loads


def compute_fairness(schedule_pairs: List[Tuple[str,str]], metric: str,
                     comp_info: Dict[str, Dict[str,float]], base_loads: Dict[str,float] | None = None
                     ) -> Tuple[Tuple[float,float,float], Dict[str,float]]:
    loads: Dict[str, float] = dict(base_loads) if base_loads else {}
    for cid, person in schedule_pairs:
        loads[person] = loads.get(person, 0.0) + pick_weight_for_component(cid, metric, comp_info)
    if not loads:
        return (0.0, 0.0, 0.0), {}
    vals = list(loads.values())
    mean = sum(vals) / len(vals)
    return (max(vals), sum(abs(v - mean) for v in vals), -min(vals)), loads

# --------------------- Penalty activation decoding ---------------------
def find_penalties(
    true_vars: List[str],
    penalty_maps: Dict[str, Dict[str, str]],
    manual_cids_from_input: Set[str] | None = None,
    penalty_weights: Dict[str, int] | None = None,
    x_to_label: Dict[str, str] | None = None,
    penalty_components: Dict[str, List[str]] | None = None,
):
    var_to_cat = {}
    var_to_label = {}
    for cat, m in penalty_maps.items():
        for v, label in m.items():
            var_to_cat[v] = cat
            var_to_label[v] = label

    activations = []
    unknown_true = []
    for v in true_vars:
        if v in var_to_cat:
            cat = var_to_cat[v]
            lbl = var_to_label.get(v, "")
            weight = (penalty_weights or {}).get(v)
            comps = (penalty_components or {}).get(v, [])

            # Skip DebugUnassigned penalties for components that were already
            # manually assigned in the input — they are not solver decisions.
            if (
                manual_cids_from_input
                and cat == "DebugUnassigned"
                and lbl.startswith("drop::")
            ):
                cid = lbl.split("::", 2)[1]
                if cid in manual_cids_from_input:
                    continue

            activations.append((v, cat, lbl, weight, comps))
        else:
            weight = (penalty_weights or {}).get(v)
            if weight is not None:
                lbl = (x_to_label or {}).get(v, "")
                comps = (penalty_components or {}).get(v, [])
                activations.append((v, "UnknownPenalty", lbl, weight, comps))
            else:
                unknown_true.append(v)

    counts: Dict[str,int] = {}
    for _, cat, _, _, _ in activations:
        counts[cat] = counts.get(cat, 0) + 1

    return activations, counts, unknown_true

# -------------------------- Main workflow ------------------------------
def main():
    args = parse_args()

    text = read_text_guess(args.models)
    models_true_vars = parse_models_from_text(text)
    if not models_true_vars:
        print("No models found in the provided file (no 'v ...' lines).", file=sys.stderr)
        sys.exit(1)

    x_to_label, penalty_maps, config, penalty_weights, penalty_components = read_varmap(args.varmap)
    penalty_totals = {cat: len(m) for cat, m in penalty_maps.items()}
    comp_rows, comp_info = load_components_info(args.components)
    comp_week, comp_fams, manual_cids_from_input = extract_comp_meta(comp_rows)
    comp_meta: Dict[str, Dict[str, str]] = {}
    for r in comp_rows:
        cid = (r.get("ComponentId") or "").strip()
        if not cid:
            continue
        comp_meta[cid] = {
            "Week": (r.get("Week") or "").strip(),
            "Day": (r.get("Day") or "").strip(),
            "Names": (r.get("Names") or "").strip(),
            "Total Effort": f"{comp_info.get(cid, {}).get('effort', 0.0):.4f}",
            "Task Count": f"{comp_info.get(cid, {}).get('taskcount', 0.0):.4f}",
        }
    manual_components_from_input, manual_loads_by_person = compute_manual_loads(comp_rows, comp_info, args.metric)

    models_summary: List[Dict[str,str]] = []
    best_idx = -1
    best_score = None
    best_pairs: List[Tuple[str,str]] = []
    best_loads: Dict[str,float] = {}
    best_penalty_acts = []
    best_penalty_counts = {}
    best_unknown_true = []

    for idx, true_vars in enumerate(models_true_vars, start=1):
        pairs = decode_pairs(true_vars, x_to_label)
        score, loads = compute_fairness(pairs, args.metric, comp_info, manual_loads_by_person)
        acts, counts, unknown_true = find_penalties(
            true_vars,
            penalty_maps,
            manual_cids_from_input,
            penalty_weights,
            x_to_label,
            penalty_components,
        )
        penalty_summary = "; ".join(
            f"{cat}:{label}" if label else cat for _, cat, label, _, _ in sorted(acts, key=lambda t: (t[1], t[0]))
        )

        models_summary.append({
            "idx": str(idx),
            "objective": "",  # numeric objective not parsed here
            "max_load": f"{score[0]:.6f}",
            "imbalance": f"{score[1]:.6f}",
            "min_load": f"{-score[2]:.6f}",
            "num_assignments": str(len(pairs)),
            "n_CooldownPRI": str(counts.get("CooldownPRI", 0)),
            "n_CooldownNON": str(counts.get("CooldownNON", 0)),
            "n_CooldownStreak": str(counts.get("CooldownStreak", 0)),
            "n_CooldownNonConsec": str(counts.get("CooldownNonConsec", 0)),
            "n_CooldownGeoPRI": str(counts.get("CooldownGeoPRI", 0)),
            "n_CooldownGeoNON": str(counts.get("CooldownGeoNON", 0)),
            "n_RepeatOverPRI": str(counts.get("RepeatOverPRI", 0)),
            "n_RepeatOverNON": str(counts.get("RepeatOverNON", 0)),
            "n_BothFallback": str(counts.get("BothFallback", 0)),
            "n_BothFallbackTotal": str(penalty_totals.get("BothFallback", 0)),
            "n_PreferredMiss": str(counts.get("PreferredMiss", 0)),
            "n_PriorityCoverage": str(counts.get("PriorityCoverage", 0)),
            "n_OneTaskDay": str(counts.get("OneTaskDay", 0)),
            "n_TwoDaySoft": str(counts.get("TwoDaySoft", 0)),  # <-- NEW
            "n_DeprioritizedPair": str(counts.get("DeprioritizedPair", 0)),  # <-- NEW
            "n_EffortFloor": str(counts.get("EffortFloor", 0)),
            "n_AutoDayMin": str(counts.get("AutoDayMin", 0)),
            "n_AutoDayMinSunday": str(counts.get("AutoDayMinSunday", 0)),
            "n_DebugRelax": str(counts.get("DebugRelax", 0)),
            "n_DebugUnassigned": str(counts.get("DebugUnassigned", 0)),
            "n_UnknownPenalty": str(counts.get("UnknownPenalty", 0)),
            "penalties": penalty_summary,
        })

        if best_score is None or score < best_score:
            best_idx, best_score = idx, score
            best_pairs, best_loads = pairs, loads
            best_penalty_acts, best_penalty_counts, best_unknown_true = acts, counts, unknown_true

    # models_summary.csv — add the new column to fieldnames
    with open(args.models_out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "idx","objective","max_load","imbalance","min_load","num_assignments",
            "n_CooldownPRI","n_CooldownNON","n_CooldownStreak","n_CooldownNonConsec",
            "n_CooldownGeoPRI","n_CooldownGeoNON","n_RepeatOverPRI","n_RepeatOverNON",
            "n_BothFallback","n_BothFallbackTotal","n_PreferredMiss","n_PriorityCoverage","n_OneTaskDay",
            "n_TwoDaySoft","n_DeprioritizedPair","n_EffortFloor","n_AutoDayMin","n_AutoDayMinSunday",
            "n_DebugRelax","n_DebugUnassigned","n_UnknownPenalty","penalties"  # <-- NEW columns
        ])
        w.writeheader()
        w.writerows(models_summary)

    # loads_by_person.csv — split Manual vs Auto
    auto_loads: Dict[str,float] = {}
    for cid, person in best_pairs:
        if cid in manual_components_from_input: continue
        auto_loads[person] = auto_loads.get(person, 0.0) + pick_weight_for_component(cid, args.metric, comp_info)

    all_people = sorted(set(list(manual_loads_by_person.keys()) + list(auto_loads.keys())))
    totals = {p: manual_loads_by_person.get(p,0.0) + auto_loads.get(p,0.0) for p in all_people}

    with open(args.loads_out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(["Person","ManualLoad","AutoLoad","TotalLoad"])
        for p in sorted(all_people, key=lambda x: (totals.get(x,0.0), x)):
            w.writerow([p,
                        f"{manual_loads_by_person.get(p,0.0):.6f}",
                        f"{auto_loads.get(p,0.0):.6f}",
                        f"{totals.get(p,0.0):.6f}"])

    # assigned_optimal.csv — fill only AUTO assignments
    cid_to_person: Dict[str,str] = {}
    for cid, person in best_pairs:
        if cid not in manual_components_from_input:
            cid_to_person[cid] = person

    with args.components.open('r', encoding='utf-8-sig', newline='') as fh:
        rdr = csv.DictReader(fh)
        out_fields = rdr.fieldnames or [
            "Kind","ComponentId","Week","Day","Repeat","RepeatMax","Task Count","Names",
            "Priority","Assigned?","Assigned To","Candidate Count","Candidates",
            "Role-Filtered Candidates","Total Effort"
        ]

    rows_out: List[Dict[str,str]] = []
    for row in comp_rows:
        r = dict(row)
        cid = (r.get("ComponentId") or "").strip()
        if cid and (cid in cid_to_person):
            r["Assigned?"] = "YES"
            r["Assigned To"] = cid_to_person[cid]
        rows_out.append(r)

    def sort_key(row: Dict[str,str]):
        pri = 0 if (row.get("Priority","").strip().upper() == "YES") else 1
        return (pri, row.get("Week",""), row.get("Day",""), row.get("ComponentId",""))
    rows_out.sort(key=sort_key)

    with open(args.assigned_out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in out_fields})

    # Component-level penalty rollup for the chosen model
    assignment_lookup: Dict[str, str] = {}
    for r in comp_rows:
        cid = (r.get("ComponentId") or "").strip()
        person = (r.get("Assigned To") or "").strip()
        if cid and person:
            assignment_lookup[cid] = person
    assignment_lookup.update(cid_to_person)

    comp_penalty_totals: Dict[str, int] = defaultdict(int)
    comp_penalty_details: Dict[str, List[str]] = defaultdict(list)
    comp_penalty_direct_totals: Dict[str, int] = defaultdict(int)
    comp_penalty_direct_details: Dict[str, List[str]] = defaultdict(list)
    for _, cat, label, weight, comps in best_penalty_acts:
        if not comps:
            continue
        detail = cat if not label else f"{cat}:{label}"
        if weight is not None:
            detail_with_weight = f"{detail} [w={weight}]"
        else:
            detail_with_weight = detail
        for cid in comps:
            if weight is not None:
                comp_penalty_totals[cid] += weight
            comp_penalty_details[cid].append(detail_with_weight)

        # "Direct" penalties are those borne by the later occurrences in a
        # sequence (e.g., the repeat or cooldown violators), not the baseline
        # component that merely existed earlier in time. If we cannot order the
        # components, attribute the penalty to all of them to keep visibility.
        direct_cids = select_direct_components(comps, comp_meta)
        for cid in direct_cids:
            if weight is not None:
                comp_penalty_direct_totals[cid] += weight
            comp_penalty_direct_details[cid].append(detail_with_weight)

    with open(args.component_penalties_out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow([
            "ComponentId",
            "Week",
            "Day",
            "Names",
            "TaskCount",
            "TotalEffort",
            "AssignedTo",
            "PenaltyWeightTotal",
            "Penalties",
            "DirectPenaltyWeight",
            "DirectPenalties",
        ])
        for cid in sorted(comp_penalty_details):
            meta = comp_meta.get(cid, {})
            w.writerow([
                cid,
                meta.get("Week", ""),
                meta.get("Day", ""),
                meta.get("Names", ""),
                meta.get("Task Count", ""),
                meta.get("Total Effort", ""),
                assignment_lookup.get(cid, ""),
                comp_penalty_totals.get(cid, 0),
                "; ".join(comp_penalty_details.get(cid, [])),
                comp_penalty_direct_totals.get(cid, 0),
                "; ".join(comp_penalty_direct_details.get(cid, [])),
            ])

    if comp_penalty_details:
        print(f"Wrote component penalty rollup → {args.component_penalties_out}")

    # Plots (best model)
    try:
        import matplotlib.pyplot as plt
        people_sorted = sorted(all_people, key=lambda p: (totals.get(p,0.0), p))
        manual_vals = [manual_loads_by_person.get(p, 0.0) for p in people_sorted]
        auto_vals   = [auto_loads.get(p, 0.0) for p in people_sorted]
        total_vals  = [totals.get(p, 0.0) for p in people_sorted]

        if args.plots_bars:
            plt.figure(figsize=(12, 5))
            plt.bar(people_sorted, manual_vals, label="Manual/Prepass")
            plt.bar(people_sorted, auto_vals, bottom=manual_vals, label="Auto")
            plt.xticks(rotation=60, ha='right')
            plt.ylabel(f"Load ({args.metric})")
            plt.title("Per-person load (chosen model) — ascending, layered")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plots_bars, dpi=160)
            plt.close('all')

        if args.plots_lorenz:
            xs = sorted(total_vals)
            cum = [0.0]; s = 0.0
            for v in xs: s += v; cum.append(s)
            if s > 0: cum = [c/s for c in cum]
            plt.figure(figsize=(6, 5))
            plt.plot([i/(len(xs)) for i in range(len(cum))], cum, marker='o')
            plt.plot([0,1],[0,1],'--')
            plt.xlabel("Fraction of people (sorted)")
            plt.ylabel("Fraction of total load")
            plt.title("Load distribution (Lorenz-like)")
            plt.tight_layout()
            plt.savefig(args.plots_lorenz, dpi=160)
            plt.close('all')
        if args.plots_bars:
            print(f"Wrote plots → {args.plots_bars} and {args.plots_lorenz}", file=sys.stderr)
    except Exception as e:
        print(f"[warn] Could not produce plots: {e}", file=sys.stderr)

    # --- reconstruct (person,family) chosen weeks and adjacent AUTO-involved pairs ---
    pf_weeks: Dict[Tuple[str,str], Dict[int,str]] = defaultdict(dict)  # (person,fam) -> {week: 'AUTO'|'MANUAL'}
    for cid, person in best_pairs:
        w = comp_week.get(cid, 0)
        fams = comp_fams.get(cid, [cid])
        side = 'AUTO' if cid not in manual_cids_from_input else 'MANUAL'
        for fam in fams:
            fam = fam.strip()
            if not fam: continue
            cur = pf_weeks[(person, fam)].get(w)
            if cur is None or cur == 'MANUAL':  # AUTO dominates
                pf_weeks[(person, fam)][w] = side

    pf_adj_pairs: Dict[Tuple[str,str], List[Tuple[int,int]]] = defaultdict(list)
    for key, weeks_map in pf_weeks.items():
        weeks_sorted = sorted(weeks_map.keys())
        for i in range(1, len(weeks_sorted)):
            w1, w2 = weeks_sorted[i-1], weeks_sorted[i]
            if w2 == w1 + 1:
                s1, s2 = weeks_map[w1], weeks_map[w2]
                if 'AUTO' in (s1, s2):  # same condition as encoder
                    pf_adj_pairs[key].append((w1, w2))

    # Debug table
    with open(args.cooldown_debug_out,'w',newline='',encoding='utf-8') as fh:
        wdbg = csv.writer(fh)
        wdbg.writerow(['Person','Family','Week','Side'])
        for (person,fam), m in sorted(pf_weeks.items()):
            for w, side in sorted(m.items()):
                wdbg.writerow([person, fam, f"W{w}", side])

    # ---------------- Penalties list (best model) ----------------
    if best_penalty_acts:
        # Parsers for added hints
        pref_k_re   = re.compile(r'::K=(\d+)\b')
        rep_over_re = re.compile(r'::t=(\d+).*?::limit=(\d+)\b', re.IGNORECASE)
        cool_t_re   = re.compile(r'::t=(\d+)\b', re.IGNORECASE)
        # Weeks pattern (for streak/basic cooldown labels that contain Wfrom->Wto)
        weeks_re    = re.compile(r'::W(\d+)->(\d+)\b')
        # person/family extraction from cooldown_geo labels
        person_re   = re.compile(r'::person=([^:]+)')
        family_re   = re.compile(r'::family=([^:]+)')

        with open(args.penalties_out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            # AdjPairs column shows WkFrom->WkTo pairs that fed each cooldown ladder
            w.writerow([
                "Var",
                "Category",
                "Label",
                "Weight",
                "Components",
                "IgnoredPairsK",
                "OverT",
                "OverLimit",
                "WeekFrom",
                "WeekTo",
                "AdjPairs",
            ])
            for v, cat, label, weight, comps in sorted(best_penalty_acts, key=lambda t: (t[1], t[0])):
                K = over_t = over_lim = wfrom = wto = adj_pairs = ""
                weight_str = "" if weight is None else str(weight)
                comp_str = ";".join(sorted(comps)) if comps else ""

                if cat == "PreferredMiss":
                    m = pref_k_re.search(label or "")
                    if m: K = m.group(1)

                if cat in ("RepeatOverPRI","RepeatOverNON"):
                    m = rep_over_re.search(label or "")
                    if m:
                        over_t, over_lim = m.group(1), m.group(2)

                if cat in ("CooldownGeoPRI","CooldownGeoNON"):
                    m = cool_t_re.search(label or "");
                    if m: over_t = m.group(1)
                    mp = person_re.search(label or "")
                    mf = family_re.search(label or "")
                    if mp and mf:
                        key = (mp.group(1), mf.group(1))
                        pairs = pf_adj_pairs.get(key, [])
                        if pairs:
                            adj_pairs = ";".join(f"W{a}->W{b}" for (a,b) in pairs)

                if cat in ("CooldownPRI","CooldownNON","CooldownStreak"):
                    m = weeks_re.search(label or "")
                    if m:
                        wfrom, wto = f"W{m.group(1)}", f"W{m.group(2)}"

                w.writerow([v, cat, label, weight_str, comp_str, K, over_t, over_lim, wfrom, wto, adj_pairs])

        print(f"Wrote penalty activations → {args.penalties_out}")
        print(f"Wrote cooldown debug → {args.cooldown_debug_out}")

    both_total = penalty_totals.get("BothFallback", 0)
    if both_total:
        activated = best_penalty_counts.get("BothFallback", 0)
        print(f"[info] Both fallback selectors encoded: {both_total}; activated in chosen model: {activated}")
    else:
        print("[info] No Both fallback selectors were encoded for this input.")

    # Pretty print best model’s weight-impacting decisions
    print("\n=== Weight-impacting decisions in chosen model ===")
    if not best_penalty_acts:
        print("(none)")
    else:
        pref_k_re   = re.compile(r'::K=(\d+)\b')
        rep_over_re = re.compile(r'::t=(\d+).*?::limit=(\d+)\b', re.IGNORECASE)
        cool_t_re   = re.compile(r'::t=(\d+)\b', re.IGNORECASE)
        weeks_re    = re.compile(r'::W(\d+)->(\d+)\b')
        person_re   = re.compile(r'::person=([^:]+)')
        family_re   = re.compile(r'::family=([^:]+)')

        by_cat: Dict[str, List[Tuple[str,str,str]]] = {}
        for v, cat, label, weight, comps in best_penalty_acts:
            hint_parts: List[str] = []
            if weight is not None:
                hint_parts.append(f"w={weight}")
            if comps:
                hint_parts.append(f"components={'+'.join(sorted(comps))}")

            if cat == "PreferredMiss":
                m = pref_k_re.search(label or "")
                if m: hint_parts.append(f"ignored_pairs={m.group(1)}")

            if cat in ("RepeatOverPRI","RepeatOverNON"):
                m = rep_over_re.search(label or "")
                if m:
                    over_by = int(m.group(1)) - int(m.group(2))
                    if over_by >= 1:
                        hint_parts.append(f"over_by={over_by}")

            if cat in ("CooldownGeoPRI","CooldownGeoNON"):
                m = cool_t_re.search(label or "")
                if m: hint_parts.append(f"ladder_step={m.group(1)}")
                mp = person_re.search(label or "")
                mf = family_re.search(label or "")
                if mp and mf:
                    key = (mp.group(1), mf.group(1))
                    pairs = pf_adj_pairs.get(key, [])
                    if pairs:
                        adj = ";".join(f"W{a}->W{b}" for (a,b) in pairs)
                        hint_parts.append(f"adj={adj}")

            if cat in ("CooldownPRI","CooldownNON","CooldownStreak"):
                m = weeks_re.search(label or "")
                if m: hint_parts.append(f"weeks=W{m.group(1)}->W{m.group(2)}")

            hint = f" ({', '.join(hint_parts)})" if hint_parts else ""
            by_cat.setdefault(cat, []).append((v, label, hint))

        for cat in sorted(by_cat.keys()):
            print(f"\n[{cat}]  count={len(by_cat[cat])}")
            for v, label, hint in sorted(by_cat[cat], key=lambda x: x[0]):
                print(f"  {v}: {label}{hint}")

    unknown_filtered = [v for v in best_unknown_true if v not in x_to_label]
    if unknown_filtered:
        print("\n[info] True variables not recognized in varmap maps (not x_to_label/penalties).")
        print("       (This list can include Tier-6 load indicators and debug selectors.)")
        for v in unknown_filtered[:50]:
            print(f"  {v}")
        if len(unknown_filtered) > 50:
            print(f"  ... (+{len(unknown_filtered)-50} more)")

    print(f"\nSelected fairest model idx={best_idx} score={best_score}")

if __name__ == '__main__':
    main()
