# Assignment rules

This document summarizes every rule enforced by `encode_sat_from_components.py`, split between hard constraints (infeasible models) and soft penalties (objective costs). The encoder always encodes tasks at the component level; the solver picks exactly one person per component while respecting the limits described below.

## Hard constraints

1. **Component coverage** – Every component must be assigned to exactly one person. Manual assignments ship as fixed literals, so the solver only decides AUTO slots. (See `encode_sat_from_components.py` lines 718‑786.)
2. **Candidate legality** – Only eligible people (time/role availability plus "Both" role expansions and sibling carry-overs) can be picked. Anyone outside the candidate list is never encoded for that component. (Lines 324‑420.)
3. **Same-day bans only** – The solver may assign multiple components to the same person on a single `(week, day)` as long as those components are not explicitly marked as mutually exclusive. The only hard blocks are the configured sibling-pair bans (`BANNED_SIBLING_PAIRS`) and same-day bans (`BANNED_SAME_DAY_PAIRS`), plus any deprioritized-pair soft costs described later. (Lines 827‑972.)
4. **Two-day rule** – Depending on the config, Sunday-inclusive pairs or all cross-day combinations are either hard-banned or softened (see the soft section). When `SUNDAY_TWO_DAY_SOFT=False` and `TWO_DAY_SOFT_ALL=False` the encoder emits hard mutex constraints. (Lines 1044‑1148.)
5. **Cooldown gates** – When `DEBUG_RELAX` is false the cooldown/repeat ladders rely on auxiliary selectors, but the presence of those selectors does not relax the hard constraints that enforce weekly cooldown counts, weekly workload ceilings, and streak detection. Violations are only allowed via explicit soft selectors. (Lines 1180‑1443.)
6. **Priority assignments** – Components flagged as `Assigned?` in the CSV stick to the provided person, so the solver cannot move them. These assignments also satisfy the "priority coverage" counters when applicable. (Lines 698‑748.)
7. **Model sanity** – Every SAT variable introduced by the encoder corresponds to either a binary assignment literal or a selector tied to a constraint. Debug relax selectors are only added when `DEBUG_RELAX=True` to explain infeasible combinations without silently dropping rules. (Lines 520‑612.)

## Soft penalties (objective tiers)

1. **Tier 1 (Priority families)**
   - *Cooldown ladder* (`W1_COOLDOWN`) – penalizes consecutive priority-family assignments that violate cooldown spacing.
   - *Repeat-over ladder* (`W1_REPEAT`) – exponential penalty (`REPEAT_OVER_GEO` base) once a person exceeds `REPEAT_LIMIT['PRI']` for the same family.
   - *Streak penalty* (`W1_STREAK`) – discourages back-to-back weeks on the same priority family.
   - *Priority miss guard* (`W_PRIORITY_MISS`) – per-person selector that fires when someone who could have taken a top-priority task receives none. When set, this weight **replaces** the `T1C` top-coverage weight instead of stacking with it, so skipping an eligible dancer is still expensive without double-charging the objective.

2. **Tier 2 (Non-priority families)**
   - Identical cooldown/repeat/streak ladders with weights `W2_COOLDOWN`, `W2_REPEAT`, `W2_STREAK`.
   - *"Both" fallback* (`W4`) – soft cost when the solver has to treat a "Both" person as both leader and follower on the same slot.

3. **Tier 3** – *Same-day nudger* (`W3`) encourages filling a second manual-only task on thin days.

4. **Tier 4** – *Deprioritized pairs* (`W4_DPR`) penalize assigning the same person to a deprioritized unordered pair on the same day.

5. **Tier 5** – *Preferred pair miss* (`W5`) charges for every feasible preferred pair that fails to appear.

6. **Tier 6 (Fairness)** – Convex ladders `W6_OVER` / `W6_UNDER` keep each person’s task load near the global mean (the defaults now keep `FAIR_MEAN_MULTIPLIER = 1.0` and `FAIR_OVER_START_DELTA = 0`, so the baseline target is the unmodified global average before availability scaling kicks in). Under-load penalties only fire when a person falls far below the mean, while over-load ladders grow quadratically to stop hoarding. (Lines 1662‑1915.)

7. **Priority coverage** – Selectors `priority_coverage_vars_top` (`T1C`) and `priority_coverage_vars_second` (`T2C`) encourage spreading top/second priority tasks across different people or families (configurable via `PRIORITY_COVERAGE_MODE`). These are *soft incentives*; they do **not** force every eligible dancer to receive a priority task. Instead they reward models that cover more distinct names. (Lines 1451‑1582.)

8. **Two-day softening** – When `SUNDAY_TWO_DAY_SOFT=True` or `TWO_DAY_SOFT_ALL=True`, violating the two-day ban adds selectors at weights `W_SUNDAY_TWO_DAY` / `W_TWO_DAY_SOFT` instead of hard-failing the model. The defaults keep `W_TWO_DAY_SOFT = 2e9` and `W_SUNDAY_TWO_DAY = 1.5e9`, which deliberately outrank the coverage selectors (`T1C = 1e9`, `T2C = 5e8`) so day-violation penalties stay stronger than the incentive to widen priority coverage. (Lines 1044‑1148.)

9. **Sunday rule softening** – The dedicated Sunday ladder uses `W_SUNDAY_TWO_DAY` so you can independently adjust that behavior even when weekday pairs stay hard-banned. (Lines 1050‑1102.)

10. **Visualization nudges** – Not directly part of the objective, but `visualize_components.py` now highlights scarce nodes via candidate-count heatmaps so you can visually inspect which rules (repeat/cooldown vs. fairness vs. two-day) are likely to bite after encoding.

## Priority-task coverage FAQ

*Is there a rule that guarantees every person who can take a priority task actually gets one?*

Not as a hard constraint, but the coverage selectors make it costly to skip them. `T1C`/`T2C` reward spreading top/second-priority tasks across people or families. If you also set `W_PRIORITY_MISS`, that weight replaces `T1C` for the top tier so each eligible person incurs **one** strong penalty when they receive zero top tasks. The guard remains a soft penalty rather than a ban, so the solver can still leave someone without a top slot when no feasible assignment exists—but without double-counting the miss. Inspect `priority_required_vars` plus the coverage maps in `varmap.json` to see which selectors fired.

## Improving and verifying the rules

- **Availability-aware fairness targets** – Tier-6 now scales each person’s load target by how many AUTO components they could legally cover. Check `varmap.json.fairness_targets` and the fairness block in `stats.txt` to confirm the scaling. Raising the `FAIRNESS_AVAILABILITY` exponent or ratio cap makes high-capacity dancers pick up more work; lowering them relaxes the distribution.
- **Auto-softening for scarce families** – Families with too few candidates automatically skip the harshest cooldown/repeat penalties. Confirm the skip list in `stats.txt` or `varmap.json.auto_soften_families` before chasing phantom penalties.
- **Assignment reports** – After solving, run `python3 report_assignments.py` so `reports/assignment_report.csv` lists per-person totals, repeats, and priority eligibility. That report is the fastest way to verify that fairness dials and priority selectors are delivering the intended distribution without digging through raw solver CSVs.
- **Test suite** – `pytest` exercises every major rule combination (two-day modes, auto-softening, repeat gating, priority coverage, fairness scaling, and the reporting pipeline). If you tweak a rule, add a fixture that reproduces the scenario so regressions show up immediately.

## Weight prioritization guidelines

- **Tier 1 (priority families) must stay dominant.** Priority cooldown/repeat/streak weights (`W1_*`) should remain at least 1–2 orders of magnitude above every lower tier so the solver never sacrifices coverage or spacing on priority families just to improve fairness or preferred-pair scores. Use the same geometric ratio inside the tier (e.g., `W1_STREAK = 10× W1_REPEAT = 10× W1_COOLDOWN`) so streaks are the most expensive pattern within priority families. 【F:docs/rules.md†L22-L31】
- **Tier 2 mirrors Tier 1 but can be slightly softer.** Keep `W2_*` within an order of magnitude of `W1_*` to avoid runaway repeats on non-priority families, yet low enough that priority penalties always dominate. When scarce families trigger auto-softening (see `varmap.json.auto_soften_families`), you can safely lower the corresponding `W2_*` dials without compromising feasibility. 【F:docs/rules.md†L22-L52】
- **Mid-tier nudges should be 3–5 orders smaller.** Weights for the same-day nudger, deprioritized pairs, “Both” fallback, and preferred-pair misses (`W3`, `W4`, `W4_DPR`, `W5`) should sit far below cooldown/repeat penalties so they only break ties between otherwise-equal schedules. A common pattern is `W3 ≈ 10^6`, `W4 ≈ 10^12`, `W4_DPR ≈ 10^12`, `W5 ≈ 10^3`, keeping them visible but never ahead of Tier 1/2 ladders. 【F:docs/rules.md†L24-L35】
- **Tier 6 fairness should rival Tier 2 once scaled.** After availability scaling, set `W6_OVER`/`W6_UNDER` so that exceeding a personalized load target by ~2 tasks costs roughly the same as another non-priority repeat hit. This keeps fairness competitive without making it impossible to staff scarce families. The defaults now pin `FAIR_MEAN_MULTIPLIER = 1.0` and `FAIR_OVER_START_DELTA = 0` so the global mean itself becomes the baseline target before availability multipliers stretch or shrink it. Adjust those knobs only when you intentionally want to bias everyone above/below the global average. 【F:docs/rules.md†L32-L52】
- **Two-day penalties outrank coverage selectors.** Keep `W_TWO_DAY_SOFT` above `W_SUNDAY_TWO_DAY`, and both above the coverage weights (`T1C`, `T2C`). The current defaults follow `2e9 > 1.5e9 > 1e9 > 5e8`, ensuring that violating a two-day ban is always costlier than duplicating coverage on a top/second-priority task. 【F:docs/rules.md†L32-L52】
- **Priority coverage weights live between Tier 2 and Tier 6.** Because `T1C`/`T2C` only reward broader coverage, keep them high enough to discourage duplicate assignments (≈10^7–10^9) but below cooldown/repeat ladders so they yield when a coverage-perfect solution would blow up a higher-tier constraint. 【F:docs/rules.md†L34-L35】
- **Two-day soft penalties stay just below Tier 2.** When the two-day rule is softened, `W_SUNDAY_TWO_DAY` and `W_TWO_DAY_SOFT` should sit slightly under `W2_*` so the solver prefers legal day combinations but can still violate the rule when the schedule would otherwise be infeasible. 【F:docs/rules.md†L36-L38】

Treat these ratios as guardrails: the exact magnitudes can shift per dataset, but keeping the orders of magnitude aligned with their tiers ensures the solver resolves hard-priority conflicts before it chases fairness, pair preferences, or visualization-driven nudges.
