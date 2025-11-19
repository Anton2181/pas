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

2. **Tier 2 (Non-priority families)**
   - Identical cooldown/repeat/streak ladders with weights `W2_COOLDOWN`, `W2_REPEAT`, `W2_STREAK`.
   - *"Both" fallback* (`W4`) – soft cost when the solver has to treat a "Both" person as both leader and follower on the same slot.

3. **Tier 3** – *Same-day nudger* (`W3`) encourages filling a second manual-only task on thin days.

4. **Tier 4** – *Deprioritized pairs* (`W4_DPR`) penalize assigning the same person to a deprioritized unordered pair on the same day.

5. **Tier 5** – *Preferred pair miss* (`W5`) charges for every feasible preferred pair that fails to appear.

6. **Tier 6 (Fairness)** – Convex ladders `W6_OVER` / `W6_UNDER` keep each person’s task load near the global mean (scaled by `FAIR_MEAN_MULTIPLIER` and shifted by `FAIR_OVER_START_DELTA`). Under-load penalties only fire when a person falls far below the mean, while over-load ladders grow quadratically to stop hoarding. (Lines 1662‑1915.)

7. **Priority coverage** – Selectors `priority_coverage_vars_top` (`T1C`) and `priority_coverage_vars_second` (`T2C`) encourage spreading top/second priority tasks across different people or families (configurable via `PRIORITY_COVERAGE_MODE`). These are *soft incentives*; they do **not** force every eligible dancer to receive a priority task. Instead they reward models that cover more distinct names. (Lines 1451‑1582.)

8. **Two-day softening** – When `SUNDAY_TWO_DAY_SOFT=True` or `TWO_DAY_SOFT_ALL=True`, violating the two-day ban adds selectors at weights `W_SUNDAY_TWO_DAY` / `W_TWO_DAY_SOFT` instead of hard-failing the model. (Lines 1044‑1148.)

9. **Sunday rule softening** – The dedicated Sunday ladder uses `W_SUNDAY_TWO_DAY` so you can independently adjust that behavior even when weekday pairs stay hard-banned. (Lines 1050‑1102.)

10. **Visualization nudges** – Not directly part of the objective, but `visualize_components.py` now highlights scarce nodes via candidate-count heatmaps so you can visually inspect which rules (repeat/cooldown vs. fairness vs. two-day) are likely to bite after encoding.

## Priority-task coverage FAQ

*Is there a rule that guarantees every person who can take a priority task actually gets one?*

No. The encoder tracks a list of top- and second-priority **task names** from `backend.csv` and adds soft selectors that reward solutions covering more distinct people/families for those tasks. If a person is eligible for a priority task but no component using that task name can be assigned to them without breaking a higher-tier constraint (like cooldown spacing or fairness ladders), the solver may leave them without a priority slot. The coverage selectors only discourage duplicate assignments; they do not require one-per-person coverage. See `priority_coverage_vars_top` and `priority_coverage_vars_second` in `varmap.json` for the actual selectors tied to weights `T1C` and `T2C`.
