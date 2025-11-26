# Penalty catalogue

This guide explains every penalty category that can appear in `penalties_activated.csv`, how the encoder creates the selector, and how to read the label/columns in the CSV. The category names below match the `Category` column emitted by `consume_saved_models.py`.

## Coverage and priority guards

- **PriorityCoverage (TOP/SECOND)** – Encourages spreading top/second-priority tasks. A selector is created per person (or per person/family in family mode) when they are eligible for a top or second task but get none. The weight is `W_PRIORITY_MISS` when set, otherwise `T1C` for top and `T2C` for second. Labels look like `priority_coverage_TOP::GLOBAL::person=…` or `priority_coverage_SECOND(NOT_TOP)::FAMILY::person=…::family=…`.
- **PriorityRequired** – When `W_PRIORITY_MISS` is positive, the encoder also records `priority_required::person=…` selectors in `varmap.json` to show who was eligible. These are metadata only; the penalty uses the PriorityCoverage selector above.

## Cooldown and streak spacing

- **CooldownPRI / CooldownNON** – Inter-week cooldown gates that turn on when the same person is assigned to the same family in consecutive weeks and at least one of those assignments is AUTO. `PRI` variants require at least one week to be a priority component; `NON` fire when neither week is priority. Labels look like `cooldown_prev::PRI::fam=…::W12->W13::Person` and populate the `WeekFrom`/`WeekTo` columns.
- **CooldownGeoPRI / CooldownGeoNON** – Geometric ladders built from the above cooldown gates. For each `(person,family)` they count how many cooldown gates are active and add increasing powers of `COOLDOWN_GEO` starting at `t=1`. Labels show `::t=<step>::<total>`, and `OverT` in the CSV holds that step. When the label contains `INTRA_WEEK`, the ladder counts distinct **days within the same week** instead of consecutive weeks.
- **CooldownStreak** – Penalizes back-to-back weeks in the same family (priority or non-priority), even if they are already part of the geometric ladder. A streak selector is an `AND` of two consecutive cooldown gates and is weighted by `W1_STREAK` or `W2_STREAK` depending on priority status. Labels look like `vprev::REPEAT::STREAK::PRI::fam=…::W12->W13::Person`.

## Repeat discouragers

- **RepeatOverPRI / RepeatOverNON** – For each `(person,family)` the encoder counts assignments and starts charging once the count exceeds `REPEAT_LIMIT['PRI'|'NON']`. The ladder is geometric with base `REPEAT_OVER_GEO`; labels include `::t=<over_count>::limit=<limit>`, and `OverT`/`OverLimit` in the CSV mirror those numbers. Manual-only families are skipped.

## Day-level rules

- **TwoDaySoft** – When Sunday-only or all cross-day bans are softened, violating the two-day rule emits selectors weighted by `W_SUNDAY_TWO_DAY` or `W_TWO_DAY_SOFT`. Labels look like `two_day_soft::W13::Person::Tuesday+Wednesday` or `sunday_two_day_soft::W11::Person::Sunday+Tuesday`. `WeekFrom`/`WeekTo` stay blank because the rule is per-week.
- **OneTaskDay** – The “fill to two” nudger (`W3`) that encourages a person on manual-only days to take a second task. Labels point to the `(week, day, person)` triple; the category appears as `OneTaskDay` in the CSV.

## Pair preferences and conflicts

- **PreferredMiss** – For each preferred unordered pair of tasks, the encoder checks whether both tasks were taken by the same person. If not, it adds a penalty weighted by `W5 * K`, where `K` is the size of the feasible pair set. Labels list the group (week/day/task names) and `K=<count>`, which is also copied into the `IgnoredPairsK` column.
- **DeprioritizedPair** – Soft cost (`W4_DPR`) when the same person takes a deprioritized pair of tasks on the same day. Labels show day, week, person, and the two component IDs.
- **BothFallback** – When a role-flexible (“Both”) person is used to cover both sides of a leader/follower split in the same family, the fallback selector fires at weight `W4`. Labels list the person and family token.

## Fairness ladders

- **Fairness over/under load** – Tier‑6 convex ladders (`W6_OVER`, `W6_UNDER`) penalize loads above or below the personalized target derived from the global mean and availability scaling. These selectors are not written to `penalties_activated.csv` by default because they would dominate the table; their targets and hits appear in `loads_by_person.csv` and `stats.txt` instead. The ladder weights follow the same geometric base as repeat penalties.

## Debugging aids

- **DebugRelax** – Only present when `DEBUG_RELAX=True`. Each hard constraint gets a relax selector so infeasibilities can be diagnosed. The category surfaces in `penalties_activated.csv` with labels that mirror the relaxed constraint.

## Reading `penalties_activated.csv`

Columns are derived from the selector label when possible:
- `Weight` is the exact numeric weight from the objective for that selector.
- `OverT`/`OverLimit` come from repeat ladders (`t=` and `limit=`). For cooldown ladders `OverT` is the ladder step.
- `WeekFrom`/`WeekTo` are parsed from labels that contain `W<from>->W<to>` (cooldown and streak). Intra-week cooldowns omit these.
- `AdjPairs` lists consecutive week pairs involving the same `(person,family)` where at least one assignment was AUTO; it helps explain why cooldown ladders grew.
- `IgnoredPairsK` is only populated for preferred-pair misses and matches the `K` value in the label.

Use this catalogue to map any `Category/Label` pair back to the encoder rule and weight that produced it.
