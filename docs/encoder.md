# Encoder and solver workflow

This document describes how the tango task distributor builds the pseudo-Boolean model and how to interpret the debug artifacts that the encoder writes next to `schedule.opb`.

## High-level pipeline

1. `extractor.py` gathers the Google Sheet data locally (pre-built CSV snapshots already ship in the repo).
2. `encode_sat_from_components.py` reads `components_all.csv` and `backend.csv`, converts each component into decision variables, and emits:
   - `schedule.opb` – the SAT4J objective/constraint file.
   - `varmap.json` – human-readable names for every generated variable as well as metadata (weights, penalty tiers, scarcity notes, etc.).
   - `stats.txt` – a prose summary of the important knobs, automatically including which families are auto-softened.
3. `run_solver.py` executes SAT4J with a 120 second timeout so long solver runs do not hang the workflow. When that limit fires, the wrapper now sends `SIGINT` (same as pressing <kbd>Ctrl+C</kbd>) so SAT4J can flush the best-so-far model before being force-killed. Every `v ...` model is written to `models.txt`, and the wrapper immediately calls `consume_saved_models.py` so the fairness plots and CSV summaries are regenerated without manual intervention.

## Auto-softening for scarce families

The encoder now inspects each sibling family and counts:

- how many distinct people can legally take any component from that family;
- how many component slots (across all weeks) belong to that family.

If a family has `<=3` unique eligible people *or* needs more than `1.5` slots per candidate, it is marked as "scarce". Scarce families skip the heaviest repeat-limit and cooldown penalties (still respecting exact-one, day rules, etc.) so the solver is not punished for unavoidable repeats. Details for every skipped family live in both `varmap.json.auto_soften_families` and `stats.txt`.

## Adding candidates automatically

The encoder already merges "Both"-role expansions, manual overrides, and sibling move links so that repeating components (leader/follower pairs) inherit each other’s candidate pools. `auto_soften_families` works on top of that deduplicated candidate view, so widening candidate pools upstream will automatically change the scarcity classification without touching code.

## Debug artifacts worth checking

| File | Purpose |
| --- | --- |
| `stats.txt` | Narrative summary including repeat/cooldown ladders, fairness mean, and now the auto-soften report. |
| `varmap.json` | Contains every objective selector name and the config snapshot. Search for `auto_soften_families` to see which families were relieved. |
| `penalties_activated.csv` | Generated after solving; lists which ladder variables fired and how often. Combine with the scarcity report to see if the solver is still bottlenecked elsewhere. |

## Automatic evaluation after solving

`run_solver.py` can be treated as a single-button "solve + analyze" step:

```bash
python3 run_solver.py --opb schedule.opb --metric effort --log logs/solver.log
```

- `models.txt` is populated with every `v ...` line that the solver printed (best model last).
- `assigned_optimal.csv`, `models_summary.csv`, `loads_by_person.csv`, `penalties_activated.csv`, and the fairness plots are refreshed immediately via `consume_saved_models.py`.
- Use `--skip-consume` if you only want the solver output, or change `--metric` when you want the fairness charts to be based on task count vs. effort.
- When the timeout fires before SAT4J emits a `v ...` assignment, the wrapper first issues `SIGINT` to request the best-so-far model, waits `--interrupt-grace` seconds (10s default), and only then force-kills the solver. Regardless of whether a model arrives, the log is preserved and the previous CSVs stay untouched so you can rerun with a longer limit.

## Running the solver safely

Run SAT4J through the wrapper so it never blocks the rest of the pipeline:

```bash
python3 run_solver.py --opb schedule.opb --log logs/solver.log
```

- Use `--timeout` if you want to shorten/extend the 120 second default.
- On timeout the wrapper now delivers `SIGINT`, waits `--interrupt-grace` seconds for SAT4J to flush a model, and returns exit code `124` if the solver never finished cleanly. The partial log is always left intact.
- All command-line arguments are optional; the defaults match the repository layout described in the root `readme`.

## Tuning tips

1. Check `stats.txt` for the total number of cooldown/repeat selectors per tier. If most of them belong to non-scarce families, widen candidate pools or bump the scarcity thresholds.
2. Inspect `loads_by_person.csv` to ensure fairness ladders are not fighting unavoidable manual assignments. When necessary, temporarily disable Tier-6 by setting `W6_OVER = W6_UNDER = 0` and re-running the encoder.
3. Use `varmap.json.cooldown_gate_info` to confirm every cooldown penalty had an AUTO gate (either current or previous week). If not, the component metadata might be missing `Assigned? = YES` entries for manual tasks.
