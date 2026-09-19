# Roadmap

What we are building, in the order we will build it. Agreed 2026-09-19.

Status values: `todo`, `wip`, `done`, `blocked`. Update the status column as work lands —
this file is the shared plan, `docs/decisions.md` is the record of why.

**The critical path is tasks 6 → 10 → 15.** Task 6 gates every context feature, task 10 is the
project's actual thesis, and task 15 is the result the report is built on. Everything else can
move around those three.

---

## Phase 0 — unblock

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 1 | Collaborator access and local git auth | Yunus | wip | Permission granted on GitHub. Local `credential.helper` was unset so git never called Git Credential Manager; now set repo-locally. First push opens a browser login. |
| 2 | Add `.gitignore`, untrack `src/pvf.egg-info/` | Yunus | todo | Nothing currently stops `git add .` from committing 200 MB of parquet. `data/`, `.venv/`, `__pycache__/`, `*.egg-info/`. |
| 3 | Commit `docs/decisions.md` and `docs/data_coverage.md` | Yunus | wip | Decision log written 2026-09-19. |

## Phase 1 — data layer

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 4 | Update `configs/config.yaml` | Yunus | done | 9 league ids and `min_anchor_season: 2013` set (decisions #3, #4, #5). Test seasons still open, see item F. |
| 5 | Finish `src/pvf/data/load.py` | | todo | Read from `FV_DATA_DIR` so the path works on both machines. Parse the text dates once. Drop the ~73% duplicate rows in `team_competitions_seasons`. |
| 6 | **Build the club → league → season map from `games`** | Yunus | wip | Module and tests done: `src/pvf/features/club_league.py`, 8 tests green. Decisions #14–16 came out of it. Still to do: validate against the real parquet (a full pass over `game_lineups`, ~3.2M rows), then wire into `build_panel.py` slot 6 with task 7. |

## Phase 2 — the player-season panel

Jack left numbered slots in `src/pvf/features/build_panel.py`; these fill them in order.

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 7 | Slot 1, player features | Yunus | done | `add_player_features` (age, position, foot, height, nationality, EU flag) plus `anchor_population`, which replaces the old cross join and so wires slot 6 in at the same time. 8 tests. Not yet run against real parquet. |
| 8 | Slot 2, value history | | todo | Current value, 12-month change, peak so far, days since last update, `years_of_history`. Null for newcomers by design (decision #6). |
| 9 | Slot 3, last season's performance | | todo | Minutes, share of starts, goals and assists per 90. No cards (decision #12). |
| 10 | Slot 4, context | | todo | **The project's thesis.** League strength, club strength rebuilt per club-year (decision #10), European participation from `games` (decision #11), value rank among same-position players at club and in league. |
| 11 | Slot 5, health and moves | | todo | Injury aggregates from dates and durations, not the reason text (decision #7). Transfer flags and fees from `transfer_history`. |
| 12 | Extend `tests/test_leakage.py` | | todo | Assert every new feature's source columns against `leakage.py`. Run it as each slot lands, not at the end. |

## Phase 3 — models and evaluation

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 13 | Baselines | | todo | No change, and the age × position curve. The model has to beat these or there is no story. |
| 14 | LightGBM per horizon, plus quantile models | | todo | One model per horizon (1, 2, 3). Quantiles 10/50/90 give the GUI a range. |
| 15 | **Time-split backtest and ablation** | | todo | Remove context, performance and injuries one group at a time. This answers "does context beat age plus current value?" — the headline result. |

## Phase 4 — delivery

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 16 | Export the prediction bundle | | todo | To the schema already defined in `src/pvf/export/schema.py`. The app reads it unchanged. |
| 17 | Report | | todo | Framed around the task 15 ablation. |

## Running alongside

| # | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| 18 | Sofascore ratings experiment | Yunus | todo | Download the Kaggle set, join on name + birth year to the top-5 subset, ablate. The result decides whether we invest in the full scrape (decision #8). |
| 19 | **Manual dataset, 500+ rows** | | todo | **Rubric requirement and currently blocking.** The only task here with no dependencies, so it can start immediately. Candidates in `docs/decisions.md`; the human-baseline-forecast option is strongest because it doubles as a benchmark for the report. |
| 20 | Confirm the second model family | | todo | `chronos` is proposed in the config. Check it against the rubric. |
| 21 | Deadline, deliverables, rubric questions | | todo | Parked 2026-09-19. The 500-vs-1000 sample discrepancy needs an instructor answer. |

---

## Who owns what

Jack owns the GUI and the prediction-bundle contract. Yunus owns the data pipeline and modelling.
The bundle schema in `src/pvf/export/schema.py` is the interface between the two, so either side
can work without waiting on the other — change it only by agreement.
