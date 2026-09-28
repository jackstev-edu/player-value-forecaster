# Scope change: the 500 hand-collected contracts become the dataset

Agreed by Yunus on 2026-09-27, branch `scope/manual-contracts`. Needs Jack's review in the PR.
Decisions #36–#42 in `docs/decisions.md` record each choice; this page explains the whole change.

## What changed, in one paragraph

Until now the project modelled a 70,150-row player-season panel built from 11 Kaggle tables
(9 leagues, seasons 2012–2025, around 40 features). The 500 contract dates we collected by hand
(Task 19) were planned as a side study. **They are now the dataset.** Each of the 500 players has
one hand-checked contract. We keep the Kaggle valuations only to measure the value before and
after the anchor, and the Kaggle players table only for age and position. We augment the
500 rows to 1,000 samples, as the course asks, by reusing each contract for its other seasons.

## Why

- The rubric wants a dataset we collected ourselves (500 minimum, 1,000 in the heading).
  Making the hand-collected rows the core puts that data at the centre of the project, not beside it.
- Contract length is the feature the Kaggle data could not give us without leakage: its contract
  columns are a snapshot from today, so they include extensions signed after each anchor.
  The hand-collected dates are what was in force on the anchor.
- The pipeline becomes small enough to explain in the report and test fully.

## The pipeline now

```
Drive: Contract Data Collection.xlsx (500 rows, filled by Yunus and Jack)
  │  scripts/build_dataset.py --sheet ...        export, dropping the Kaggle hint columns
  ▼
data/manual/contracts.csv  ──►  pvf.data.manual.check_contracts  ──►  data/manual/contract_issues.csv
  │  usable_contracts: drop rows with an error
  ▼
pvf.data.augment.span_contracts   extra 1 July anchors inside each contract (origin = "span")
  ▼
pvf.features.samples.build_samples   features at the anchor + y_h1 from Kaggle valuations
  ▼
pvf.data.augment.shift_unanchored / backfill_first_value   manual rows with no anchor value (#44)
  ▼
pvf.features.samples.add_panel_features   gathered features for the sample's own season (#43)
  │  drop_unmatched_augmented: span rows without a panel row are dropped
  ▼
pvf.data.augment.cap_augmented   all manual rows + seeded sample of span rows = 1,000
  ▼
data/processed/samples.parquet   and   data/synthetic/contract_span_samples.csv
  ▼
pvf.evaluation.splits.player_folds   5-fold, grouped by player, only manual rows scored
```

Run it with (the panel first; it reads the full Kaggle tables and needs a few GB free):

```bash
python scripts/build_panel.py --data-dir "<Drive>/Dataset"
python scripts/build_dataset.py --data-dir "<Drive>/Dataset" --sheet "<Drive>/Contract Data Collection.xlsx"
```

## Features (all known at the anchor)

| Feature | Source | Meaning |
| --- | --- | --- |
| `years_left` | manual | Contract end minus anchor, in years |
| `contract_years` | manual | Full length of the contract |
| `years_into_contract` | manual | Anchor minus contract start |
| `value_now` | Kaggle valuations | Latest value on or before the anchor, at most 365 days old |
| `value_change_12m` | Kaggle valuations | log of value now over value a year earlier |
| `age`, `position` | Kaggle players | Static profile |
| 42 `PANEL_FEATURES` | Panel, same player and season (#43) | Playing time and output last season; club and league strength and European football; rank against positional peers at the club and in the league; value peak and history; injuries and transfers in the last 12 months; sub-position, foot, height, EU status |

The panel features are listed in `pvf.features.samples.PANEL_FEATURES`. In the first build
with them, 38 of the 49 features have no gaps. `transfer_fee_12m` is 76% empty (most players had
no transfer), and the injury columns are 15% empty (players the injury table does not track).

Target: `y_h1 = log(value one year after the anchor / value_now)`. Only one horizon: the
2023/24 rows (194 of 500) cannot have a 2- or 3-season target before the data ends on 2026-06-12.

## Anchor rule for summer signings

The sheet pins each row to 1 July of a season. 204 of the filled rows have a contract that
started after that date: a summer transfer or an extension signed in July or August. On
1 July that contract did not exist yet, so using it there would leak. **The anchor moves to the
contract start** (`anchor = max(1 July, contract_start)`), and the value and target are measured
from that date. A start a full year or more after the 1 July date is treated as an error:
that contract belongs to a later season.

## Augmentation: contract-span expansion

A contract signed in 2019 and running to 2023 was also in force on 1 July 2020, 2021 and 2022.
Each of those dates becomes a sample with the same contract dates and its own anchor, so its
`years_left`, value and target are all real observations, not generated numbers. We drop a
date when:

- it is outside `[contract_start, contract_end)`, or is the season we already collected;
- its one-year target would run past 2026-06-12;
- the player's Transfermarkt club on that date differs from his club during the collected season,
  which means he had moved and the contract no longer applied.

**Known weakness:** we cannot see an extension signed between the collected season and a
span date. There, `years_left` is too short. The club check removes transfers, not extensions.
The report should state this. Any measured effect should come from the manual rows alone,
which is why only manual rows are ever scored.

The augmented rows are written separately to `data/synthetic/contract_span_samples.csv` with their
parent `rank`, as the rubric requires (synthetic kept apart from manual).

## Evaluation

A time split no longer works with 500 rows skewed to 2023/24. `player_folds` gives 5 folds
grouped by player: a player's manual row and his span rows are always on the same side, and
the test side holds manual rows only. The baselines (`NoChange`, `AgePositionCurve`) stay. Pass
a smaller `min_count` to `AgePositionCurve`, since cells are much thinner now.

## Result of the build (2026-09-27, after the sheet fixes)

| Step | Rows |
| --- | --- |
| Sheet rows | 500 |
| Errors after the sheet fixes | 0 |
| Usable contracts | **500** |
| Manual rows with a value at the sampled anchor | 431 |
| Moved to a later season of the same contract (`anchor_shifted`, #44) | +55 |
| Nearest Transfermarkt value backfilled (`value_backfilled`, #44) | +14 |
| Manual rows modelled | **500** |
| Span candidates with a target and a panel row | 639 |
| Final dataset (seed 42) | **1,000** = 500 manual + 500 span |

The first build (before the fixes) had 486 usable contracts and 419 modelled manual rows. The
10 rows with missing dates were filled in the sheet, and 4 rows with another contract's dates
were corrected with sourced dates (Noa Lang, Maddison, Danso, Núñez; Jack to confirm Noa Lang).

69 manual rows had no value on their sampled anchor. Transfermarkt's own value history, checked
for all 69, starts on the same date as Kaggle's, so those values never existed: the sheet
generator picked a season before a young player's first valuation (Palmer 2020, Musiala 2019).
Decision #44 recovers all of them. The flagged rows grow faster than the rest (median `y_h1`
+0.49 normal, +1.10 shifted, +1.61 backfilled), because they catch young players breaking
through. Report scores with and without `value_backfilled` rows, since those use a value from
after the anchor.

Warnings that do not drop rows: 247 `source_url` entries are not URLs (mostly Yunus's half,
written as "Team Website"). The sheet made the URL required, and it is the audit trail for the
manual-data rubric item, so these should be replaced with links.

## What was removed

The panel pipeline was first removed, then restored the same day (#43) as the source of the
gathered features. `scripts/build_panel.py` rebuilds it. What stays removed: the temporal
splits (`time_split`, `rolling_origins`) and their two tests, and the panel's own targets,
which never reach the samples.

## Files touched

| File | Change |
| --- | --- |
| `src/pvf/data/manual.py` | New: read and check the contract sheet |
| `src/pvf/data/augment.py` | New: contract-span augmentation and the 1,000 cap |
| `src/pvf/features/samples.py` | New: features and target per contract row |
| `src/pvf/evaluation/splits.py` | Rewritten: player-grouped folds |
| `scripts/build_dataset.py` | New: end-to-end build |
| `scripts/make_contract_sheet.py` | Committed as the record of how the 500 players were picked |
| `tests/test_{manual,samples,augment,splits}.py` | New, 48 tests |
| `src/pvf/features/{build_panel,club_league,context,health,history,leakage,performance}.py` | Removed, then restored (#43) as the source of the 42 panel features |
| `scripts/build_panel.py` | New: rebuilds `data/processed/panel.parquet` |
| `configs/config.yaml` | Panel, leagues and season split replaced by samples, augment and fold settings |
| `app/app.py`, `app/README.md` | Explanation text and data credit only |
| `README.md`, `docs/*`, `data/*/README.md`, `.gitignore`, `requirements.txt` | Updated to the new scope |

## Still open

- **Jack:** the GUI still offers 2- and 3-season horizons, and the model now has only one.
  Also check whether the bundle schema's league fields should be filled from
  `player_valuations.player_club_domestic_competition_id`.
- Replace the 247 non-URL sources with links (a warning only; the rows are used).
- Jack: confirm the corrected Noa Lang contract (rank 448); the old entry held Pascal Struijk's dates.
- `scripts/build_panel.py` has not yet been run end to end (the C: drive was full); the build
  used the existing `panel.parquet`. Re-run it once to confirm it reproduces the file.
- Question G (public repo): `data/manual/` and `data/synthetic/` CSVs stay out of git and live on
  the Drive, like all other data. If the rubric needs them in the repo, change `.gitignore`.
