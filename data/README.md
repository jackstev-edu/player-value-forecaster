# Data

| Folder | Contents | In git? |
| --- | --- | --- |
| `raw/player-scores/` | Kaggle `davidcariboo/player-scores` parquet; only `player_valuations` and `players` are read | No |
| `interim/` | Cleaned tables with parsed dates | No |
| `processed/` | `samples.parquet`, the 1,000 modelling samples | No |
| `manual/` | `contracts.csv`, the 500 hand-collected contracts, and `contract_issues.csv` | README only; data on Drive |
| `synthetic/` | `contract_span_samples.csv`, the augmented rows, never mixed with `manual/` | README only; data on Drive |

Column meanings, completeness and join rates: `Dataset/Dataset_Overview.xlsx` in the team Drive.

Known traps: dates are text, `height = 0` and `value = 0` mean unknown, `team_competitions_seasons` is ~73% duplicate rows, and snapshot columns leak the future (see `src/pvf/features/leakage.py`).

Both sources are scraped from Transfermarkt and identify real people. Use for coursework only, do not redistribute raw files, and credit the Kaggle authors and Transfermarkt in the Space.
