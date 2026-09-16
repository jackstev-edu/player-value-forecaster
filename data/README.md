# Data

| Folder | Contents | In git? |
| --- | --- | --- |
| `raw/player-scores/` | Kaggle `davidcariboo/player-scores` parquet (copy from team Drive `Dataset/player-scores`) | No |
| `raw/football-datasets/` | Kaggle `xfkzujqjvx97n/football-datasets` (salimt) parquet | No |
| `interim/` | Cleaned tables with parsed dates | No |
| `processed/` | Player-season panel used for training | No |
| `manual/` | Our manually collected samples (rubric requirement, see its README) | Yes |
| `synthetic/` | Any augmented or generated data, never mixed with `manual/` | Yes |

Column meanings, completeness and join rates: `Dataset/Dataset_Overview.xlsx` in the team Drive.

Known traps: dates are text, `height = 0` and `value = 0` mean unknown, `team_competitions_seasons` is ~73% duplicate rows, and snapshot columns leak the future (see `src/pvf/features/leakage.py`).

Both sources are scraped from Transfermarkt and identify real people. Use for coursework only, do not redistribute raw files, and credit the Kaggle authors and Transfermarkt in the Space.
