---
license: cc-by-4.0
language:
  - en
pretty_name: Football contracts 500 (hand-collected) for market-value forecasting
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-regression
  - time-series-forecasting
tags:
  - football
  - soccer
  - transfermarkt
  - contracts
  - market-value
configs:
  - config_name: samples
    default: true
    data_files:
      - split: manual
        path: samples/manual.parquet
      - split: augmented
        path: samples/augmented.parquet
  - config_name: contracts
    data_files:
      - split: manual
        path: contracts/manual.csv
      - split: augmented
        path: contracts/augmented.csv
  - config_name: value_history
    data_files:
      - split: train
        path: value_history.parquet
---

# Football contracts 500

Contract start and end dates for **500 professional football players, collected by hand** by
Yunus Polatoglu and Jack Stevens for CMU 24-679 Project 1. The data is joined to Transfermarkt
market values to train and test a model that forecasts each player's value one season ahead.
Another **500 augmented rows** sit in their own split and never mix with the manual rows.

- Primary model (trained from scratch): https://huggingface.co/{HF_USER}/player-value-lgbm-core
- Secondary model (off-the-shelf): https://huggingface.co/{HF_USER}/player-value-chronos-bolt
- App: https://huggingface.co/spaces/jackstev/player-value-forecaster
- Code: https://github.com/jackstev-edu/player-value-forecaster
- EDA notebook: https://colab.research.google.com/github/jackstev-edu/player-value-forecaster/blob/main/notebooks/01_dataset_eda.ipynb

```python
from datasets import load_dataset
samples = load_dataset("{HF_USER}/football-contracts-500", "samples")      # splits: manual, augmented
contracts = load_dataset("{HF_USER}/football-contracts-500", "contracts")
```

## What is in it

| Config | Split | Rows | What a row is |
| --- | --- | --- | --- |
| `contracts` | `manual` | 500 | One hand-collected contract: player, season, club, start, end, source |
| `contracts` | `augmented` | 500 | The same contract reused at another 1 July inside it (see Augmentation) |
| `samples` | `manual` | 500 | Model-ready row: 49 features known at the anchor plus the target `y_h1` |
| `samples` | `augmented` | 500 | The same for augmented rows (training only, never scored) |
| `value_history` | `train` | 10,420 | Every Transfermarkt valuation of the 500 players (input for the time-series model) |

**Target:** `y_h1 = log(value one season after the anchor / value at the anchor)`. 0 means no
change, +0.69 means doubled.

### Collection process (the manual split)

1. `scripts/make_contract_sheet.py` listed the **500 most valuable players** by their current
   Transfermarkt value and pinned each one to **one random past season** (1 July anchor) with
   their club that season.
2. The two of us split the list by rank (Yunus odd, Jack even; 250 each). For each player we
   looked up the contract in force with that club on that 1 July, using club announcements,
   league and news sites and Wikipedia. We recorded `contract_start`, `contract_end`,
   `source_url` and `notes` (loans, option years, oddities).
3. `pvf.data.manual.check_contracts` validated every row. **Errors**, which would drop a row:
   missing dates, an end that is not after the start, a contract that expired before the anchor,
   a contract starting a full season late, duplicate players. **Warnings:** a source that is not
   a URL. The final sheet has **0 errors**. Four contracts were found to be wrong and were
   re-sourced and corrected. **247 warnings** remain: those sources are named ("Barcelona
   official website") rather than linked.
4. The features and target come from the dated values, so every row was labelled by joining it
   to Transfermarkt valuations as of its anchor. 55 rows moved to a later season because the
   player had no value that summer (`anchor_shifted`). 14 rows took the nearest value from just
   after the anchor (`value_backfilled`). That is a small leak, so it is flagged, and results are
   reported with and without these rows.

### Augmentation (the augmented split, kept separate)

**Contract-span expansion.** A contract also covers the other 1 July dates between its start and
end, so each such date becomes a new row with the same contract and its own anchor. Dates are
dropped when the player had changed club or the one-year target runs past the data. A seeded
sample (seed 42) keeps 500, so the total is 1,000. Each augmented row keeps its parent `rank`.
**Augmented rows are used only for training. Every reported score uses manual rows only**,
under 5-fold cross-validation grouped by player. The known blind spot: contract extensions we
did not collect are invisible, so `years_left` can be overstated on augmented rows.

### Columns (`samples`)

| Column | Meaning |
| --- | --- |
| `rank`, `player_id`, `player_name` | Collection rank (1–500), Transfermarkt id, name |
| `origin` | `manual` or `span` (augmented) |
| `season`, `anchor_date` | Season year; the snapshot date (1 July, or a later signing date) |
| `age`, `position`, `value_now`, `value_change_12m` | Core features (the shipped model uses only these 4) |
| `years_left`, `contract_years`, `years_into_contract` | From the hand-collected dates |
| `y_h1` | Target, log change over one season |
| `anchor_shifted`, `value_backfilled` | Collection flags (see above) |
| 42 more | Last season's minutes/goals, club and league strength, European football, rank vs positional peers, value history, injuries, transfers, profile. Joined from a player-season panel measured **on or before** the anchor. Blank when the player was outside the 9 leagues covered (15.6% of rows) |

## Exploratory analysis

![Samples per season](figures/rows_per_season.png)

Manual rows cluster in recent seasons: today's most valuable players were mostly in their
current contract in 2022–2024. Augmented rows reach 2024/25 because those contracts are still
running.

![Target distribution](figures/target_distribution.png)

**79% of manual rows rose in value over the season (median +0.59 log, about ×1.8).** This is
the dataset's defining property. The players were chosen for being valuable *today*, so their
past seasons are mostly rises: a survivor sample. Augmented rows sit closer to zero because
they include the later, flatter years of the same careers.

![Change by age](figures/change_by_age.png)

Age is the strongest single signal. Teenagers in the sample rose by a median of ×4 within a
season, and by 27 the rise is gone. Median age is 20.7, the range 16–29.

![Contract years left](figures/years_left.png)

Most manual rows have 1 to 5 years left on the contract, peaking at 4–5 years: many were
fresh signings. Augmented rows have less time left by construction, because they are later
dates inside the same contracts.

Positions (manual): Attack 170, Midfield 154, Defender 153, Goalkeeper 23. Median value at the
anchor: €8M (10th–90th percentile €0.4M–€40M). Full numbers: `eda_summary.json`.

## Intended use

Training and evaluating one-season market-value forecasts in coursework, and studying
**selection bias**: a model trained here learns how future stars move, and our evaluation
shows it over-predicts rises for ordinary players (97% predicted vs 28% observed).
**Not suitable** for valuing players for transfers, contracts or betting.

## License and sources

- **Our contribution** (the contract dates, sources, notes and checks): **CC BY 4.0**, credit
  "Yunus Polatoglu and Jack Stevens, CMU 24-679 (2026)".
- **Market values, player profiles and panel features** come from two Kaggle datasets, both
  **CC0: Public Domain**:
  [player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (davidcariboo;
  Transfermarkt scrape) and
  [football-datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets) (salimt;
  injuries and transfer history). Valuations run through 5 June 2026.
- Contract facts come from public announcements. Each row's `source_url` says where.

## Ethical notes

- **Real, named people.** Every row is an identifiable professional player. Contract dates and
  market values are already public and the players are public figures, but forecasts about a
  named person's worth can mislead or upset. Do not publish forecasts about individuals as fact.
- **Market values are opinions.** Transfermarkt values are crowd and editor estimates, not
  prices. They carry the platform's biases (league, nationality, visibility).
- **Survivor bias is built in** (see EDA). Any model trained here must be checked on a general
  population before its forecasts are read as typical.
- **Coverage.** Men's football only, mostly in European top leagues.
- **AI assistance.** Claude (Anthropic) wrote pipeline code, including the sheet checks and the
  augmentation, under our direction, and looked up sources for the 4 corrected contracts. Every
  date in the manual split was entered or verified by one of us. The log is in the repo's
  `docs/genai_log.md`.

## Citation

```
Polatoglu, Y. and Stevens, J. (2026). Football contracts 500: hand-collected contract dates for
market-value forecasting. CMU 24-679 Designing and Prototyping AI Systems, Project 1.
https://huggingface.co/datasets/{HF_USER}/football-contracts-500
```
