# Player Market Value Forecaster

CMU 24-679 Designing & Prototyping AI Systems, Project 1 (Yunus Polatoglu & Jack Stevens).

Forecasts how a football player's Transfermarkt market value will move over the next season, with a 10 to 90% range, and serves the results in a Gradio app on Hugging Face Spaces.

**Dataset (since 2026-09-27):** contract start and end dates for 500 players, collected by hand by the team, augmented to 1,000 samples by reusing each contract for its other seasons. Kaggle is used only for the players' values, age and position. Why and how: `docs/scope_change.md`.

## Prototype package

| Part | Link | What it is |
| --- | --- | --- |
| **App** | [Space](https://huggingface.co/spaces/jackstev/player-value-forecaster) | Search any of 4,727 league players and see next season's value with a likely range; featured examples on the home screen |
| **Primary model** (trained from scratch) | [player-value-lgbm-core](https://huggingface.co/ypolatog/player-value-lgbm-core) | LightGBM quantile model on 4 features, conformally calibrated band. CV log MAE 0.608, 80% band coverage 0.79 |
| **Secondary model** (off-the-shelf) | [player-value-chronos-bolt](https://huggingface.co/ypolatog/player-value-chronos-bolt) | Amazon Chronos-Bolt, zero-shot on each player's value history. The bias check on the primary model |
| **Dataset** | [football-contracts-500](https://huggingface.co/datasets/ypolatog/football-contracts-500) | 500 hand-collected contracts plus 500 augmented rows in a separate split, with value histories, card and EDA |
| **Notebook 1** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jackstev-edu/player-value-forecaster/blob/main/notebooks/01_dataset_eda.ipynb) | Dataset exploration, read straight from the Hub |
| **Notebook 2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jackstev-edu/player-value-forecaster/blob/main/notebooks/02_end_to_end.ipynb) | End to end: data, cross-validated training, both models compared, the published model forecasting a player you describe |

**What the two models found.** On the 500 future stars, LightGBM beats every baseline (log MAE
0.61 against 0.96 for "no change"), and Chronos does worse than "no change" (1.33). On 2,000
ordinary league players, the population the app serves, it flips. LightGBM predicts a rise for
97% when 28% rose, and Chronos predicts 28%. Each model leans towards the population it learned
from. The app keeps LightGBM and says so on the page. Details are in the model cards and
`docs/decisions.md` #52.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/config.yaml` | Every tunable decision in one place (anchor, horizon, augmentation total, folds) |
| `src/pvf/` | Pipeline package: check the contract sheet, build samples, augment, train, evaluate, export |
| `src/pvf/models/` | `gbm.py` (primary, trained from scratch), `foundation.py` (Chronos wrapper, off-the-shelf), `baselines.py` |
| `data/raw/` | Kaggle `player-scores` parquet (valuations and players only), never committed |
| `data/manual/` | Manually collected samples, kept separate per the rubric |
| `data/synthetic/` | Augmented or generated data, kept separate per the rubric |
| `app/` | Self-contained Hugging Face Space. Reads only `app/predictions/` |
| `hf/` | Model and dataset cards, filled in and uploaded by `scripts/publish_hf.py` |
| `notebooks/` | The two Colab notebooks above |
| `scripts/` | CLI entry points, listed in run order below |
| `tests/` | Sheet checks, sample leakage, augmentation, folds, metrics, export, schema, GUI and model checks |
| `docs/` | Decision log, roadmap, GenAI usage log, report figures (`docs/figures/`) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                                    # the app tests use a fixed mock bundle in tests/fixtures/
python app/app.py                         # the committed real bundle, http://127.0.0.1:7860
```

No local setup is needed to see the results: the Space runs in the browser, and both notebooks run on a free Colab CPU runtime, reading the data and model from the Hub.

## Rebuilding everything

Each step reads the one before. `<Dataset>` is the team Drive's `Dataset` folder (or set `FV_DATA_DIR`).

| Step | Command | Writes |
| --- | --- | --- |
| 1. Player-season panel | `python scripts/build_panel.py --data-dir <Dataset>` | `data/processed/panel.parquet` |
| 2. 1,000 samples | `python scripts/build_dataset.py --data-dir <Dataset> [--sheet "<Drive>/Contract Data Collection.xlsx"]` | `data/processed/samples.parquet`, `data/synthetic/` |
| 3. Cross-validation | `python scripts/train_gbm.py [--ablation]` | out-of-fold predictions and metrics in `data/processed/` |
| 4. Off-the-shelf model | `python scripts/eval_chronos.py --data-dir <Dataset>` | Chronos forecasts, CV comparison and the league-player check in `data/processed/` |
| 5. App bundle | `python scripts/export_bundle.py --data-dir <Dataset>` | `app/predictions/` |
| 6. Player backtest | `python scripts/backtest_players.py --data-dir <Dataset>` | `docs/figures/backtest_players.{png,csv}` |
| 7. Hub dataset | `python scripts/build_hf_dataset.py --data-dir <Dataset>` | `data/processed/hf_dataset/` (files, EDA figures, card) |
| 8. Hub models | `python scripts/export_models.py` | `data/processed/hf_models/{lgbm-core,chronos}/` |
| 9. Publish | `hf auth login`, then `python scripts/publish_hf.py <user> [--dry-run]` | dataset, both model repos and the Space |

`scripts/make_contract_sheet.py` made the blank contract sheet that task 19 filled in; it is kept for the record. `scripts/deploy_space.py` uploads the app alone, for example to a preview Space.

## Pipeline contract

The model side ends by writing a **prediction bundle** (`players`, `history`, `forecasts` parquet + `manifest.json`). The app only reads that bundle. Schema lives in `src/pvf/export/schema.py`.

## Use of AI tools

We used Claude (Anthropic), mostly through Claude Code, for code, data checks, documentation and these cards. The team chose every modelling and scope decision, and the hand-collected contract dates were entered or verified by one of us. `docs/genai_log.md` lists each use, what we kept or changed, and what we learned. `docs/decisions.md` records the reasoning behind every decision, including mistakes caught along the way (for example the target leak in #45).
