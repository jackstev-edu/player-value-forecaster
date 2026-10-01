# Player Market Value Forecaster

CMU 24-679 Designing & Prototyping AI Systems, Project 1 (Yunus Polatoglu & Jack Stevens).

Forecasts how a football player's Transfermarkt market value will move over the next season, with a 10 to 90% range, and serves the results in a Gradio app on Hugging Face Spaces.

**Dataset (since 2026-09-27):** contract start and end dates for 500 players, collected by hand by the team, augmented to 1,000 samples by reusing each contract for its other seasons. Kaggle is used only for the players' values, age and position. Why and how: `docs/scope_change.md`.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/config.yaml` | Every tunable decision in one place (anchor, horizon, augmentation total, folds) |
| `src/pvf/` | Pipeline package: check the contract sheet, build samples, augment, train, evaluate, export |
| `data/raw/` | Kaggle `player-scores` parquet (valuations and players only), never committed |
| `data/manual/` | Manually collected samples, kept separate per the rubric |
| `data/synthetic/` | Augmented or generated data, kept separate per the rubric |
| `app/` | Self-contained Hugging Face Space. Reads only `app/predictions/` |
| `scripts/` | CLI entry points, listed in run order below |
| `tests/` | Sheet checks, sample leakage, augmentation, folds, metrics, export, schema and GUI checks |
| `docs/` | Decision log, roadmap, GenAI usage log, report figures (`docs/figures/`) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                                    # the app tests use a fixed mock bundle in tests/fixtures/
python app/app.py                         # the committed real bundle, http://127.0.0.1:7860
```

## Rebuilding everything

Each step reads the one before. `<Dataset>` is the team Drive's `Dataset` folder (or set `FV_DATA_DIR`).

| Step | Command | Writes |
| --- | --- | --- |
| 1. Player-season panel | `python scripts/build_panel.py --data-dir <Dataset>` | `data/processed/panel.parquet` |
| 2. 1,000 samples | `python scripts/build_dataset.py --data-dir <Dataset> [--sheet "<Drive>/Contract Data Collection.xlsx"]` | `data/processed/samples.parquet`, `data/synthetic/` |
| 3. Cross-validation | `python scripts/train_gbm.py [--ablation]` | out-of-fold predictions and metrics in `data/processed/` |
| 4. App bundle | `python scripts/export_bundle.py --data-dir <Dataset>` | `app/predictions/` |
| 5. Player backtest | `python scripts/backtest_players.py --data-dir <Dataset>` | `docs/figures/backtest_players.{png,csv}` |
| 6. Deploy | `python scripts/deploy_space.py <user>/<space>` | the Hugging Face Space |

`scripts/make_contract_sheet.py` made the blank contract sheet that task 19 filled in; it is kept for the record.

## Pipeline contract

The model side ends by writing a **prediction bundle** (`players`, `history`, `forecasts` parquet + `manifest.json`). The app only reads that bundle. Schema lives in `src/pvf/export/schema.py`.
