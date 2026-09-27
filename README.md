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
| `scripts/` | CLI entry points (`build_dataset.py`, contract sheet, mock data, deploy) |
| `tests/` | Sheet checks, sample leakage, augmentation, folds, metrics, schema and GUI checks |
| `docs/` | Report drafts, decision log, GenAI usage log |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/build_dataset.py --data-dir <Drive>/Dataset --sheet "<Drive>/Contract Data Collection.xlsx"
python scripts/make_mock_predictions.py   # writes a fake bundle so the GUI runs today
python app/app.py                         # opens http://127.0.0.1:7860
pytest
```

## Pipeline contract

The model side ends by writing a **prediction bundle** (`players`, `history`, `forecasts` parquet + `manifest.json`). The app only reads that bundle. Schema lives in `src/pvf/export/schema.py`.
