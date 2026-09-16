# Player Market Value Forecaster

CMU 24-679 Designing & Prototyping AI Systems, Project 1 (Yunus Polatoglu & Jack Stevens).

Forecasts how a football player's Transfermarkt market value will move over the next 1, 2 and 3 seasons, with a 10 to 90% range, and serves the results in a Gradio app on Hugging Face Spaces.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/config.yaml` | Every tunable decision in one place (anchor date, horizons, splits, leagues) |
| `src/pvf/` | Pipeline package: load, build player-season panel, train, evaluate, export |
| `data/raw/` | Kaggle parquet, never committed (see `data/README.md`) |
| `data/manual/` | Manually collected samples, kept separate per the rubric |
| `data/synthetic/` | Augmented or generated data, kept separate per the rubric |
| `app/` | Self-contained Hugging Face Space. Reads only `app/predictions/` |
| `scripts/` | CLI entry points (mock data, deploy) |
| `tests/` | Leakage, split, metric and schema checks |
| `docs/` | Report drafts, decision log, GenAI usage log |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/make_mock_predictions.py   # writes a fake bundle so the GUI runs today
python app/app.py                         # opens http://127.0.0.1:7860
pytest
```

## Pipeline contract

The model side ends by writing a **prediction bundle** (`players`, `history`, `forecasts` parquet + `manifest.json`). The app only reads that bundle. Schema lives in `src/pvf/export/schema.py`.
