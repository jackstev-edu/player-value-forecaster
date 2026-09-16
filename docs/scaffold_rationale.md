# Repository scaffold: rationale

CMU 24-679 Project 1, Yunus Polatoglu & Jack Stevens. Draft 2026-09-16.

## 1. Context pulled from the team Drive

The system forecasts how a football player's Transfermarkt market value will move over the next 1, 2 and 3 seasons, and lets a user browse players by league, league country, nationality, age and position (Project Proposal). The Analysis Guide sets the method: one row per player per season at a fixed anchor date, a log-ratio target per horizon, LightGBM quantile models, two simple baselines, time-based splits, and an ablation that asks whether context (league, club, European football, positional rank) beats age plus current value. The Dataset Overview confirms the two Kaggle Transfermarkt sources share player IDs (30,361 players in both) and lists the snapshot columns that would leak future information. The Team Charter favours clear milestones and independent work toward defined goals, which the structure below is meant to support.

## 2. Structure and why

**Pipeline and GUI are separate, joined by a file contract.** The model side writes a prediction bundle (players, value history, forecasts with 10/50/90% bands, and a manifest). The app reads only that bundle. This follows the Analysis Guide's hand-off section and has three benefits: the Space starts quickly and needs no model libraries, either teammate can work on one side without breaking the other, and the GUI could be built today against a mock bundle. `src/pvf/export/schema.py` defines the contract and a test checks the committed bundle against it.

**Installable `src/` package, not loose notebooks.** Reusable logic lives in `src/pvf` (load, features, models, evaluation, export). Notebooks are for exploration only. This is the standard layout for Python projects because imports behave the same in tests, scripts and notebooks, and code review happens on plain `.py` files instead of notebook JSON.

**One config file.** Anchor date, horizons, quantiles, split seasons and league list are in `configs/config.yaml`. The Analysis Guide leaves several of these open, so they can change without editing code, and the report can cite exact settings.

**Methodological safeguards are code, not just notes.** The leakage list is enforced by `assert_no_leakage`. The panel builder uses a backward as-of join, so a row can only see values dated on or before its anchor. `time_split` removes training rows whose target resolves after the test anchor, and `rolling_origins` drops test seasons whose horizon runs past the last valuation (June 2026). Each of these has a unit test, and GitHub Actions runs the tests on every push. The baselines are implemented now so every model is compared against them from its first run.

**Data folders mirror the rubric.** Raw scraped data stays out of git: it is large, and it identifies real people. `data/manual/` and `data/synthetic/` are separate so the manually collected count can be checked, as the spec requires. The manual dataset is a placeholder with a required-fields template, and candidate ideas are in `docs/decisions.md`.

**Two model families have slots.** `models/gbm.py` (trained from scratch) and `models/foundation.py` (a pretrained Chronos time-series model, off-the-shelf and possibly fine-tuned) cover the rubric's "at least two of three." Chronos sees only a player's own value history, so it also works as a history-only comparison for the context question. Both expose the same prediction shape so the evaluation code treats them identically. Unsettled logic is marked with numbered `>>> <<<` placeholders.

## 3. GUI choices

**Gradio.** It is the default SDK on Hugging Face Spaces and deploys by pushing `app/`. I could not open the course GUIs module, so this is an assumption to confirm. If the course used Streamlit, only `app/` changes, because of the bundle contract.

**Design.** The layout follows the user's steps: filters on the left, results table in the middle, then the selected player's chart and forecast summary. The chart shows history as a step line, because Transfermarkt values change in discrete updates. The forecast is a dashed median with a shaded 10 to 90% band, so users see uncertainty instead of a single confident line. A banner appears automatically whenever the manifest marks the bundle as mock. The page credits the data sources and states the limits of use, following the course's GUI and data practices.

**Mock data uses invented players on purpose.** Assigning made-up forecasts to real, named players would misrepresent real people, even temporarily.

## 4. Rubric mapping

| Requirement | Where it lives | Status |
| --- | --- | --- |
| Functional and useful, with performance measurement | `evaluation/`, baselines, ablation plan | Scaffolded |
| 500+ manual samples, synthetic kept separate | `data/manual/`, `data/synthetic/` | **Undecided** |
| Two of three model types | `models/gbm.py`, `models/foundation.py` | Slots ready |
| Public GUI on HF Spaces | `app/`, `scripts/deploy_space.py` | Runs locally on mock data |
| GenAI documentation | `docs/genai_log.md` | Started |

## 5. Next steps

1. Merge with Yunus's existing scripts (`build_dataset_overview.py`) under `scripts/`.
2. Settle decisions 1 to 4 and 6 in `docs/decisions.md` at the next in-person meeting.
3. Fill `build_panel.py` markers 1 to 5, then run the Analysis Guide's exploration notebooks.
4. Fit the baselines, then LightGBM, then Chronos. Swap the mock bundle for a real one.
5. Deploy the Space early with the mock bundle so hosting problems surface before the deadline.
