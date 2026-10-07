---
license: apache-2.0
base_model: amazon/chronos-bolt-small
library_name: chronos-forecasting
tags:
  - time-series-forecasting
  - zero-shot
  - football
  - market-value
datasets:
  - {HF_USER}/football-contracts-500
pipeline_tag: time-series-forecasting
---

# Player value forecaster: Chronos-Bolt, zero-shot (secondary model)

**Category: off-the-shelf.** The secondary model of CMU 24-679 Project 1 (Yunus Polatoglu and
Jack Stevens). We use Amazon's pretrained
[`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small) (48M parameters,
Apache-2.0) **without any training or fine-tuning**. This repo holds no new weights. It holds
how we call the model, its calibration, our wrapper code and the evaluation against our
from-scratch model.

- Primary model (trained from scratch): https://huggingface.co/{HF_USER}/player-value-lgbm-core
- Dataset: https://huggingface.co/datasets/{HF_USER}/football-contracts-500
- App: https://huggingface.co/spaces/{HF_USER}/player-value-forecaster
- Code: https://github.com/jackstev-edu/player-value-forecaster (`src/pvf/models/foundation.py`, `scripts/eval_chronos.py`)
- Notebook that runs this model in Colab: https://colab.research.google.com/github/jackstev-edu/player-value-forecaster/blob/main/notebooks/02_end_to_end.ipynb

## What it does

It reads one thing, the player's own Transfermarkt value history up to the snapshot date, and
forecasts the value 12 months later as quantiles 0.1, 0.5 and 0.9. We report them as
`log(forecast / value today)`, the same target as the primary model, so the two are scored on
identical rows.

**Input:** the player's valuations up to the anchor as a monthly step series. Each month holds
the latest valuation on or before it, as Transfermarkt shows it; oldest first, at most 240
months. No valuation after the anchor is read, and a unit test guards that.
**Output:** `q10`, `q50`, `q90` log ratios. A player with no history falls back to "no change"
with no band (14 of 1,000 samples).

## Why this model

The rubric asks for a second model category. More importantly, it asks a different question
from the primary model: **how far does the value path alone get you, with no age, position or
contract?** Chronos-Bolt is a strong, widely used pretrained forecaster. It runs on a CPU in
about a minute for 1,000 players, and because it is zero-shot it cannot absorb the selection
bias of our training set, which makes it a natural check on the primary model.

## Evaluation

Same five folds and same 500 hand-collected rows as the primary model (`scripts/eval_chronos.py`).
"Conformal" widens the band by a split-conformal shift fitted on each fold's training rows only.

| Model | log MAE | 80% band coverage |
| --- | --- | --- |
| No change | 0.959 | – |
| LightGBM core (primary) | **0.608** | 0.788 |
| **Chronos-Bolt zero-shot** | 1.327 | 0.405 |
| **Chronos-Bolt + conformal band** | 1.327 | 0.695 |
| LightGBM core + Chronos forecast as 2 extra features | 0.609 | 0.802 |

On these players, the future stars, Chronos does worse than "no change". 79% of them rose, often
5× or more from a low base, and Chronos predicts a rise for only 23%. We inspected the worst
rows: a series that climbs from €0.15M to €10M is forecast to fall back towards its own average,
both on raw euros and on log values, so this is the model's behaviour, not an input error.
Feeding its forecast to LightGBM as two extra features changes nothing (+0.001, 95% interval
−0.012 to +0.013).

**Ordinary league players** (2,000 panel rows anchored in 2023 and 2024, none of them training
players) show the other side:

| Model | log MAE | predicts a rise | 80% band coverage |
| --- | --- | --- | --- |
| No change | **0.457** | 0% | – |
| LightGBM core (primary) | 1.110 | 97% | 0.401 |
| **Chronos-Bolt zero-shot** | **0.560** | **28%** | **0.709** |
| Actual outcome | – | 28% | – |

Here Chronos gets the direction right (it predicts a rise for 28%, and 28% rose). Its error is
half the primary model's, and its band is far closer to 80%. Neither beats "no change" on this
population: one-season changes for established players are mostly noise around zero.

**Reading the two together:** the from-scratch model learned what a breakout looks like, and it
over-applies that to everyone. The off-the-shelf model knows how time series usually behave, and
it under-predicts breakouts. Combining them by population is future work. The app stays on the
primary model and states its bias on the page.

## Intended use and limits

**Intended for** coursework: a zero-shot baseline and a bias check on the primary model.
**Not for** decisions about real players.

- History only: it cannot see age, injuries, transfers or contracts.
- Short histories (under a year) are close to guesswork. The median context here is 38 months.
- Calibrated only on our 1,000 samples (the conformal shift is in `config.json`). For future stars the band is still too narrow.

## Use it

```python
# pip install chronos-forecasting torch pandas
import numpy as np, pandas as pd
from chronos_value import ChronosForecaster, monthly_context   # chronos_value.py in this repo

history = pd.DataFrame({"player_id": 1,
                        "date": pd.to_datetime(["2023-01-15", "2023-09-01", "2024-02-01"]),
                        "market_value_in_eur": [10e6, 18e6, 25e6]})
ctx = monthly_context(history, 1, pd.Timestamp("2024-07-01"))
q = ChronosForecaster("amazon/chronos-bolt-small").predict([ctx], np.array([25e6]))
print((25e6 * np.exp(q[["q10", "q50", "q90"]])).round(-5))
```

Files: `config.json` (how the model is called, conformal shift), `metrics.json` (every number
above), `chronos_value.py` (our wrapper, identical to `src/pvf/models/foundation.py`).

## Ethics and credit

Model by Amazon Science (Ansari et al., *Chronos: Learning the Language of Time Series*, 2024),
Apache-2.0. Player values are published Transfermarkt estimates about real people, so keep outputs
inside coursework. Built with help from Claude (Anthropic). The fold-by-fold comparison was set up
before any result was seen. The league-player check was added after the first result, to test
the primary model's known bias. Both steps are logged in the repo's `docs/decisions.md`.
