---
license: mit
library_name: lightgbm
tags:
  - tabular-regression
  - quantile-regression
  - football
  - market-value
  - conformal-prediction
datasets:
  - {HF_USER}/football-contracts-500
metrics:
  - mae
pipeline_tag: tabular-regression
model-index:
  - name: lgbm-core-v1
    results:
      - task:
          type: tabular-regression
          name: One-season log change in player market value
        dataset:
          type: {HF_USER}/football-contracts-500
          name: Football contracts 500 (manual split, 5-fold player-grouped CV)
        metrics:
          - type: mae
            name: log MAE
            value: 0.608
          - type: coverage
            name: 80% band coverage
            value: 0.788
---

# Player value forecaster: LightGBM core model (`lgbm-core-v1`)

**Category: trained from scratch.** The primary model of CMU 24-679 Project 1 (Yunus Polatoglu
and Jack Stevens), and the model behind the live app.

- App: https://huggingface.co/spaces/{HF_USER}/player-value-forecaster
- Dataset: https://huggingface.co/datasets/{HF_USER}/football-contracts-500
- Second model (off-the-shelf): https://huggingface.co/{HF_USER}/player-value-chronos-bolt
- Code: https://github.com/jackstev-edu/player-value-forecaster
- End-to-end notebook: https://colab.research.google.com/github/jackstev-edu/player-value-forecaster/blob/main/notebooks/02_end_to_end.ipynb

## What it does

Given a football player on a 1 July snapshot, it forecasts how their Transfermarkt market value
will change over the next season: a median and a 10th to 90th percentile range of
`log(value in one year / value today)`. The app turns those into euros.

**Inputs (4 features):** `age` (years, decimal), `position` (Attack, Midfield, Defender,
Goalkeeper), `value_now` (EUR), `value_change_12m` (log change in value over the past year).

**Outputs:** `q10`, `q50`, `q90` as log ratios. Forecast in euros = `value_now * exp(q)`.

## Why this model

- **Small tabular data.** 1,000 rows of mixed numeric and categorical columns, where gradient
  boosting is the standard strong choice. Shallow trees (15 leaves), slow learning (0.03) and
  early stopping on held-out players keep it from memorising 800 rows.
- **Ranges, not just points.** One LightGBM per quantile (0.1, 0.5, 0.9), then split-conformal
  calibration (CQR) on held-out players, so the 10 to 90% band really holds about 80%.
- **Four features, not 49.** The full model had 49 features (contract, minutes, club and league
  strength, peer rank, injuries, transfers). An ablation with paired bootstrap intervals found
  no feature group beyond age, position and value that measurably helps (decision #47 in the
  repo), so the simplest model shipped. It also needs no contract or panel row, so it can score
  every league player.

## Training data

The [football-contracts-500](https://huggingface.co/datasets/{HF_USER}/football-contracts-500)
dataset: 500 contracts collected by hand (the 500 most valuable players, each at one past season)
plus 500 augmented rows that reuse each contract for its other seasons. Values come from
Transfermarkt via the CC0 Kaggle dataset
[player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores).

Fitting: all 1,000 samples, with 15% of players held out for early stopping and band
calibration. Seed 42. Parameters and the band shift are in `metrics.json`.

## Evaluation

**Five-fold cross-validation grouped by player**, so no player is in both training and test.
Only the 500 hand-collected rows are scored; augmented rows are used for training only.

| Model | log MAE | 80% band coverage |
| --- | --- | --- |
| No change (predict 0) | 0.959 | – |
| Age × position median curve | 0.744 | – |
| Ridge linear baseline | 0.602 | – |
| **LightGBM core (this model)** | **0.608** | **0.788** |
| LightGBM core + Chronos forecast as features | 0.609 | 0.802 |
| Chronos-Bolt zero-shot | 1.327 | 0.405 |

A log MAE of 0.61 is a typical miss of about ×1.8. The linear baseline ties it (difference
−0.006, 95% interval −0.028 to +0.016); LightGBM is kept for its calibrated band.

**Walk-forward backtest on five named players** (Mbappé, Gavi, Anton, Schrijvers, Salah, picked
by a stated rule from 2024 data, not by outcome): log MAE 0.50, 6 of 10 actuals inside the range.

**Ordinary league players** (2,000 panel rows anchored in 2023 and 2024, none of them training
players), the population the app serves:

| Model | log MAE | predicts a rise | 80% band coverage |
| --- | --- | --- | --- |
| No change | **0.457** | 0% | – |
| **LightGBM core (this model)** | 1.110 | 97% | 0.401 |
| Chronos-Bolt zero-shot | 0.560 | 28% | 0.709 |
| Actual outcome | – | 28% | – |

## Intended use and limits

**Intended for** coursework and illustration: how a forecast with an honest range can be built
from a small hand-collected dataset. **Not for** transfer, contract, betting or any decision
about a real person.

- **Strong upward bias outside its training population.** The 500 players were chosen because
  they are among the most valuable *today*, so the cheap ones in training are future stars
  (selection on the outcome). On ordinary league players it predicts a rise for 97% when 28%
  rose, and its band covers 40% instead of 80%. It suits young players on a breakout path; it
  is not a general league forecast. The app states this on the page.
- **Training covered ages 16 to 31** and values from about EUR 0.75M (10th percentile) upward.
  Older or cheaper players are extrapolations.
- **One season ahead only.** Two and three seasons were measured and held back.
- **Blind to news.** Injuries, transfers and contract changes after the snapshot are invisible.

## Use it

```python
# pip install lightgbm pandas huggingface_hub git+https://github.com/jackstev-edu/player-value-forecaster
import numpy as np, pandas as pd
from huggingface_hub import snapshot_download
from pvf.models.gbm import QuantileGBM

model = QuantileGBM.load(snapshot_download("{HF_USER}/player-value-lgbm-core"))
X = pd.DataFrame({"age": [21.5], "position": ["Attack"], "value_now": [30e6],
                  "value_change_12m": [0.4]})
q = model.predict(X)[["q10", "q50", "q90"]]   # log ratios
print((30e6 * np.exp(q)).round(-5))             # euros one season ahead
```

Files: `h1_q10.txt`, `h1_q50.txt`, `h1_q90.txt` (LightGBM boosters), `meta.json` (features,
category levels, conformal band shift), `metrics.json` (every number on this card).

## Ethics

Players are public figures and the values are published Transfermarkt estimates, but they are
real people, and a forecast of a named person's worth can be wrong in ways that matter to them.
Keep outputs inside coursework. Built with help from Claude (Anthropic); every modelling decision
and its reason is logged in the repo's `docs/decisions.md` and `docs/genai_log.md`.
