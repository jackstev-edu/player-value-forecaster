---
title: Player Value Forecaster
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Forecast football player market values one season ahead
---

# Player Value Forecaster

Forecasts where a football player's Transfermarkt market value is likely to go over the
next season, with a range rather than a single number.

CMU 24-679 Project 1, by Yunus Polatoglu and Jack Stevens. Coursework only, not for
transfer or betting decisions.

## How to use it

1. **Find a player.** Type part of a name, or open "More filters" for league, country,
   nationality, position and age. Accents are optional: typing `mbappe` finds Mbappé.
   Or click one of the four examples under the filters.
2. **Select a row** in the table to open that player.
3. **Read the chart.** The solid line is the player's value history, the dashed line is
   the forecast, and the shaded band is the likely range.
4. **Sort** by current value, biggest predicted rise or fall, or most uncertain forecast.

## What the numbers mean

- **Median** is the model's middle estimate: it judges the real value equally likely to
  land above or below it.
- **Likely range** runs from the 10th to the 90th percentile. The model aims for the real
  value to land inside it 8 times out of 10, so roughly 1 player in 5 should fall outside.
- **⚠** marks a low-confidence forecast, either because the range is among the widest 20%
  or because the player has fewer than 3 past valuations.

## Limitations worth knowing

- **The forecasts lean upward.** The model was trained on 500 players we collected by hand
  because they were among the most valuable *today*, so it has seen the careers that worked
  and very few that stalled. Across the app 96% of players are predicted to rise, against
  32% for a comparable set of league players over 2013 to 2025. Read the direction with
  that in mind; the app repeats this warning on the page.
- **Forecasts cover one season ahead.**
- **It cannot see the news.** Injuries, transfers and contract changes after the 1 July
  snapshot are invisible to it.
- **Edges are extrapolation.** Training covered ages 16 to 31 and values from about EUR
  0.75M upward. Players older or cheaper than that are outside what it learned.

## Model and data

Forecasts come from `lgbm-core-v1`, a LightGBM quantile model using four features: age,
position, current value, and how that value moved over the past year. It was tested by
five-fold cross-validation grouped by player, so no player appears in both training and
test data. The exact model version and build date are shown in the footer of the app.

Data: contract dates for 500 players, collected by hand by the team, plus Transfermarkt
values and player profiles via the Kaggle datasets
[player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) by davidcariboo
and [football-datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets) by
salimt. Player names and values identify real people and are used for coursework only.

Design rationale and the class practices behind each feature:
`docs/gui_design_rationale.md` in the project repository.
