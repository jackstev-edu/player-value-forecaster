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

CMU 24-679 Project 1 (Yunus Polatoglu & Jack Stevens). Search players by name (case and accents ignored) and filter by league, country, nationality, age and position, then select one to see their Transfermarkt value history and a one-season forecast with a 10 to 90% range.

Sort by current value, biggest predicted rise or fall, or most uncertain forecast. The app reads the horizons from the bundle it is given: the current model forecasts one season, so the Look ahead control is hidden until a bundle carries more.

The page opens on the most valuable player, and four examples (Established star, Rising young player, Veteran in decline, Hardest to predict) are picked from the data by fixed rules each time the app starts, so they stay valid when the bundle changes.

Forecasts come from a four-feature LightGBM (age, position, current value and its change over the past year) trained on 500 hand-collected contracts. Those players were collected because they became valuable, so the forecasts lean upward; the app says so on the page.

Data: contract dates for 500 players, collected by hand by the team, and Transfermarkt values via the Kaggle dataset [player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) by davidcariboo. Coursework only.
