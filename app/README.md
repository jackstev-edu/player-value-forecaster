---
title: Player Value Forecaster
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
short_description: Forecast football player market values 1 to 3 seasons out
---

# Player Value Forecaster

CMU 24-679 Project 1 (Yunus Polatoglu & Jack Stevens). Search players by name (case and accents ignored) and filter by league, country, nationality, age and position, then select one to see their Transfermarkt value history and a 1 to 3 season forecast with a 10 to 90% range.

Choose how far to look ahead (1, 2 or 3 seasons) and sort by current value, biggest predicted rise or fall, or most uncertain forecast. The table's Change and Likely range columns follow the chosen season.

The page opens on the most valuable player, and four examples (Established star, Rising young player, Veteran in decline, Hardest to predict) are picked from the data by fixed rules each time the app starts, so they stay valid when the bundle changes.

Data: Transfermarkt, via the Kaggle datasets [player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) by davidcariboo and [football-datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets) by salimt. Coursework only.
