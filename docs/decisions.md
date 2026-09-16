# Decision log

Open items from the Analysis Guide plus scaffold-time questions. Record the decision, date and who agreed.

| # | Question | Options | Status |
| --- | --- | --- | --- |
| 1 | Season anchor date | 1 July (config default) | Open |
| 2 | Leagues to model | All tracked domestic leagues, or top 5 plus feeder leagues | Open |
| 3 | Final feature list | Apply the four rules in Analysis Guide section 3 | Open |
| 4 | v1 data sources | Injuries joined on `player_id` (yes?), FBref deferred to v2 | Open |
| 5 | Second model type | Chronos off-the-shelf (zero-shot), optionally fine-tuned | Proposed |
| 6 | Manual dataset (500+) | See candidates below | Open, blocking |
| 7 | GUI framework | Gradio assumed; confirm against the course GUIs module | Confirm |
| 8 | Heading says 1000 samples, text says 500 | Ask instructor | Open |

## Manual dataset candidates

The manual set should feed the project, not sit beside it.

- **Human baseline forecasts.** Each of us predicts the 1-season value change for 250+ players using only information shown for a past anchor (blind to the outcome). This gives a "can the model beat two fans" benchmark for the report. Risk: remembering real outcomes, so pick lower-profile players.
- **Hand-curated context table.** Per league and season: European places, coefficient rank, notable rule changes. Directly feeds the context features. Risk: may land under 500 rows unless it goes to club-season level.
- **Labeled transfer or value-change reasons.** For 500+ large value jumps, tag the cause (breakout season, transfer, injury, contract, national team). Supports error analysis and possibly a small classifier.
