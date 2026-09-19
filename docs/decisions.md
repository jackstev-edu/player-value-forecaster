# Decision log

Every decision that shapes the model, with the reason and the evidence behind it.
Add a row when something is settled; never delete a row, supersede it instead.

Evidence marked "measured" was produced by the coverage analysis on 2026-09-19 against the
parquet in the team Drive, and is written up in `docs/data_coverage.md`.

---

## Settled

| # | Decision | Date | Why |
| --- | --- | --- | --- |
| 1 | **Anchor date is 1 July**, one row per player per season | 2026-09-19 | Sits just before the season starts. Transfermarkt re-values roughly every 160 days (measured), so an anchor value is at most ~5 months stale. |
| 2 | **Target is `log(value at anchor+h / value at anchor)`**, h = 1, 2, 3 | 2026-09-15 | Values run €10k to €200M. Predicting raw euros lets the stars dominate the loss. Convert back to euros for the GUI. |
| 3 | **Nine leagues**: GB1, IT1, ES1, L1, FR1, PO1, BE1, NL1, TR1 | 2026-09-19 | The set Yunus and Jack care about, minus Poland (#4). |
| 4 | **Poland (PL1) dropped** | 2026-09-19 | Measured: `games` covers Ekstraklasa only from season 2024. Club strength, league position and results do not exist for earlier anchors, so every context feature would be null for 12 of 13 seasons. |
| 5 | **First anchor is 2013, not 2012** | 2026-09-19 | Measured: `appearances` starts 2012-07-03, so a 2012-07-01 anchor has no prior-season performance data. Supersedes `min_anchor_season: 2012` in the original config. `game_lineups` starts later still (2013-07-02), so season 2012 is covered by `appearances` alone — see #15. |
| 6 | **Value-history depth is a feature, not a filter.** Minimum ≥1 year; `years_of_history`, `value_12m_ago`, `peak_value_so_far` are columns that may be null | 2026-09-19 | See "Why we did not require 8 years" below. This one is easy to get wrong later, so read that section before changing it. |
| 7 | **Injuries stay in v1**, using the existing `player_injuries` table | 2026-09-19 | See "Why we kept our own injury data" below. |
| 8 | **Sofascore ratings: validate on the free sample before scraping** | 2026-09-19 | See "Why we are staging the Sofascore work" below. The scrape is intended, not cancelled. |
| 9 | **League membership is derived from `appearances`/`games`, never read off `player_valuations.player_club_domestic_competition_id`** | 2026-09-19 | Measured: that column's club→league mapping goes stale from 2024. Players carrying a top-9 id drop 12,495 (2023) → 6,573 (2024) → 5,906 (2025) while total valuation volume barely moves and the null rate stays flat. Trusting it silently halves the most recent test seasons. |
| 10 | **Per-season club strength is rebuilt from `player_valuations` aggregated by club-year** | 2026-09-19 | Measured: `clubs.total_market_value` is 100% empty, and the whole `clubs` table is an undated scrape-time snapshot. |
| 11 | **European participation comes from `games`** where `competition_id` is in CL, CLQ, EL, ELQ, UCOL, ECLQ | 2026-09-19 | Measured: `team_competitions_seasons` contains no UEFA competitions. Searching it for "Champions/Europa/Conference" returns only the English Championship, USL Championship and Scottish Championship — second-tier domestic leagues, not European football. |
| 12 | **Yellow and red cards dropped** from the feature list | 2026-09-19 | Jack and Yunus, column review. Little plausible link to value once minutes are known. |
| 13 | **Nationality and club history added** to the feature list | 2026-09-19 | Measured: `country_of_citizenship` is 99.3% filled across 172 countries; `transfer_history` covers 85.9% of our players with a median of 10 moves each, including loans. |
| 14 | **A player is in the panel at anchor Y if they were named in a matchday squad for a 9-league club in season Y-1** | 2026-09-19 | Purely historical, so it cannot leak the summer window. Using `game_lineups` rather than `appearances` keeps unused substitutes — verified: 62.3% of `substitutes` rows have no appearance row, so they never took the field. That matters because an appearance-based rule would drop players who missed a season injured, the exact group whose injury history the model is meant to learn from. Minutes are carried as a feature instead of being turned into a cutoff. |
| 15 | **Season 2012 falls back to `appearances`** for squad membership | 2026-09-19 | `game_lineups` starts 2013-07-02, so a squad-only rule would push the first anchor to 2014 and cost ~5,700 rows. Season 2012 lands in the earliest training slice, where a slightly narrower population (no unused subs) is low-risk. Implemented as a union of the two sources, so there is no special case in the code. |
| 16 | **The panel population is ~5,700 players per season, not ~11,400** | 2026-09-19 | A consequence of #14, recorded so it is not mistaken for data loss. The larger count came from counting anyone whose valuation row carried a 9-league club id, which includes youth, reserve and loaned-out players. The squad rule keeps actual first-team players, giving roughly 68,000 anchor rows across 2013–2025. |

---

## Why we did not require 8 years of value history

**The question.** Deeper history per player should mean better features, so why not require every
training row to have 8 years of prior valuations? The data does support it: 51.3% of our players
clear that bar, giving 49,254 rows across 2013–2025.

**Why we did not.** Transfermarkt starts valuing players at 16–18, so "8 years of history" is
almost exactly "age ≥ 25". At the 2020 anchor:

| Age band | All rows | Survive ≥8 yrs | Kept |
| --- | --- | --- | --- |
| 19–21 | 1,017 | 0 | 0% |
| 21–23 | 1,531 | 1 | 0.1% |
| 23–25 | 1,629 | 24 | 1.5% |
| 25–27 | 1,641 | 356 | 21.7% |
| 27–30 | 2,151 | 1,426 | 66.3% |
| 30–33 | 1,750 | 1,581 | 90.3% |
| 33+ | 1,516 | 1,481 | 97.7% |

Median age moves from 26.5 to 31.1. The rule deletes essentially every player under 23 — the
population whose value actually moves, and the one a scouting tool is most useful for. Training
rows at the 3-season horizon fall from 65,838 to 17,921.

**What we do instead.** Minimum ≥1 year of history (keeps 93.5% of players) and expose depth as
features that are null for newcomers. LightGBM handles nulls natively, so veterans get the deep
history without the young players being thrown away.

**If you are tempted to revisit this:** the honest test is an ablation, not an argument. Train
with the ≥1-year rule and add `years_of_history` as a feature, then check whether error on the
under-23 slice is acceptable. If it is not, the answer is a separate model for young players, not
a filter that removes them.

---

## Why we kept our own injury data

**The question.** Our `player_injuries` table has 349 distinct free-text `injury_reason` values
and 19% of rows say literally "unknown injury". Is there a cleaner source?

**What we checked (2026-09-19).**

| Source | Records | Players | Verdict |
| --- | --- | --- | --- |
| **Ours (football-datasets)** | **143,195** | **34,561** | Largest available |
| FigShare "Injuries from Transfermarkt.com" | 107,000 | 18,500 | Smaller subset |
| Kaggle European Football Injuries 2020-2025 | 15,000 | Top-5, 5 seasons | Subset, useful only as a cross-check |
| Kaggle Player Injuries & Team Performance | ~600 | 7 PL clubs | Negligible |

Every alternative is scraped from Transfermarkt too, so none of them fixes the text quality —
they are smaller copies of the same thing.

**Why it does not block us.** The features that matter most (`days_injured_last_season`,
`n_injuries_3yr`, `longest_absence`, `had_acl`) are computed from dates and durations, not from
the reason text — and `days_missed` and `games_missed` are **100% filled**, median 22 days,
p90 129. The messy text only affects injury *type*, which we group into ~8 buckets
(muscular, knee/ligament, ankle/foot, illness, knock, surgery, other, unknown).

**Known limit to state in the report:** 61% of our players have at least one injury record.
Absence of a record means "not recorded", not "never injured", so a zero in
`days_injured_last_season` is partly wrong. The injury table also ends December 2025.

---

## Why we are staging the Sofascore work

**The intent.** Yunus wants Sofascore player ratings in the model and believes they will move
the needle. This decision is about *sequencing*, not about dropping them.

**The situation.** Neither of our datasets has a rating column — not `appearances`, not
`player_performances`. So ratings must come from outside. Two routes:

1. **Kaggle `akshankrithick/sofascore-seasonwise-ratings-football-soccer`** — 13,018 player-season
   rows, MIT licensed, free, today. Limits: top-5 leagues only (no PO1/BE1/NL1/TR1), seasons
   2017-18 to 2023-24, and **no Transfermarkt `player_id`** so it joins on name + birth year.
   Covers roughly 10–15% of our ~94k training rows.
2. **Scraping Sofascore's public API** — full coverage of all 9 leagues and all seasons. Costs:
   ~22k players × 13 seasons of rate-limited requests, a name-based join to build anyway, and
   bulk collection that Sofascore's terms do not authorise. That last point needs a defensible
   answer in the report, not a shrug.

**The decision.** Do route 1 first as an experiment, then decide on route 2 with evidence.
Specifically: join the Kaggle ratings to the top-5 subset, train with and without the rating
feature, and compare error on the overlapping rows.

**Why this order.** Both possible outcomes are useful. If ratings help, we have a measured
effect size that justifies the scrape and makes it a defensible, documented choice in the
writeup. If they do not, we saved days of scraping and the negative result is itself a finding
worth reporting. Either way the report gets a real answer instead of an assumption, which is
the point of the ablation framing in the Analysis Guide.

**Trigger to start route 2:** the top-5 ablation shows a meaningful error reduction from the
rating feature. Revisit the terms-of-use question at that point.

---

## Still open

| # | Question | Notes |
| --- | --- | --- |
| A | Second model family | `chronos` is proposed in the config as the off-the-shelf model. Confirm against the rubric. |
| B | Manual dataset (500+ rows) | Blocking, rubric requirement. Candidates below. |
| C | GUI framework | Gradio assumed; confirm against the course GUIs module. |
| D | Rubric says 1000 samples in the heading, 500 in the text | Ask instructor. |
| E | Deadline and deliverables | Parked by Yunus on 2026-09-19, to revisit. |
| F | Test-season split | Original config uses 2022–2025, but a 2023+ anchor has no 3-season target (values end 2026-06-12) and 2024+ is affected by decision #9. Needs setting once #9 is implemented. |
| G | Repo is public | Worth a look at the course academic-integrity policy before the report goes in. |

### Manual dataset candidates

- **Human baseline forecasts.** Each of us predicts the 1-season value change for 250+ players
  from a past anchor, blind to the outcome. Gives a "can the model beat two fans" benchmark.
  Risk: remembering real outcomes, so pick lower-profile players.
- **Hand-curated context table.** Per league and season: European places, coefficient rank,
  notable rule changes. Feeds the context features directly. Risk: may fall under 500 rows
  unless taken to club-season level.
- **Labeled value-change reasons.** For 500+ large value jumps, tag the cause (breakout season,
  transfer, injury, contract, national team). Supports error analysis.
