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
| 17 | **A club's strength for season S is priced as of the anchor that opens season S+1**, not averaged over the season | 2026-09-21 | Pricing at the season-end anchor makes the leakage rule identical to the one that picks the population: a feature on an anchor can only ever have read valuations dated on or before that anchor. A season average would be defensible too, but it would need its own leakage argument for no obvious gain. Verified on real data: for club 3 season 2023 the latest valuation used is 2024-05-31 against a 2024-07-01 cutoff, and a player revalued in the summer window carries his pre-window price. |
| 18 | **Club strength is four numbers — squad size, total, median and top-5 mean** | 2026-09-21 | Total and median move apart when a club is top-heavy, and the top-5 mean is the star-power measure that neither captures: Real Madrid's top-5 mean (€138M) beats Manchester City's (€124M) in 2023 even though City's total squad value is higher. Squad size is carried so the model can tell "cheap squad" from "small squad". |
| 19 | **Positional rank is split out of task 10 into its own task** | 2026-09-21 | Club strength and European flags are club-season aggregates that join on `(club_id, season)`; positional ranks have to be computed per anchor without looking forward, which is a different and fiddlier leakage problem. Landing the join-based half first keeps the risky part isolated. |
| 20 | **League strength is built from the club-season strengths, not from players directly** | 2026-09-21 | A league is then the sum of its clubs, and `league_median_club_value_eur` is the median *club* rather than the median player, which is the quantity that actually separates leagues. Verified on real data: the 2023 totals rank GB1 €12.7bn ≫ ES1 €5.6bn > IT1 €5.4bn > L1 €4.9bn > FR1 €4.3bn > PO1 > NL1 > BE1 > TR1, and club counts per league match the real competitions exactly. Note ES1 outranks IT1 on total but trails it on median — Spain is top-heavy, which is exactly why both numbers are carried. |
| 21 | **`club_value_share_of_league` is carried alongside the raw values** | 2026-09-21 | The same squad value means a different thing in the Premier League than in the Eredivisie, and neither the club number nor the league number says that alone. Real data: median share 3.3%, max 38.3% (club 610 in NL1, 2018). |
| 22 | **Positional rank is computed against the panel's own members at each anchor**, in two scopes: the player's club and his league | 2026-09-21 | A cross-sectional comparison at a single moment cannot look forward, and `value_eur` is already the as-of-anchor value, so no extra leakage argument is needed. Peers are panel members, so a club-mate without a fresh valuation is not counted — `position_peers_*` carries the group size so the model can tell rank 3 of 4 from rank 3 of 20. Verified on real data: the top-ranked league attacker at anchor 2024 is Haaland (GB1), Vinicius (ES1), Mbappé (FR1), Lautaro (IT1), Kane (L1). |
| 23 | **Players whose position is the sentinel `"Missing"` get null ranks, not a rank** | 2026-09-21 | `players.position` has no nulls but 586 players hold the literal string `"Missing"`. Ranking them grouped each into a set of one or two and returned rank 1 with a percentile of 1.0, which a model reads as "best in his position" when it means the opposite. 57 such rows reach the panel. See the fourth trap in `docs/data_coverage.md`. |
| 24 | **Value history is six columns, and the 12-month lookback is staleness-capped the same way the anchor value is** | 2026-09-21 | `peak_value_eur`, `value_vs_peak`, `value_12m_ago_eur`, `value_change_12m`, `years_of_history`, `n_valuations_so_far`. The cap stops a five-year-old price standing in for "a year ago". Days since the last revaluation is already on the panel as `value_age_days`, so it is not recomputed. Verified: 0 rows value above their own running peak, `value_vs_peak` maxes at exactly 1.000, and 39.9% of rows sit at their peak. |
| 25 | **Newcomers keep their rows and take nulls** in the columns that need a past | 2026-09-21 | This is decision #6 made concrete. 10.0% of panel rows (7,017) have no 12-month lookback, and the 5,676 rows with under a year of history have a **median age of 19.6** — precisely the group an experience filter would have deleted, and the group whose value moves most. |
| 26 | **`years_of_history` is a second age column and comes out in the ablation** | 2026-09-21 | Settled 2026-09-21, measured on the full panel: `corr(years_of_history, age) = +0.896` over 70,098 rows. The earlier entry recorded this as a suspicion from their separate correlations with `y_h1` (age −0.394, years_of_history −0.374); the correlation between the two had not been computed. It has now. The column stays in the panel so task 15 can drop it as a measured group rather than on an assumption, but it is not expected to survive. The two history features that remain genuinely additive are `value_vs_peak` (+0.292) and `value_change_12m` (+0.247). |
| 27 | **Starts come from `game_lineups`, scoring from `appearances`, and season 2012 gets null starts rather than zero** | 2026-09-21 | Only `game_lineups` says who began a match and only `appearances` says what happened in it, so slot 3 reads both. `game_lineups` starts 2013-07-02 while `appearances` starts 2012-07-03, so counting starting-XI rows returns 0 for all 4,278 season-2012 panel rows (6.1%) — a model would read that as "nobody started that season". Any season the lineup table does not cover is nulled; inside a covered season a missing lineup row is a real "was not named" and keeps its zero. Verified on the real panel: `starts` is 93.9% filled, the 6.1% gap is exactly season 2012, and 0 of 65,872 known rows have starts above squad games. Fifth trap in `docs/data_coverage.md`. |
| 28 | **Per-90 rates are kept raw, with `minutes` carried as the reliability denominator** | 2026-09-21 | Same shape as #22, which carries peer counts beside ranks so the model can tell rank 3 of 4 from rank 3 of 20. A per-90 built on a cameo is a high-variance estimate, not a wrong one, so it is shrunk by nothing and explained by `minutes`, which is already on the panel. Measured: 207 rows (0.33%) exceed 2.0 goals+assists per 90 and their median minutes is 31; the maximum, 90.0, is one assist in one minute. Above 900 minutes the p99 is 0.968. A minutes floor would null ~16% of rows and delete the fringe players the injury and rotation story needs. Revisit in task 15 if the low-minute tail shows up in the error analysis. |
| 29 | **Goals and assists are season totals across all competitions, per club** | 2026-09-21 | `player_club_season` already aggregates minutes across everything in `games`, so scoring follows the same rule and the two stay comparable. Keyed on (player, club, season), so a winter-break move is measured at each club separately and the panel row for the primary club carries only that club's half. Sanity-checked at anchor 2024 above 1,800 minutes: the top ten by goals+assists per 90 are Boniface, Kane, de Jong, Guirassy, Gyökeres, Mbappé, Dovbyk, Salah, Undav and Raphinha. |

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
