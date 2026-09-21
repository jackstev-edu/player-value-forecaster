# Data coverage

What the data can actually support, measured on 2026-09-19 against the parquet in the team Drive
(`Dataset/player-scores` and `Dataset/football-datasets`). Figures are for the nine chosen
leagues unless stated. `docs/decisions.md` records what we decided off the back of this.

Read this before adding a feature. Each trap below was assumed to work during design, and each
would have failed **silently** — producing null or wrong columns rather than raising.

---

## Time windows

| Source | Covers | Notes |
| --- | --- | --- |
| `player_valuations` | 2000-01-20 → 2026-06-12 | Thin before 2004, crosses 20k rows/year in 2012 |
| `appearances` | **2012-07-03** → 2026-06-28 | **The binding constraint.** Any prior-season performance feature means the first usable anchor is 1 July 2013 |
| `games` (9 leagues) | season 2012 → 2025 | `home_club_position` 100% filled every season |
| `player_injuries` | real volume from ~2008, ends 2025-12-22 | |
| `transfer_history` | 1900 → 2027 | Dirty edges, needs trimming |

Transfermarkt re-values players about every **160 days** (median gap, stable since 2012), so a
1 July anchor value is at most ~5 months stale.

## Player retention

Share of anchored players who still have a value later. Stable, no collapse:

| Horizon | Retention |
| --- | --- |
| +1 season | ~82% |
| +2 seasons | ~79% |
| +3 seasons | ~75% |

## Training rows available, 2013–2025 anchors

| Horizon | All | With ≥3 yrs history | With ≥8 yrs history |
| --- | --- | --- | --- |
| 1 season | 93,886 | 72,216 | 31,779 |
| 2 seasons | 79,500 | 60,011 | 24,356 |
| 3 seasons | 65,838 | 48,753 | 17,921 |

History depth is a feature, not a filter — see decision #6 for why the ≥8 column is a trap.

---

## The six traps

### 1. `clubs.total_market_value` is 100% empty

Every row. Not stale, not partially filled — empty. The whole `clubs` table is an undated
scrape-time snapshot anyway (`squad_size`, `average_age` carry no date), so it cannot describe a
club at a past anchor even if the column were populated.

**Do instead:** aggregate `player_valuations` per club-year. There are 258+ club-years with ≥10
valued players every year from 2012 on, which is enough for squad value, median value and
positional ranks.

### 2. European participation is not in `team_competitions_seasons`

Searching that table for "Champions | Europa | Conference" returns only the **English
Championship**, **USL Championship** and **Scottish Championship** — second-tier domestic
leagues whose names happen to match. There are no UEFA competition rows in it at all.

**Do instead:** read `games` where `competition_id` is in `CL, CLQ, EL, ELQ, UCOL, ECLQ`.
Coverage is clean for seasons 2012–2025:

| Competition | Clubs per season |
| --- | --- |
| Champions League (CL) | 32, rising to 36 from 2024 |
| Europa League (EL) | 56, 40 from 2021 |
| Conference League (UCOL) | 40 from 2021, when it started |

### 3. `player_club_domestic_competition_id` goes stale from 2024

Distinct players holding a top-9 league id on their valuation rows:

| Year | Players |
| --- | --- |
| 2022 | 12,438 |
| 2023 | 12,495 |
| 2024 | **6,573** |
| 2025 | 5,906 |
| 2026 | 4,593 |

Total valuation volume barely moves across those years and the null rate stays flat at ~20–25%,
so this is the column's club→league mapping decaying, not players leaving. Filtering leagues on
it would silently halve our most recent test seasons.

**Do instead:** derive league membership from `appearances` / `games` — who the player actually
played for in that season. This is task 6 on the roadmap and it gates every context feature.

---

## Field-level notes

**Clean:** `player_valuations` has 0 nulls, 0 duplicate `(player_id, date)` pairs and exactly
**1** zero-value row across 656,301 rows (min non-zero €10k, max €200M). Guard the log ratio
against that single row and the table needs no other defence.

**Identification:** players are uniquely identified by name + birth date. `country_of_citizenship`
is 99.3% filled across 172 countries; `date_of_birth` 99.9%; `position` has no nulls but
**586 players carry the literal string `"Missing"`** (see the fourth trap); `foot` 95.0%;
`height_in_cm` 95.3% but **4.7% are zero, meaning unknown**.

**Injuries:** 143,195 rows over 34,561 players. `days_missed` and `games_missed` are 100% filled
(median 22 days, p90 129, max 8,655 — that upper tail is bad data). But `injury_reason` is free
text with **349 distinct values**, the largest being "unknown injury" at 27,028 rows (19%), plus
inconsistent duplicates like "Muscle injury" (6,433) vs "muscular problems" (4,880). Group into
~8 buckets before use. 61% of our players have at least one record, so a missing record means
"not recorded", not "never injured".

**No player ratings exist** in either dataset. Neither `appearances` nor `player_performances`
has a rating column. See decision #8.

**Transfers:** prefer `transfer_history` (1.1M rows, 85.9% of our players, median 10 moves each,
typed Transfer / Loan / Return from loan / Draft, with `value_at_transfer`) over `transfers`
(175k rows), because it includes loans.

**Duplicates:** `team_competitions_seasons` is ~73% exact duplicate rows — deduplicate on load.
Smaller duplicate counts exist in `player_injuries` (111), `player_national_performances` (32)
and `transfer_history` (123).

---

### 4. `players.position` says "Missing" instead of null

`position` has zero nulls, which reads as 100% coverage and is how it was first recorded here.
It is not: **586 of 50,149 players carry the literal string `"Missing"`**, with `sub_position`
null alongside. Counting nulls will never find them.

This bites anything that groups by position. Ranking players within position put these 586 in
groups of one or two, handing each one rank 1 and a percentile of 1.0 — a model reads that as
"the best attacker at his club" when it means "we do not know what he plays". 57 of them reach
the panel.

**Do instead:** treat `"Missing"` as unknown, not as a position. `pvf.features.context`
keeps the set in `UNKNOWN_POSITIONS` and nulls the rank columns for those rows. The signature
that it is working: `position_peers_in_league` has a minimum of 22 (the thinnest real group is
goalkeepers) rather than 1.

### 5. `game_lineups` does not exist for season 2012, so starts read as zero

`game_lineups` begins on **2013-07-02**. `appearances` begins a year earlier, on 2012-07-03, and
season 2012 is in the panel because of that: it is the prior season for the 2013 anchor, and the
squad rule falls back to appearances for it (decision #15).

The two tables do not cover the same span, and only `game_lineups` says who *started*. Counting
starting-XI rows per player-season therefore returns **0 for every player in season 2012** — not
because they were substitutes, but because the table has no rows to count. A model reads a
column of zeros as "nobody in this season ever started". **4,278 panel rows** are drawn from
season 2012, 6.1% of the panel.

**Do instead:** null the start columns for any season `game_lineups` does not cover, rather than
letting the count stand. `pvf.features.performance` checks which seasons appear in the joined
lineup frame and nulls `starts` outside them; within a covered season a player with no lineup row
genuinely was not named, and keeps his zero. Minutes, goals and assists come from `appearances`
and are unaffected, so season 2012 rows keep every other performance column.

### 6. Injury coverage grows across the window, and collapses at the 2026 anchor

`player_injuries` covers whoever Transfermarkt had recorded by scrape time, and that is a
different set of players in 2013 than in 2023. Measured per anchor, on panel rows:

| Anchor | 2013 | 2016 | 2019 | 2020 | 2022 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Share of rows whose player has any injury record | 0.718 | 0.785 | 0.839 | **0.856** | 0.840 | 0.777 | 0.760 | 0.700 |
| Share with an injury in the 12 months before | 0.253 | 0.305 | 0.414 | 0.430 | 0.501 | 0.484 | 0.513 | **0.136** |
| Mean days injured in that window | 19.9 | 23.3 | 29.6 | 29.4 | 32.6 | 38.3 | **41.8** | 7.5 |

The apparent injury rate **doubles** between 2013 and 2021 and mean days lost roughly doubles
again by 2025. Footballers did not become twice as fragile; the record got fuller. Our split
puts training at anchors ≤ 2020 and testing at 2022–2025, so a model learns injury from the
thin half of the record and is scored on the full half.

The 2026 anchor is the sharp version of the same thing: `player_injuries` ends **2025-12-22**,
so the 12 months before 1 July 2026 contain barely six months of data. Injury rate falls to
0.136 and mean days to 7.5 — not a healthy season, a truncated file. The 2026 anchor already
carries no targets and is unfit for training; this makes it wrong for **serving** too unless
the injury columns are suppressed for it.

**Do instead:** carry `has_injury_record` so the model can separate "not hurt" from "not
tracked", and treat the injury group as one block in the task 15 ablation, where the time-split
backtest will show whether it survives the drift. Do not read a raw rise in `days_injured_12m`
across seasons as a finding about football. The same caution applies, much more weakly, to
transfers: coverage there runs 0.885 → 0.962 and is close to flat from 2018 on.

## The Europa League share falls in 2022, and that is real football, not a data break

`played_uel` sits at 0.21–0.24 of panel rows for anchors 2013–2021, then drops to 0.15 (2022),
0.15 (2023), 0.13 (2024) and stays there. Nothing broke: UEFA launched the Conference League in
2021-22, which took a third tier of clubs out of the Europa League. `played_uecl` picks them up,
running at 3.4% of rows overall and effectively all of it from anchor 2022 onward. `played_ucl`
is flat at 0.16–0.20 throughout, as it should be — the Champions League field did not change.

Anyone who sees the `played_uel` step and reaches for the data-quality explanation should stop
here. The three flags have to be read together.

## League sizes change, and every change in our window is a real one

`league_club_count` is not constant per league, which looks like a mapping bug and is not.
Across 2012-2025 anchors: Ligue 1 20 -> 18 (2023-24), the Super Lig 18 -> 21 (2023-24), the
Belgian First Division 16 -> 18, the Primeira Liga 16 -> 18. The Premier League, Serie A, La
Liga hold at 20 and the Bundesliga and Eredivisie at 18 throughout, which is correct. Any count
outside 16-21 would be the bug; none appears.

## How to reproduce

The measurements came from throwaway analysis scripts, not committed code. When Phase 1 lands,
`src/pvf/data/load.py` should carry the deduplication and date parsing, and these numbers are
worth re-checking against any refreshed download.
