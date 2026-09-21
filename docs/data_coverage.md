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

## The three traps

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
is 99.3% filled across 172 countries; `date_of_birth` 99.9%; `position` 100%; `foot` 95.0%;
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
