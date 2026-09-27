# Manual dataset: contract start and end for 500 players

Task 19, the rubric's "500+ samples we collected ourselves", and since 2026-09-27 the core of
the project (`docs/scope_change.md`, decision #36).

**How it was collected.** `scripts/make_contract_sheet.py` picked the 500 most valuable players
(today's Transfermarkt value) and pinned each to one random past season. Yunus took odd ranks and
Jack even ranks, 250 each, and looked up the contract in force with that season's club on 1 July,
from club announcements, news and Wikipedia. The sheet is `Contract Data Collection.xlsx` on the
team Drive.

**Files** (kept on the Drive, not in the public repo, like all our data):

| File | Written by | Contents |
| --- | --- | --- |
| `contracts.csv` | `scripts/build_dataset.py --sheet` | The sheet minus the Kaggle-derived hint columns |
| `contract_issues.csv` | `scripts/build_dataset.py` | One row per problem found by `pvf.data.manual.check_contracts` |

| Column | Meaning |
| --- | --- |
| `rank` | Stable key, 1 to 500 |
| `collector` | `Yunus` or `Jack` |
| `player_id` | Transfermarkt id |
| `player_name`, `season`, `anchor_date`, `club_that_season` | What to look up (prefilled) |
| `contract_start`, `contract_end` | Collected, `YYYY-MM-DD` |
| `source_url` | Collected: the page that states the dates |
| `notes` | Loans, option years, anything odd |

**Checks.** Errors drop a row from modelling: missing dates, end not after start, contract expired
before the anchor, contract starting a full season late, duplicate player. Warning only: a source
that is not a URL.
