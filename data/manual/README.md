# Manual dataset (placeholder)

**Status: undecided.** The rubric requires at least 500 samples we authored or collected ourselves (the spec heading says 1000; confirm with the instructor). Kaggle scrapes do not count.

Whatever we choose, each sample must record:

| Field | Why |
| --- | --- |
| `sample_id` | Stable key |
| `collector` | `jack` or `yunus` |
| `collected_at` | Audit trail |
| `source_note` | How it was produced |
| `player_id` (if relevant) | Joins to Transfermarkt IDs |

Candidate ideas are listed in `docs/decisions.md`.
