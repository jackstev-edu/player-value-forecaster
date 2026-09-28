# GenAI usage log

The rubric asks us to document and reflect on GenAI use. Add a row whenever a tool materially shapes code, data or writing.

| Date | Who | Tool | Task | What we kept / changed | Reflection |
| --- | --- | --- | --- | --- | --- |
| 2026-09-16 | Jack | Claude | Read team Drive, scaffolded repo, GUI with mock data, rationale doc | Pending team review | |
| 2026-09-27 | Yunus | Claude (Claude Code) | Scope change to the 500 hand-collected contracts: sheet checks, sample builder, contract-span augmentation to 1,000, player-grouped folds, removal of the Kaggle panel, docs (`docs/scope_change.md`) | Yunus chose each option (Kaggle for values only, span augmentation, delete old code, new branch off main); code written test-first, 34 new tests | Pending Jack's review in the PR |
| 2026-09-27 | Yunus | Claude (Claude Code) | Restored the panel as 42 joined features (#43); looked up 4 wrong contracts with sources; checked Transfermarkt history for the 69 unvalued rows; later-season fallback and value backfill (#44) | Yunus chose all feature groups, dropping unmatched augmented rows, and backfilling the last 14 rows; Claude's sheet notes were removed at Yunus's request | Backfilled rows use a value from after the anchor: flagged, report with and without |
