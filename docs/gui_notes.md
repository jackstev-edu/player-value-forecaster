# GUI maintenance notes

## Recheck after any Gradio upgrade

The Space runs Gradio 6.28.0, pinned by `sdk_version` in `app/README.md`. The current
list of Gradio internals the app leans on is under "Gradio internals the redesign depends
on" below. Rerun the probes in `scripts/ui_probes/` after any bump, since every one of
those rules fails silently: the app still runs and only the look degrades.

### Historical, for the table based app (replaced by the FIFA redesign, 2 Oct 2026)

The old single page app had a `gr.Dataframe` and an accordion, and its phone CSS targeted
`virtual-row`, `virtual-body`, `header-table`, `first-column`, `tab-like-container` and
`label-wrap`. It also worked around `gr.Dataframe(pinned_columns=1)`, which Gradio 6.28
accepts but never renders, by making the first column `position: sticky`. None of those
elements exist in the redesign, so the rules went with them. The note is kept because the
`pinned_columns` gap is still real if a table ever returns.

## Real prediction bundle (branch `gui/real-data`, 1 Oct 2026)

The app now ships `lgbm-core-v1`, the 4-feature core model (decision #48), covering
4,727 players across 9 leagues. The mock bundle moved to `tests/fixtures/mock_bundle/`
so the callback tests keep exact expectations while the shipped bundle changes.

### How tests pick a bundle

`app/app.py` reads `PVF_BUNDLE_DIR`, defaulting to `app/predictions/`. The bundle is read
at import, so an override only has to hold while the module executes.

| File | Bundle | Why |
| --- | --- | --- |
| `tests/test_app_callbacks.py` | `tests/fixtures/mock_bundle/` | Invented players with known names and all three horizons, so assertions can be exact |
| `tests/test_app_smoke.py` | whatever is in `app/predictions/` | Catches a real export that breaks the display; asserts shape and sanity, never a named player |

Run the smoke test after every re-export: `pytest tests/test_app_smoke.py`.

### What the real data changed

1. **One horizon, not three.** The model forecasts one season (`target.horizons: [1]`).
   The app no longer hard-codes 1, 2 and 3: `offered_horizons()` reads the horizons out of
   the forecasts table, and the "Look ahead" radio is hidden when only one is offered.
   Before this, picking 2 or 3 seasons showed a full table of `n/a`, and "Most uncertain"
   silently fell back to current value. When a bundle carries more horizons again, the
   control reappears on its own, with no code change.
2. **Gaps must not read as `nan`.** One player had `position = "Missing"` and a null
   `sub_position`, which reached the table and the card as the literal text `nan`.
   `clean_labels()` turns nulls and the placeholders in `MISSING_TOKENS` into `Unknown`
   across the six shown text columns, so the filter choice reads "Unknown" too.
3. **The forecasts lean upward, and the page now says so.** Decision #48 accepted a known
   bias: training players were collected because they became valuable, so 96% of forecasts
   predict a rise, median 2.5x. `optimism_note()` recomputes that share and median from the
   loaded bundle and shows them in the banner that used to carry the demo-data warning. It
   stays silent below a 60% lean, so a less biased bundle drops the banner by itself.
4. **"How the forecast works" states real results.** It previously promised that results
   would appear once the real model arrived. It now names the cross-validation, the
   walk-forward check (6 of 10 inside the band) and the three known limits.
   `test_how_it_works_states_its_limits` fails if any limit is edited out.

### Measured on the real bundle

| Item | Value |
| --- | --- |
| Players, history rows, forecasts | 4,727 / 89,537 / 4,727 |
| Bundle size on disk | 536 KB total, largest file 257 KB (`history.parquet`) |
| Startup | 9.4 s, of which 4.5 s is `import gradio` and 1.5 s the two `groupby` lookups |
| Filter, search, open a player | 14 ms, 8 ms, 14 ms |
| First paint (`initial_view`) | 342 ms |
| Flagged low confidence at 1 season | 20.6% (20.0% wide range, 2.7% short history) |

Startup is dominated by importing Gradio, so a Space cold start is unavoidably several
seconds. Nothing here is close to the 50 MB file size GitHub warns about.

### Known limits, not fixed here

- **A null age would hide a player completely.** `age.between(lo, hi)` is False for a null,
  so such a player could never be reached by any filter. The current bundle has no null
  ages, and `test_app_smoke.py` asserts every age sits inside the slider range, so this
  would be caught on the next export rather than shipped silently.
- **The card's forecast table wraps badly at 390px.** "Median" breaks as "Media n" and
  "€70.4m" as "€70.4 m". Pre-existing, cosmetic, and inherited by the FIFA redesign only if
  that layout is reused.
- The probes in `scripts/ui_probes/` target the FIFA redesign hooks and do not run against
  this app; screenshots here were taken with an ad-hoc Playwright script.

## Gradio internals the redesign depends on

These selectors are Gradio's own class names, not ours, so an upgrade can change them
with no error. Recheck each after any Gradio bump.

| Selector in `app/style.css` | What it does | How it fails |
| --- | --- | --- |
| `.pvf-chips .wrap`, `.pvf-chips label` | Turns the sort Radio into a chip row | Chips fall back to a stacked radio list |
| `.pvf-chips > span.svelte-1gfkn6j`, `.block-title` | Hides the form label above the chips | A stray "Sort by" label reappears |
| `.wrap.default`, `div[data-testid="status-tracker"]` | Darkens the loading indicator | A white flash on every screen change |
| `.gradio-container footer` | Panels Gradio's own footer over the pitch | Footer links drop below 4.5:1 at tall windows |
| `div.modebar-container` (Plotly) | Hides the chart toolbar on phones | Untappable icons return on small screens |

The spike also found that `gr.Navbar(visible=False)` does not hide the multipage navbar in
6.28. The app is single page now, so that is not used, but it is worth knowing.

Rerun the probes after an upgrade, especially the contrast sweep across window heights:

```powershell
& $py scripts/ui_probes/contrast.py --space --sizes 1366x700,1366x1200
```

## Redesign rules (FIFA style, branch `gui/fifa`)

The redesign is built on the long-running branch `gui/fifa`. The spike on `spike/fifa-ui`
tested what Gradio 6.28 can do; its findings are in `spike/README.md` on that branch.

Decided on 26 Sep 2026: screens switch by `gr.Column` visibility; player cards are
`gr.HTML` with `trigger()`; age uses two core sliders in one row; a CSS-only pitch floor;
big `size="lg"` buttons with Barlow Condensed loaded by `@import`. Gradio's footer stays,
on a dark panel. The browser Back script is included (Forward is not handled). More than
24 results appear through a "Load more" button that adds 24 at a time.

1. **Scope every CSS rule under one `#pvf` id.** Wrap each page in
   `gr.Column(elem_id="pvf")` and start every selector in `style.css` with `#pvf `. Locally
   Gradio prefixes `css=` rules with `.gradio-container ... .contain`, so they win. On
   Spaces (SSR mode) it does neither: the sheet is unprefixed and loads before Gradio's own
   sheets, so rules of equal weight lose, for example to `.gradio-style button` and
   `.prose *`. An id outranks every class-only Gradio rule in both places. The one
   exception is an element outside `#pvf`, such as the hidden history button, which uses
   `!important`.
2. **Put page-level CSS in `head=`, not `css=`.** This covers the `html` background, the
   transparent `body` and `.gradio-container`, the pitch (`body::before` and `::after`),
   the navbar hide rule and the footer panel. `css=` rules are prefixed under `.contain`
   locally, so `html` and `body` selectors never match there. Two details:
   - SSR puts `background: var(--body-background-fill)` inline on `<html>`, so the
     page background needs `!important`.
   - `gr.Navbar(visible=False)` does not hide the multipage navbar in 6.28, so hide
     `.gradio-container > .nav-holder` with CSS.

   The font `@import` can stay in `css=`, since Gradio hoists `@import` lines out of it.
3. **Run the scroll-restore JS after every screen change.** Chain
   `.then(None, js=AFTER_NAV_JS)` on every event that changes screen. That includes each
   Back button and the hidden button the browser Back script clicks. Going forward
   scrolls to the top; going back restores the saved position. Without this, SEARCH at the
   bottom of the phone Search screen opens Results halfway down the list. Known limit: at
   the very bottom of Results the restore can land about 14px short, because the page is
   briefly shorter while the cards re-mount.
4. **Escape player names in card HTML.** `gr.HTML` renders raw markup, so pass every
   bundle string (name, club, league, nationality, position) through `html.escape` before
   it goes into a card or header. An unescaped `&` or `<` in a name breaks the card or
   injects markup.
5. **Keep cards keyboard accessible.** Each card is a real
   `<button type="button" class="pvf-card" data-pid="...">` with an `aria-label` naming
   the player, so Tab reaches it and Enter or Space opens it. Show a visible
   `:focus-visible` ring. Listen with one delegated click handler on `element` in
   `js_on_load` that calls `trigger('click', {pid})`, never a `div` with its own handler.
   "Load more" is a real button too, and new cards are appended after the old ones, so
   the Tab order stays in reading order.
6. **Keep the probe hooks.** `scripts/ui_probes/` finds the app through class hooks
   (`.pvf-home`, `button.pvf-go`, `button.pvf-card[data-pid]`, `.pvf-player-head[data-pid]`,
   `button.pvf-back` and others). The full list is in `scripts/ui_probes/README.md`. Keep
   them, or update `probe_common.py` in the same commit.

After a Gradio upgrade, also recheck the navbar hide rule (`.nav-holder` is an internal
class) and whether SSR still sets the inline `<html>` background.

## Local run that matches the Space (SSR mode)

A plain local run is not a faithful preview of styling (see rule 1). Spaces serve
Gradio in SSR mode, which needs Node.js 20 or higher. Gradio looks for Node in
`GRADIO_NODE_PATH`, then on `PATH`.

```powershell
# PowerShell, from the repo root; set only for this terminal
$env:GRADIO_SSR_MODE = "true"
.venv\Scripts\python.exe app\app.py
```

```bash
# Git Bash
GRADIO_SSR_MODE=true .venv/Scripts/python.exe app/app.py
```

To check it really is SSR, view the page source. The `<html>` tag should carry an inline
`background: var(--body-background-fill)`, and asset links start with `_app/immutable/`.

To compare it with the preview Space, use the screenshot and compare probes (see
`scripts/ui_probes/README.md`):

```powershell
$py = "$env:USERPROFILE\.venvs\pvf-probes\Scripts\python.exe"
& $py scripts\ui_probes\screenshots.py --local --out scripts\ui_probes\out\local
& $py scripts\ui_probes\screenshots.py --space --out scripts\ui_probes\out\space
& $py scripts\ui_probes\compare.py
```

Status on 2 Oct 2026: Node.js 24.19.0 is installed, so the SSR run is available, but the
comparison against the preview Space has not been run yet. Until it has, the preview Space
`jackstev/pvf-preview` stays the reference for styling, and a CSS change should be checked
there rather than only locally. Note that `app/app.py` reads `style.css` once at import,
so a local run must be restarted before a CSS edit shows.
