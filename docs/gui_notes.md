# GUI maintenance notes

## Recheck after any Gradio upgrade

The Space runs Gradio 6.28.0 (pinned by `sdk_version` in `app/README.md`). Two parts of
`app/style.css` rely on details Gradio does not promise to keep. Both can break silently:
the app still runs, but the phone layout degrades.

### 1. The phone layout CSS uses Gradio's internal class names

The rules inside `@media (max-width: 640px)` target elements by Gradio's own class
names, not by classes the app sets:

| Class | What the rule does |
|---|---|
| `virtual-row`, `virtual-body`, `header-table`, `first-column` | Fixed column widths so the table scrolls sideways instead of squeezing words letter by letter |
| `tab-like-container` | Lets the number boxes beside the age sliders grow to 44px without clipping |
| `label-wrap` | Centres the label and arrow on the taller accordion bars |
| `modebar-container` (Plotly) | Hides the chart toolbar, whose icons are too small to tap |

If Gradio renames any of these, the matching rule stops applying with no error.

### 2. The pinned Player column is a CSS workaround

`gr.Dataframe(pinned_columns=1)` is accepted by Gradio 6.28 but does nothing. The
front end receives the setting and never uses it. The Player column stays in view on
phones only because `style.css` makes the `first-column` cells `position: sticky`.
The `pinned_columns=1` argument is kept so Gradio's own pinning takes over if a
later version implements it. At that point, delete the CSS rule so the two do not
fight.

### How to recheck

Open the app at 390px wide (browser dev tools, phone mode) in light and dark, then:

1. The table scrolls sideways inside its own box, the page itself does not, and
   words are not split letter by letter.
2. Scroll the table sideways: the Player column stays pinned on the left.
3. Open "More filters": the age boxes show whole numbers, and the accordion label
   and arrow sit on one line.
4. The chart shows no toolbar on the phone, and on desktop it still has one.

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

Status on 26 Sep 2026: not yet verified. Node was not found on the development machine
(not on `PATH`, not in the installed programs list). Without Node, a run with
`GRADIO_SSR_MODE=true` printed nothing for 150 seconds instead of its URL. Until this is
verified, the preview Space `jackstev/pvf-preview` is the reference for styling.
