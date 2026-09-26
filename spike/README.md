---
title: PVF FIFA UI spike
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Throwaway UI experiment, not the product
---

# FIFA style UI spike (throwaway)

Step 2 of 6 of the FIFA style redesign. This folder is an experiment that tests
what Gradio 6.28 can do before the real design is built. It lives only on the
`spike/fifa-ui` branch and is never merged into `main`.

- `/` switches screens by toggling `gr.Column` visibility.
- `/tabs` switches screens with `gr.Tabs`, tab bar hidden, `selected` set in Python.
- The Search screen has a "Card engine" picker to compare four clickable card methods.
- Add `?nohist` to the URL to turn off the browser Back hook.

Demo data only (mock bundle copied from `app/predictions/`).

## Findings (measured 26 Sep 2026)

Live numbers come from the private preview Space on free `cpu-basic`, driven by
Playwright from a Windows laptop: median of 6 runs after one warm up, 24 cards.

| Option | Works | Speed on live Space | Styling freedom | Gotchas |
|---|---|---|---|---|
| A1 Column visibility | yes | screen change ~160 ms; card to Player 155 ms; Back to Results 147 to 356 ms | full | Hidden screens are unmounted (values survive in the store, DOM does not) |
| A2 Tabs, bar hidden | yes | screen change ~180 ms; card to Player 223 ms; Back to Results 158 to 222 ms | full | Bar hidden via Gradio's internal `.tab-wrapper` class; all screens stay mounted |
| B1 `@gr.render` Buttons | yes | 319 to 440 ms to cards; 215 to 328 ms to Player | low: plain text, one font size | Second render pass per search; 24 listeners rebuilt each time |
| B2a `gr.Dataset` (html) | yes | 179 to 191 ms; 177 to 245 ms | high: real HTML per card | Index based, needs a parallel id list; stray "Examples" label; wrapper styles fight the grid |
| B2b `gr.Radio` | yes | 206 to 224 ms; 219 to 281 ms | low: plain text only | Arrow keys move between cards; read out as radio buttons; value must be cleared so a card can be reopened |
| B3 `gr.HTML` + `trigger` | yes | 159 to 160 ms; 155 to 223 ms | full | Must escape names; use one delegated listener on `element` |
| C1 two core Sliders | yes | n/a | medium | 74 px tall on desktop, 85 px on a 390 px phone; about 4.5 px per year of age |
| C2 `gradio_rangeslider` 0.0.8 | **no** | n/a | n/a | Requires `gradio<6`; 1.2 MB wheel; its frontend crashes in 6.28 and renders nothing |
| D CSS pitch floor | yes | no cost measured | full | Page rules must go in `head=`; Spaces SSR needs `html { background: ... !important }`; Gradio footer text over the pitch is only 2.7 to 3.8:1 |
| E big buttons + @import font | yes | font loads on the live Space | full | Gradio still fetches Source Sans Pro and IBM Plex Mono too |

All four card engines opened the exact clicked id in every run, and all four
work from the keyboard (Enter on a focused card, or Space on a Radio option).

### Other things learned

- Local preview is not faithful. Spaces serve Gradio in SSR mode, where the
  `css=` sheet is unprefixed and loads before Gradio's own sheets, so ties lose.
  Locally Gradio adds a `.contain` prefix and the same rules win. Fix used here:
  wrap each page in `gr.Column(elem_id="pvf")` and scope every rule under `#pvf`.
- `gr.Navbar(visible=False)` does not hide the multipage navbar in 6.28; CSS does.
- Browser Back leaves the app with either nav option. A 25 line `head` script
  (pushState on forward clicks, popstate clicks a hidden button) makes Back walk
  Player, Results, Search, Home, then leave. This also works inside an iframe,
  as on huggingface.co. Forward is not handled.
- Both options drop the scroll position. A client only `.then(js=...)` after
  each screen change restores it on Back and scrolls to the top going forward.
  Without it, SEARCH at the bottom of the phone Search screen opens Results mid list.
- Worst case contrast on panels over the pitch: 5.47:1 (red arrow on the lighter
  card end); everything else on panels is 5.6:1 or better.
- An occasional ~1.5 s screen change (about 1 in 12) happened on both nav options.
- The mock bundle flags no player, so the ⚠ marker is in the markup but not visible.

### Recommendations

- **A:** Column visibility, because it is as fast as Tabs, faster to the Player
  screen, and does not depend on hiding Gradio's internal tab bar classes.
- **B:** `gr.HTML` with `trigger`, because it was fastest and is the only option with
  full card styling, real buttons for the keyboard and the exact id in the event.
- **C:** Two core sliders in one row, because the only range slider component does not
  run on Gradio 6 and the pair fits a 390 px phone.
- **D:** CSS pitch via `head=`, as built here, with the Gradio footer hidden or given
  a panel, because everything else passed on the live Space.
- **E:** `size="lg"` plus scoped CSS and Barlow Condensed via `@import`, because it
  loads on the live Space; trim the import to the weights actually used.

Screenshots of all four screens at 1366 px and 390 px are in `screenshots/`.
