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
