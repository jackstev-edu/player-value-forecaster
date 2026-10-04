# UI probes

Browser checks for the FIFA style redesign, driven by Playwright. Each one runs against a local run of the app or a Hugging Face Space, private or public. They came out of the `spike/fifa-ui` experiment, and its findings are in `spike/README.md` on that branch.

| Script | Checks | Passes when |
|---|---|---|
| `timing.py` | Home to Search, SEARCH to 24 cards, card to Player, Back to Results (medians) | Always prints times; also reports whether every click opened the clicked player id |
| `screenshots.py` | All four screens at 1366x900 and 390x844, plus whether the web font loaded | The font loaded and no screen scrolls sideways |
| `contrast.py` | Every visible text node against the pixels really behind it | Every text item reaches 4.5:1 |
| `back_button.py` | In app Back keeps scroll and search text; browser Back walks Player, Results, Search, Home, then leaves | All of those hold |
| `compare.py` | Two screenshot folders, pixel by pixel (for example local SSR against the Space) | At most 2% of pixels differ in every pair |

`contrast.py`, `back_button.py` and `compare.py` exit with code 1 on failure, so they can gate a script.

## One time setup (Windows)

Keep the probe venv at a short path. Inside the OneDrive project folder, Playwright's `greenlet` DLL fails with "The filename or extension is too long".

```powershell
python -m venv $env:USERPROFILE\.venvs\pvf-probes
& $env:USERPROFILE\.venvs\pvf-probes\Scripts\python.exe -m pip install -r scripts\ui_probes\requirements.txt
& $env:USERPROFILE\.venvs\pvf-probes\Scripts\python.exe -m playwright install chromium
```

Examples below call that interpreter `$py`:

```powershell
$py = "$env:USERPROFILE\.venvs\pvf-probes\Scripts\python.exe"
```

## Against a local run

Start the app in one terminal, then probe it from another. `--local` defaults to `http://127.0.0.1:7860`; pass a URL to change it.

Until the redesign lands in `app/`, the current app has none of the hooks below, so probe the spike instead. It lives only on its own branch, so export a copy:

```powershell
git worktree add ..\pvf-spike spike/fifa-ui
.venv\Scripts\python.exe ..\pvf-spike\spike\app.py
```

```powershell
# Terminal 1: plain local run (see docs/gui_notes.md for the SSR run that matches Spaces)
.venv\Scripts\python.exe app\app.py

# Terminal 2
& $py scripts\ui_probes\timing.py --local
& $py scripts\ui_probes\screenshots.py --local
& $py scripts\ui_probes\contrast.py --local
& $py scripts\ui_probes\back_button.py --local
```

A plain local run is not a faithful preview of styling. Spaces serve Gradio in SSR mode, which loads custom CSS differently. Use the SSR run from `docs/gui_notes.md` when checking looks, or probe the preview Space.

## Against jackstev/pvf-preview

`--space` with no value means `jackstev/pvf-preview`; give `OWNER/NAME` for another Space. Deploy first with `python scripts/deploy_space.py jackstev/pvf-preview --folder <folder> --private`.

```powershell
& $py scripts\ui_probes\timing.py --space
& $py scripts\ui_probes\screenshots.py --space
& $py scripts\ui_probes\contrast.py --space
& $py scripts\ui_probes\back_button.py --space --iframe
```

For a private Space the probes read your Hugging Face token from `HF_TOKEN`, or from the file `huggingface-cli login` writes. They exchange it for a short lived signed URL, so the token only goes to huggingface.co, never to the Space or any font host. `--iframe` repeats the Back test with the app inside an iframe, as on huggingface.co/spaces.

## Compare local SSR with the Space

```powershell
& $py scripts\ui_probes\screenshots.py --local --out scripts\ui_probes\out\local
& $py scripts\ui_probes\screenshots.py --space --out scripts\ui_probes\out\space
& $py scripts\ui_probes\compare.py scripts\ui_probes\out\local scripts\ui_probes\out\space
```

`compare.py` writes red-marked diff images to `scripts/ui_probes/out/diff/`. Everything under `out/` is git ignored.

## Options worth knowing

- `--path tabs` probes another page of a multipage app. In Git Bash leave off the leading slash, because Git Bash rewrites `/tabs` into a Windows path.
- `timing.py --reps 6 --cards 24 --toggle Midfield`: the toggle is a filter label flipped every run, so each search shows a different first card and stale cards are never timed.
- `contrast.py --allow-large` accepts 3:1 for large text, as WCAG AA does. `--show-all` lists passing text too. `--pct 0.01` judges each item against its worst 1% of background pixels.
- `back_button.py --query a` sets the text typed into the name box.
- In Git Bash, set `PYTHONIOENCODING=utf-8` so the ⚠ and € characters print.

## Probe contract

The probes find the app only through these class hooks, defined in `probe_common.py`. Keep them in the redesign, or update `probe_common.py` in the same commit.

| Hook | Where |
|---|---|
| `.pvf-screen` plus `.pvf-home`, `.pvf-search`, `.pvf-results`, `.pvf-player` | One container per screen |
| `.pvf-home button.pvf-start` (or the first `.pvf-fwd` button there) | Opens Search |
| `.pvf-search button.pvf-go` | Runs the search |
| `.pvf-search input[type=text]` | The player name box |
| `button.pvf-card[data-pid]` | One per player card, carrying the player id |
| `.pvf-player-head[data-pid]` | Player screen header, carrying the shown id |
| `.pvf-<screen> button.pvf-back` | In app Back on each screen |
| A label with the text `Midfield` on Search | Only needed by `timing.py`; change it with `--toggle` |
