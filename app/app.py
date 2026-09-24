"""Gradio front end for the player market value forecaster.

Reads only the prediction bundle in ./predictions, never the model,
so the Space stays fast and the pipeline can change independently.
"""
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

BUNDLE_DIR = Path(__file__).parent / "predictions"
MAX_ROWS = 200
AGE_FLOOR, AGE_CEILING = 15, 45

# Columns the table always shows, even when no player matches
COLUMNS = ["Player", "Age", "Position", "Club", "League", "Nationality", "Value",
           "Change", "Likely range"]

# Radio labels mapped to the forecast horizon in seasons
HORIZONS = {"1 season": 1, "2 seasons": 2, "3 seasons": 3}
DEFAULT_HORIZON = "1 season"
# Sort label to (column prefix, ascending); prefixes gain the horizon suffix
SORTS = {"Current value": ("current_value_eur", False),
         "Biggest predicted rise": ("change", False),
         "Biggest predicted fall": ("change", True),
         "Most uncertain": ("width", False)}
DEFAULT_SORT = "Current value"

CARD_PROMPT = "Select a player in the table."
CARD_GONE = "That player isn't in the current results. Select another player in the table."
CARD_MISSING = "That player could not be found in this bundle. Select another player in the table."
CARD_STALE = "That row is no longer in the table. Select another player in the table."

# Palette: pitch green, chalk lines, gold for the forecast band
PITCH, CHALK, INK, GRASS, GOLD, MUTED = "#14402F", "#F2F3EC", "#17201C", "#3E7A5B", "#D4A72C", "#6B7A72"


def load_bundle(bundle_dir: Path = BUNDLE_DIR):
    """Load the three tables and manifest written by the pipeline."""
    players = pd.read_parquet(bundle_dir / "players.parquet")
    history = pd.read_parquet(bundle_dir / "history.parquet")
    forecasts = pd.read_parquet(bundle_dir / "forecasts.parquet")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return players, history, forecasts, manifest


def build_lookups(players, history, forecasts):
    """Index every table by player_id so a click never scans a frame."""
    by_history = {int(pid): group.sort_values("date")
                  for pid, group in history.groupby("player_id", sort=False)}
    by_forecast = {int(pid): group.sort_values("horizon")
                   for pid, group in forecasts.groupby("player_id", sort=False)}
    # to_dict keeps Timestamps intact, unlike a list of numpy records
    by_player = {int(pid): row for pid, row
                 in players.set_index("player_id", drop=False).to_dict("index").items()}
    return by_history, by_forecast, by_player


# Letters that NFKD leaves whole, mapped to what people type instead
EXTRA_FOLDS = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ð": "d", "æ": "ae",
                             "œ": "oe", "ı": "i", "þ": "th"})


def normalise_name(text) -> str:
    """Lower case, accent free form so 'mbappe' matches 'Mbappé'."""
    if not isinstance(text, str):
        return ""
    # NFKD splits é into e plus a combining accent, which we drop
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(bare.translate(EXTRA_FOLDS).split())


def add_derived_columns(players, forecasts):
    """Precompute search keys and per horizon forecast numbers once at startup."""
    players = players.copy()
    players["name_key"] = players["name"].map(normalise_name)
    # One row per player, columns keyed by (quantile, horizon)
    wide = forecasts.pivot_table(index="player_id", columns="horizon",
                                 values=["p10_eur", "p50_eur", "p90_eur"], aggfunc="first")
    current = players["current_value_eur"].where(players["current_value_eur"] > 0)
    for h in HORIZONS.values():
        for q in ("p10", "p50", "p90"):
            col = (f"{q}_eur", h)
            # A horizon missing from the bundle becomes all NaN, not a KeyError
            src = wide[col] if col in wide.columns else pd.Series(dtype=float)
            players[f"{q}_{h}"] = players["player_id"].map(src).astype(float)
        median = players[f"p50_{h}"].where(players[f"p50_{h}"] > 0)
        # Zero or missing denominators give NaN, which sort_rows puts last
        players[f"change_{h}"] = players[f"p50_{h}"] / current - 1
        players[f"width_{h}"] = (players[f"p90_{h}"] - players[f"p10_{h}"]) / median
    return players


PLAYERS, HISTORY, FORECASTS, MANIFEST = load_bundle()
PLAYERS = add_derived_columns(PLAYERS, FORECASTS)
HISTORY_BY_ID, FORECASTS_BY_ID, PLAYER_BY_ID = build_lookups(PLAYERS, HISTORY, FORECASTS)


def fmt_eur(v) -> str:
    """Transfermarkt style money, e.g. 12.5m or 750k."""
    # Sparse bundles carry gaps; never let them render as nan
    if v is None or pd.isna(v):
        return "unknown"
    return f"€{v / 1e6:.1f}m" if v >= 1e6 else f"€{v / 1e3:.0f}k"


def fmt_change(v) -> str:
    """Signed whole percent, e.g. +18%, or n/a without a forecast."""
    return "n/a" if v is None or pd.isna(v) else f"{v:+.0%}"


def fmt_range(lo, hi) -> str:
    """Likely range as '€8.0m to €15.0m', or n/a when either end is missing."""
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "n/a"
    return f"{fmt_eur(lo)} to {fmt_eur(hi)}"


def horizon_of(label) -> int:
    """Radio label to seasons ahead, falling back to the default."""
    return HORIZONS.get(label, HORIZONS[DEFAULT_HORIZON])


def sort_rows(df, sort_by, h: int):
    """Order on raw numbers; no forecast for this horizon always goes last."""
    prefix, ascending = SORTS.get(sort_by, SORTS[DEFAULT_SORT])
    key = prefix if prefix == "current_value_eur" else f"{prefix}_{h}"
    # Ties fall back to higher value, then lower id, so order is deterministic
    by = ["_no_forecast", key] + [c for c in ("current_value_eur", "player_id") if c != key]
    asc = [True, ascending] + [c == "player_id" for c in by[2:]]
    return (df.assign(_no_forecast=df[f"p50_{h}"].isna())
              .sort_values(by, ascending=asc, na_position="last", kind="stable")
              .drop(columns="_no_forecast"))


def options(col: str) -> list[str]:
    return sorted(PLAYERS[col].dropna().unique().tolist())


def coerce_age(value, default: int) -> int:
    """Sliders may send None or floats, so normalise to a plain int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def no_match_message(lo: int, hi: int, active: list[str]) -> str:
    """Name the live filters so the user knows what to loosen."""
    msg = f"No players match. Try widening the age range (currently {lo} to {hi})"
    if active:
        msg += " or removing a filter: " + ", ".join(active)
    return msg + "."


def filter_players(search, leagues, countries, nations, positions, age_min, age_max,
                   horizon, sort_by, selected_id):
    """Filter the roster and keep table, ids, count and selection in step."""
    df = PLAYERS
    active = []
    query = normalise_name(search)
    if query:
        # Plain substring match on the precomputed key, never a regex
        df = df[df["name_key"].str.contains(query, regex=False)]
        active.append(f'Search "{search.strip()}"')
    # None and an empty list both mean no filter on that field
    for label, col, chosen in (("League", "league_name", leagues),
                               ("League country", "league_country", countries),
                               ("Nationality", "nationality", nations),
                               ("Position", "position", positions)):
        if chosen:
            df = df[df[col].isin(chosen)]
            active.append(label)
    lo, hi = sorted((coerce_age(age_min, AGE_FLOOR), coerce_age(age_max, AGE_CEILING)))
    df = df[df["age"].between(lo, hi)]

    # Judge the selection against everyone, not the slice on screen
    matched_ids = set(df["player_id"].tolist())
    total = len(df)
    h = horizon_of(horizon)
    sort_by = sort_by if sort_by in SORTS else DEFAULT_SORT
    df = sort_rows(df, sort_by, h).head(MAX_ROWS)

    # Formatting happens after sorting, so text never drives the order
    view = pd.DataFrame({
        "Player": df["name"], "Age": df["age"], "Position": df["sub_position"],
        "Club": df["club_name"], "League": df["league_name"], "Nationality": df["nationality"],
        "Value": df["current_value_eur"].map(fmt_eur),
        "Change": df[f"change_{h}"].map(fmt_change),
        "Likely range": [fmt_range(lo, hi) for lo, hi in zip(df[f"p10_{h}"], df[f"p90_{h}"])],
    }, columns=COLUMNS)
    if total:
        shown = (f" (showing the first {MAX_ROWS}, sorted by {sort_by.lower()})"
                 if total > MAX_ROWS else "")
        count = f"**{total} players** match these filters{shown}. Select a row to see the forecast."
    else:
        count = no_match_message(lo, hi, active)

    if selected_id is None:
        chart_out, card_out, sel_out = None, CARD_PROMPT, None
    elif int(selected_id) in matched_ids:
        # Chart stays put; the card redraws to highlight the chosen horizon
        chart_out, card_out, sel_out = gr.skip(), make_card(int(selected_id), h), selected_id
    else:
        chart_out, card_out, sel_out = None, CARD_GONE, None
    return view, df["player_id"].tolist(), count, chart_out, card_out, sel_out


def default_filters() -> list:
    """Starting value for every filter, in the same order as the inputs."""
    return ["", [], [], [], [], AGE_FLOOR, AGE_CEILING, DEFAULT_HORIZON, DEFAULT_SORT]


def reset_filters(selected_id):
    """Clear every filter and refresh the results in one single run."""
    defaults = default_filters()
    return (*defaults, *filter_players(*defaults, selected_id))


def history_points(pid: int, player: dict):
    """Value history as x and y lists, or the current value alone."""
    h = HISTORY_BY_ID.get(pid)
    if h is None or h.empty:
        # Players with no history still anchor on the players table row
        return [player["value_date"]], [player["current_value_eur"]]
    return h["date"].tolist(), h["value_eur"].tolist()


def make_chart(pid: int):
    """Value history line plus 1 to 3 season forecast with a 10 to 90% band."""
    player = PLAYER_BY_ID.get(pid)
    if player is None:
        return None
    hx, hy = history_points(pid, player)
    f = FORECASTS_BY_ID.get(pid)
    has_forecast = f is not None and not f.empty
    last_date, last_val = hx[-1], hy[-1]
    m = 1e6

    fig = go.Figure()
    if has_forecast:
        # Forecast traces start at the last known value so lines connect
        fx = [last_date] + f["target_date"].tolist()
        lo = [last_val] + f["p10_eur"].tolist()
        mid = [last_val] + f["p50_eur"].tolist()
        hi = [last_val] + f["p90_eur"].tolist()
        fig.add_trace(go.Scatter(x=fx + fx[::-1], y=[v / m for v in hi + lo[::-1]], fill="toself",
                                 mode="lines", fillcolor="rgba(212,167,44,0.22)", line=dict(width=0),
                                 hoverinfo="skip", name="10 to 90% range"))
    fig.add_trace(go.Scatter(x=hx, y=[v / m for v in hy], mode="lines+markers",
                             line=dict(color=GRASS, width=2.5, shape="hv"), marker=dict(size=5),
                             name="Market value", hovertemplate="%{x|%b %Y}<br>€%{y:.1f}m<extra></extra>"))
    if has_forecast:
        fig.add_trace(go.Scatter(x=fx, y=[v / m for v in mid], mode="lines+markers",
                                 line=dict(color=GOLD, width=2.5, dash="dash"), marker=dict(size=7),
                                 name="Median forecast",
                                 hovertemplate="%{x|%b %Y}<br>€%{y:.1f}m<extra></extra>"))
        # Dotted rule marks where observed history stops and forecast starts
        if pd.notna(last_date):
            fig.add_vline(x=last_date, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", size=13),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Value (€m)", gridcolor="rgba(107,122,114,0.25)", rangemode="tozero"),
        xaxis=dict(gridcolor="rgba(107,122,114,0.12)"),
    )
    return fig


def make_card(pid: int, horizon: int = 1) -> str:
    """Short markdown summary of the selected player and forecast numbers."""
    player = PLAYER_BY_ID.get(pid)
    if player is None:
        return CARD_MISSING
    h = HISTORY_BY_ID.get(pid)
    f = FORECASTS_BY_ID.get(pid)
    value_date = player["value_date"]
    updated = f"{value_date:%b %Y}" if pd.notna(value_date) else "date unknown"
    parts = [f"### {player['name']}\n"
             f"{player['sub_position']}, age {player['age']}, {player['club_name']} "
             f"({player['league_name']})\n\n"
             f"Value now: **{fmt_eur(player['current_value_eur'])}** (updated {updated})"]
    if f is not None and not f.empty:
        lines = []
        for r in f.itertuples():
            cells = [f"{r.target_date:%b %Y}", fmt_eur(r.p50_eur), fmt_range(r.p10_eur, r.p90_eur)]
            # Bold the look ahead row and mark it so it reads at a glance
            if r.horizon == horizon:
                cells = [f"**{c}**" for c in cells]
                cells[0] = f"▸ {cells[0]}"
            lines.append("| " + " | ".join(cells) + " |")
        rows = "\n".join(lines)
        parts.append(f"| Season | Median | Likely range |\n|---|---|---|\n{rows}")
    else:
        parts.append("No forecast is available for this player.")
    if h is None or h.empty:
        parts.append("No value history is available for this player.")
    return "\n\n".join(parts)


def make_footer(manifest: dict) -> str:
    """Credits, coursework notice and bundle stamp shown under everything."""
    version = manifest.get("model_version") or "unknown"
    try:
        # Manifest stamps are ISO 8601; render them as a short readable date
        built = datetime.fromisoformat(manifest["created_at"]).strftime("%d %b %Y").lstrip("0")
    except (KeyError, TypeError, ValueError):
        built = "unknown"
    kaggle = "https://www.kaggle.com/datasets"
    # Credits open in a new tab so filter state survives
    return (
        '<p>Data: Transfermarkt, via the Kaggle datasets '
        f'<a href="{kaggle}/davidcariboo/player-scores" target="_blank" rel="noopener">player-scores</a>'
        ' by davidcariboo and '
        f'<a href="{kaggle}/xfkzujqjvx97n/football-datasets" target="_blank" rel="noopener">football-datasets</a>'
        ' by salimt.</p>'
        '<p>Coursework for CMU 24-679 (Yunus Polatoglu &amp; Jack Stevens). '
        'Not for transfer or betting decisions.</p>'
        f'<p>Model {version}, built {built}.</p>'
    )


def on_select(ids: list[int], horizon: str, evt: gr.SelectData):
    """Row click handler; evt.index is [row, col] in the visible table."""
    row = evt.index[0] if evt is not None and evt.index else None
    # A click can land after the table shrank underneath the user
    if not ids or row is None or not 0 <= row < len(ids):
        return None, CARD_STALE, None
    pid = int(ids[row])
    return make_chart(pid), make_card(pid, horizon_of(horizon)), pid


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
).set(body_background_fill=CHALK, block_background_fill="#FFFFFF",
      button_primary_background_fill=PITCH, button_primary_text_color=CHALK)
CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="Player value forecaster") as demo:
    ids_state = gr.State([])
    selected_state = gr.State(None)

    gr.HTML(f"""
      <header class="pvf-head">
        <h1>Player value forecaster</h1>
        <p>Where a player's Transfermarkt value is heading over the next three seasons,
        based on age, form and the league and club around them.</p>
      </header>""")
    if MANIFEST.get("is_mock"):
        gr.HTML('<div class="pvf-mock">Demo data: these players and forecasts are invented '
                'while the models are being trained.</div>')

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260, elem_classes="pvf-filters"):
            search = gr.Textbox(label="Search player", placeholder="Type part of a name",
                                max_lines=1)
            league = gr.Dropdown(options("league_name"), multiselect=True, label="League")
            country = gr.Dropdown(options("league_country"), multiselect=True, label="League country")
            nation = gr.Dropdown(options("nationality"), multiselect=True, label="Nationality")
            position = gr.CheckboxGroup(options("position"), label="Position")
            age_min = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_FLOOR, step=1, label="Youngest age")
            age_max = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_CEILING, step=1, label="Oldest age")
            reset = gr.Button("Clear filters", variant="secondary")

        with gr.Column(scale=3):
            with gr.Row(equal_height=True, elem_classes="pvf-view"):
                horizon = gr.Radio(list(HORIZONS), value=DEFAULT_HORIZON, label="Look ahead")
                sort_by = gr.Dropdown(list(SORTS), value=DEFAULT_SORT, label="Sort by")
            count = gr.Markdown()
            table = gr.Dataframe(interactive=False, max_height=340, wrap=True, elem_classes="pvf-table",
                                 column_widths=["13%", "5%", "12%", "14%", "11%", "12%", "8%",
                                                "9%", "16%"])
            with gr.Row(equal_height=False):
                chart = gr.Plot(show_label=False, scale=3)
                with gr.Column(scale=2, min_width=320):
                    card = gr.Markdown(CARD_PROMPT, elem_classes="pvf-card")

    with gr.Accordion("How the forecast works", open=False):
        gr.Markdown(
            "Each forecast predicts the change in value 1, 2 and 3 seasons after the "
            "1 July snapshot. The shaded band is the range the model expects the value to land "
            "in 8 times out of 10. Models were backtested on later seasons than they were "
            "trained on.")

    # Footer sits outside every container so it stays visible at all times
    gr.HTML(make_footer(MANIFEST), elem_classes="pvf-foot")

    filters = [search, league, country, nation, position, age_min, age_max, horizon, sort_by]
    results = [table, ids_state, count, chart, card, selected_state]
    # .input fires on user edits only, so code-set values never refilter
    gr.on(triggers=[comp.input for comp in filters] + [demo.load],
          fn=filter_players, inputs=filters + [selected_state], outputs=results,
          trigger_mode="always_last", api_name="filter_players")
    table.select(on_select, [ids_state, horizon], [chart, card, selected_state],
                 api_name="select_player")
    # One run sets every filter and its results together, no cascade
    reset.click(reset_filters, selected_state, filters + results, api_name="reset_filters")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
