"""Gradio front end for the player market value forecaster.

Reads only the prediction bundle in ./predictions, never the model,
so the Space stays fast and the pipeline can change independently.
"""
import json
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

BUNDLE_DIR = Path(__file__).parent / "predictions"
MAX_ROWS = 200
AGE_FLOOR, AGE_CEILING = 15, 45

# Columns the table always shows, even when no player matches
COLUMNS = ["Player", "Age", "Position", "Club", "League", "Nationality", "Value"]

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


PLAYERS, HISTORY, FORECASTS, MANIFEST = load_bundle()
HISTORY_BY_ID, FORECASTS_BY_ID, PLAYER_BY_ID = build_lookups(PLAYERS, HISTORY, FORECASTS)


def fmt_eur(v) -> str:
    """Transfermarkt style money, e.g. 12.5m or 750k."""
    # Sparse bundles carry gaps; never let them render as nan
    if v is None or pd.isna(v):
        return "unknown"
    return f"€{v / 1e6:.1f}m" if v >= 1e6 else f"€{v / 1e3:.0f}k"


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


def filter_players(leagues, countries, nations, positions, age_min, age_max, selected_id):
    """Filter the roster and keep table, ids, count and selection in step."""
    df = PLAYERS
    active = []
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
    df = df.sort_values("current_value_eur", ascending=False).head(MAX_ROWS)

    view = pd.DataFrame({
        "Player": df["name"], "Age": df["age"], "Position": df["sub_position"],
        "Club": df["club_name"], "League": df["league_name"], "Nationality": df["nationality"],
        "Value": df["current_value_eur"].map(fmt_eur),
    }, columns=COLUMNS)
    if total:
        shown = f" (showing the top {MAX_ROWS} by value)" if total > MAX_ROWS else ""
        count = f"**{total} players** match these filters{shown}. Select a row to see the forecast."
    else:
        count = no_match_message(lo, hi, active)

    if selected_id is None:
        chart_out, card_out, sel_out = None, CARD_PROMPT, None
    elif int(selected_id) in matched_ids:
        # Selection survived the new filters, so leave its panels alone
        chart_out, card_out, sel_out = gr.skip(), gr.skip(), selected_id
    else:
        chart_out, card_out, sel_out = None, CARD_GONE, None
    return view, df["player_id"].tolist(), count, chart_out, card_out, sel_out


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


def make_card(pid: int) -> str:
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
        rows = "\n".join(
            f"| {r.target_date:%b %Y} | {fmt_eur(r.p50_eur)} | {fmt_eur(r.p10_eur)} to {fmt_eur(r.p90_eur)} |"
            for r in f.itertuples())
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


def on_select(ids: list[int], evt: gr.SelectData):
    """Row click handler; evt.index is [row, col] in the visible table."""
    row = evt.index[0] if evt is not None and evt.index else None
    # A click can land after the table shrank underneath the user
    if not ids or row is None or not 0 <= row < len(ids):
        return None, CARD_STALE, None
    pid = int(ids[row])
    return make_chart(pid), make_card(pid), pid


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
            league = gr.Dropdown(options("league_name"), multiselect=True, label="League")
            country = gr.Dropdown(options("league_country"), multiselect=True, label="League country")
            nation = gr.Dropdown(options("nationality"), multiselect=True, label="Nationality")
            position = gr.CheckboxGroup(options("position"), label="Position")
            age_min = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_FLOOR, step=1, label="Youngest age")
            age_max = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_CEILING, step=1, label="Oldest age")
            reset = gr.Button("Clear filters", variant="secondary")

        with gr.Column(scale=3):
            count = gr.Markdown()
            table = gr.Dataframe(interactive=False, max_height=340, wrap=True, elem_classes="pvf-table",
                                 column_widths=["17%", "7%", "16%", "20%", "14%", "14%", "12%"])
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

    filters = [league, country, nation, position, age_min, age_max]
    # Explicit triggers keep selected_state an input without retriggering here
    gr.on(triggers=[comp.change for comp in filters] + [demo.load],
          fn=filter_players, inputs=filters + [selected_state],
          outputs=[table, ids_state, count, chart, card, selected_state],
          trigger_mode="always_last", api_name="filter_players")
    table.select(on_select, ids_state, [chart, card, selected_state], api_name="select_player")
    # Clearing six filters fires six changes; always_last collapses them
    reset.click(lambda: (None, None, None, None, AGE_FLOOR, AGE_CEILING), None, filters,
                api_name="reset_filters")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
