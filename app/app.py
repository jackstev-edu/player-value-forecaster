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

# Palette: pitch green, chalk lines, gold for the forecast band
PITCH, CHALK, INK, GRASS, GOLD, MUTED = "#14402F", "#F2F3EC", "#17201C", "#3E7A5B", "#D4A72C", "#6B7A72"


def load_bundle(bundle_dir: Path = BUNDLE_DIR):
    """Load the three tables and manifest written by the pipeline."""
    players = pd.read_parquet(bundle_dir / "players.parquet")
    history = pd.read_parquet(bundle_dir / "history.parquet")
    forecasts = pd.read_parquet(bundle_dir / "forecasts.parquet")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return players, history, forecasts, manifest


PLAYERS, HISTORY, FORECASTS, MANIFEST = load_bundle()


def fmt_eur(v: float) -> str:
    """Transfermarkt style money, e.g. 12.5m or 750k."""
    return f"€{v / 1e6:.1f}m" if v >= 1e6 else f"€{v / 1e3:.0f}k"


def options(col: str) -> list[str]:
    return sorted(PLAYERS[col].dropna().unique().tolist())


def filter_players(leagues, countries, nations, positions, age_min, age_max):
    """Apply sidebar filters and return the table, id order and a count line."""
    df = PLAYERS
    # Empty selection means no filter on that field
    for col, chosen in [("league_name", leagues), ("league_country", countries),
                        ("nationality", nations), ("position", positions)]:
        if chosen:
            df = df[df[col].isin(chosen)]
    lo, hi = sorted((age_min, age_max))
    df = df[df["age"].between(lo, hi)]
    total = len(df)
    df = df.sort_values("current_value_eur", ascending=False).head(MAX_ROWS)

    view = pd.DataFrame({
        "Player": df["name"], "Age": df["age"], "Position": df["sub_position"],
        "Club": df["club_name"], "League": df["league_name"], "Nationality": df["nationality"],
        "Value": df["current_value_eur"].map(fmt_eur),
    })
    shown = f" (showing the top {MAX_ROWS} by value)" if total > MAX_ROWS else ""
    count = f"**{total} players** match these filters{shown}. Select a row to see the forecast."
    return view, df["player_id"].tolist(), count


def make_chart(pid: int) -> go.Figure:
    """Value history line plus 1 to 3 season forecast with a 10 to 90% band."""
    h = HISTORY[HISTORY["player_id"] == pid].sort_values("date")
    f = FORECASTS[FORECASTS["player_id"] == pid].sort_values("horizon")
    last_date, last_val = h["date"].iloc[-1], h["value_eur"].iloc[-1]

    # Forecast traces start at the last known value so lines connect
    fx = [last_date] + f["target_date"].tolist()
    lo = [last_val] + f["p10_eur"].tolist()
    mid = [last_val] + f["p50_eur"].tolist()
    hi = [last_val] + f["p90_eur"].tolist()
    m = 1e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=[v / m for v in hi + lo[::-1]], fill="toself", mode="lines",
                             fillcolor="rgba(212,167,44,0.22)", line=dict(width=0),
                             hoverinfo="skip", name="10 to 90% range"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["value_eur"] / m, mode="lines+markers",
                             line=dict(color=GRASS, width=2.5, shape="hv"), marker=dict(size=5),
                             name="Market value", hovertemplate="%{x|%b %Y}<br>€%{y:.1f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=fx, y=[v / m for v in mid], mode="lines+markers",
                             line=dict(color=GOLD, width=2.5, dash="dash"), marker=dict(size=7),
                             name="Median forecast", hovertemplate="%{x|%b %Y}<br>€%{y:.1f}m<extra></extra>"))
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
    p = PLAYERS.loc[PLAYERS["player_id"] == pid].iloc[0]
    f = FORECASTS[FORECASTS["player_id"] == pid].sort_values("horizon")
    rows = "\n".join(
        f"| {r.target_date:%b %Y} | {fmt_eur(r.p50_eur)} | {fmt_eur(r.p10_eur)} to {fmt_eur(r.p90_eur)} |"
        for r in f.itertuples())
    return (f"### {p['name']}\n"
            f"{p['sub_position']}, age {p['age']}, {p['club_name']} ({p['league_name']})\n\n"
            f"Value now: **{fmt_eur(p['current_value_eur'])}** (updated {p['value_date']:%b %Y})\n\n"
            f"| Season | Median | Likely range |\n|---|---|---|\n{rows}")


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
    pid = ids[evt.index[0]]
    return make_chart(pid), make_card(pid)


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
).set(body_background_fill=CHALK, block_background_fill="#FFFFFF",
      button_primary_background_fill=PITCH, button_primary_text_color=CHALK)
CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="Player value forecaster") as demo:
    ids_state = gr.State([])

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
            age_min = gr.Slider(15, 45, value=15, step=1, label="Youngest age")
            age_max = gr.Slider(15, 45, value=45, step=1, label="Oldest age")
            reset = gr.Button("Clear filters", variant="secondary")

        with gr.Column(scale=3):
            count = gr.Markdown()
            table = gr.Dataframe(interactive=False, max_height=340, wrap=True, elem_classes="pvf-table",
                                 column_widths=["17%", "7%", "16%", "20%", "14%", "14%", "12%"])
            with gr.Row(equal_height=False):
                chart = gr.Plot(show_label=False, scale=3)
                with gr.Column(scale=2, min_width=320):
                    card = gr.Markdown("Select a player in the table.", elem_classes="pvf-card")

    with gr.Accordion("How the forecast works", open=False):
        gr.Markdown(
            "Each forecast predicts the change in value 1, 2 and 3 seasons after the "
            "1 July snapshot. The shaded band is the range the model expects the value to land "
            "in 8 times out of 10. Models were backtested on later seasons than they were "
            "trained on.")

    # Footer sits outside every container so it stays visible at all times
    gr.HTML(make_footer(MANIFEST), elem_classes="pvf-foot")

    filters = [league, country, nation, position, age_min, age_max]
    for comp in filters:
        comp.change(filter_players, filters, [table, ids_state, count])
    table.select(on_select, ids_state, [chart, card])
    reset.click(lambda: (None, None, None, None, 15, 45), None, filters)
    demo.load(filter_players, filters, [table, ids_state, count])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
