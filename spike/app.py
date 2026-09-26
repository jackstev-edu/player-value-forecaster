"""SPIKE, not the product: FIFA style screens on Gradio 6.28.

Throwaway experiment for branch spike/fifa-ui. It tests screen switching,
four ways to make clickable player cards, a CSS pitch floor and a condensed
display font. Forecast logic is copied from app/app.py so this folder
deploys on its own; nothing in app/ or src/ imports it.

Pages: "/" switches screens with Column visibility, "/tabs" with hidden Tabs.
"""
import contextlib
import html
import json
import math
import unicodedata
from functools import partial
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

try:
    # Only installed for the local range slider test, never on the Space
    from gradio_rangeslider import RangeSlider
except Exception:  # noqa: BLE001, a broken install should not stop the spike
    RangeSlider = None

HERE = Path(__file__).parent
BUNDLE_DIR = HERE / "predictions"
CARD_COUNT = 24
AGE_FLOOR, AGE_CEILING = 15, 45

HORIZONS = {"1 season": 1, "2 seasons": 2, "3 seasons": 3}
DEFAULT_HORIZON = "1 season"
SORTS = {"Current value": ("current_value_eur", False),
         "Biggest predicted rise": ("change", False),
         "Biggest predicted fall": ("change", True),
         "Most uncertain": ("width", False)}
DEFAULT_SORT = "Current value"
LOW_CONFIDENCE_SHARE = 0.2
MIN_HISTORY_ROWS = 3
FLAG = "⚠ "

# Card engines under test; the label is what the Search screen shows
ENGINES = {"HTML + trigger": "html", "Dataset": "dataset", "Radio": "radio",
           "gr.render buttons": "render"}
DEFAULT_ENGINE = "HTML + trigger"

SCREENS = ["home", "search", "results", "player"]
# Linear flow, so Back always means the screen before this one
BACK_TO = {"home": "home", "search": "home", "results": "search", "player": "results"}

# Night palette: near black pitch, dark panels, chalk text, gold accent
BG, PANEL, PANEL_2, LINE = "#05100A", "#0C1D15", "#12291E", "#2C4D3D"
TEXT, TEXT_DIM, GOLD, UP, DOWN = "#F2F3EC", "#B5C2BA", "#D4A72C", "#6FD69A", "#FF8A7A"


# Copied from app/app.py: bundle loading, search keys and derived columns

def load_bundle(bundle_dir: Path = BUNDLE_DIR):
    players = pd.read_parquet(bundle_dir / "players.parquet")
    history = pd.read_parquet(bundle_dir / "history.parquet")
    forecasts = pd.read_parquet(bundle_dir / "forecasts.parquet")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return players, history, forecasts, manifest


def build_lookups(players, history, forecasts):
    by_history = {int(pid): group.sort_values("date")
                  for pid, group in history.groupby("player_id", sort=False)}
    by_forecast = {int(pid): group.sort_values("horizon")
                   for pid, group in forecasts.groupby("player_id", sort=False)}
    by_player = {int(pid): row for pid, row
                 in players.set_index("player_id", drop=False).to_dict("index").items()}
    return by_history, by_forecast, by_player


EXTRA_FOLDS = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ð": "d", "æ": "ae",
                             "œ": "oe", "ı": "i", "þ": "th"})


def normalise_name(text) -> str:
    if not isinstance(text, str):
        return ""
    # NFKD splits é into e plus a combining accent, which we drop
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(bare.translate(EXTRA_FOLDS).split())


def add_derived_columns(players, forecasts):
    players = players.copy()
    players["name_key"] = players["name"].map(normalise_name)
    wide = forecasts.pivot_table(index="player_id", columns="horizon",
                                 values=["p10_eur", "p50_eur", "p90_eur"], aggfunc="first")
    current = players["current_value_eur"].where(players["current_value_eur"] > 0)
    for h in HORIZONS.values():
        for q in ("p10", "p50", "p90"):
            col = (f"{q}_eur", h)
            src = wide[col] if col in wide.columns else pd.Series(dtype=float)
            players[f"{q}_{h}"] = players["player_id"].map(src).astype(float)
        median = players[f"p50_{h}"].where(players[f"p50_{h}"] > 0)
        players[f"change_{h}"] = players[f"p50_{h}"] / current - 1
        players[f"width_{h}"] = (players[f"p90_{h}"] - players[f"p10_{h}"]) / median
    return players


PLAYERS, HISTORY, FORECASTS, MANIFEST = load_bundle()
PLAYERS = add_derived_columns(PLAYERS, FORECASTS)
HISTORY_BY_ID, FORECASTS_BY_ID, PLAYER_BY_ID = build_lookups(PLAYERS, HISTORY, FORECASTS)
WIDTH_CUTOFF = {h: PLAYERS[f"width_{h}"].quantile(1 - LOW_CONFIDENCE_SHARE)
                for h in HORIZONS.values()}


def fmt_eur(v) -> str:
    if v is None or pd.isna(v):
        return "unknown"
    return f"€{v / 1e6:.1f}m" if v >= 1e6 else f"€{v / 1e3:.0f}k"


def fmt_change(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:+.0%}"


def fmt_range(lo, hi) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "n/a"
    return f"{fmt_eur(lo)} to {fmt_eur(hi)}"


def horizon_of(label) -> int:
    return HORIZONS.get(label, HORIZONS[DEFAULT_HORIZON])


def seasons_text(h: int) -> str:
    return "1 season" if h == 1 else f"{h} seasons"


def history_count(pid: int) -> int:
    h = HISTORY_BY_ID.get(pid)
    return 0 if h is None else len(h)


def low_confidence_reasons(width, n_history: int, h: int) -> list[str]:
    if width is None or pd.isna(width):
        return []
    reasons = []
    cutoff = WIDTH_CUTOFF.get(h)
    # Float noise must not split ties; the mock gives every player one width
    if (cutoff is not None and pd.notna(cutoff) and width > cutoff
            and not math.isclose(width, cutoff, rel_tol=1e-9)):
        reasons.append("width")
    if n_history < MIN_HISTORY_ROWS:
        reasons.append("history")
    return reasons


def sort_rows(df, sort_by, h: int):
    prefix, ascending = SORTS.get(sort_by, SORTS[DEFAULT_SORT])
    key = prefix if prefix == "current_value_eur" else f"{prefix}_{h}"
    by = ["_no_forecast", key] + [c for c in ("current_value_eur", "player_id") if c != key]
    asc = [True, ascending] + [c == "player_id" for c in by[2:]]
    return (df.assign(_no_forecast=df[f"p50_{h}"].isna())
              .sort_values(by, ascending=asc, na_position="last", kind="stable")
              .drop(columns="_no_forecast"))


def options(col: str) -> list[str]:
    return sorted(PLAYERS[col].dropna().unique().tolist())


def coerce_age(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Copied from app/app.py: chart, card and explanation, recoloured for dark panels

def money_label(v: float) -> str:
    if v == 0:
        return "€0"
    return f"€{v / 1e6:g}m" if v >= 1e6 else f"€{v / 1e3:g}k"


def money_ticks(top: float, target: int = 5):
    if top is None or pd.isna(top) or top <= 0:
        top = 1e6
    rough = top / target
    magnitude = 10 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rough)
    values = [i * step for i in range(1, math.ceil(top / step) + 1)]
    return values, [money_label(v) for v in values]


def make_chart(pid: int):
    player = PLAYER_BY_ID.get(pid)
    if player is None:
        return None
    h = HISTORY_BY_ID.get(pid)
    if h is None or h.empty:
        hx, hy = [player["value_date"]], [player["current_value_eur"]]
    else:
        hx, hy = h["date"].tolist(), h["value_eur"].tolist()
    f = FORECASTS_BY_ID.get(pid)
    has_forecast = f is not None and not f.empty
    last_date, last_val = hx[-1], hy[-1]
    fig = go.Figure()
    top = max((v for v in hy if pd.notna(v)), default=0)
    if has_forecast:
        tx = f["target_date"].tolist()
        lo, mid, hi = f["p10_eur"].tolist(), f["p50_eur"].tolist(), f["p90_eur"].tolist()
        top = max([top] + [v for v in hi if pd.notna(v)])
        bx = [last_date] + tx
        fig.add_trace(go.Scatter(x=bx + bx[::-1], y=[last_val] + hi + lo[::-1] + [last_val],
                                 fill="toself", mode="lines", fillcolor="rgba(212,167,44,0.25)",
                                 line=dict(width=0), hoverinfo="skip", name="Likely range"))
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", name="Market value",
                             line=dict(color=UP, width=2.5, shape="hv"), marker=dict(size=5)))
    if has_forecast:
        fig.add_trace(go.Scatter(x=[last_date] + tx, y=[last_val] + mid, mode="lines+markers",
                                 name="Median forecast",
                                 line=dict(color=GOLD, width=2.5, dash="dash")))
    ticks, labels = money_ticks(top)
    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=13, color=TEXT),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=LINE, font=dict(color=TEXT)),
        yaxis=dict(tickvals=ticks, ticktext=labels, range=[0, ticks[-1] * 1.02],
                   gridcolor="rgba(181,194,186,0.18)", zeroline=False),
        xaxis=dict(hoverformat="%b %Y", gridcolor="rgba(181,194,186,0.08)", zeroline=False),
    )
    return fig


def make_card(pid: int, horizon: int = 1) -> str:
    f = FORECASTS_BY_ID.get(pid)
    if f is None or f.empty:
        return "No forecast is available for this player."
    lines = []
    for r in f.itertuples():
        cells = [f"{r.target_date:%b %Y}", fmt_eur(r.p50_eur), fmt_range(r.p10_eur, r.p90_eur)]
        # Bold the chosen look ahead row so it reads at a glance
        if r.horizon == horizon:
            cells = [f"**{c}**" for c in cells]
            cells[0] = f"▸ {cells[0]}"
        lines.append("| " + " | ".join(cells) + " |")
    return "| Season | Median | Likely range |\n|---|---|---|\n" + "\n".join(lines)


def make_explanation(pid: int, horizon: int = 1) -> str:
    player = PLAYER_BY_ID.get(pid)
    f = FORECASTS_BY_ID.get(pid)
    rows = {} if f is None else {int(r.horizon): r for r in f.itertuples()}
    r = rows.get(horizon)
    if player is None or r is None or pd.isna(r.p50_eur):
        return "No forecast is available for this player yet."
    return (f"In {seasons_text(horizon)}, the model's middle estimate is {fmt_eur(r.p50_eur)}. "
            f"Its likely range is {fmt_range(r.p10_eur, r.p90_eur)}. The model aims for the "
            "real value to land inside this range 8 times out of 10.")


# Search and card data, the spike's own code

def find_players(search, leagues, positions, age_min, age_max, horizon, sort_by):
    """Filter like app.filter_players, but return raw rows for the cards."""
    df = PLAYERS
    query = normalise_name(search)
    if query:
        df = df[df["name_key"].str.contains(query, regex=False)]
    if leagues:
        df = df[df["league_name"].isin(leagues)]
    if positions:
        df = df[df["position"].isin(positions)]
    lo, hi = sorted((coerce_age(age_min, AGE_FLOOR), coerce_age(age_max, AGE_CEILING)))
    df = df[df["age"].between(lo, hi)]
    h = horizon_of(horizon)
    total = len(df)
    df = sort_rows(df, sort_by if sort_by in SORTS else DEFAULT_SORT, h).head(CARD_COUNT)
    return df, total, h


def card_data(row, h: int) -> dict:
    """Everything one card shows, already formatted as text."""
    pid = int(row["player_id"])
    change = row[f"change_{h}"]
    trend = "flat" if pd.isna(change) or round(change, 2) == 0 else ("up" if change > 0 else "down")
    flagged = bool(low_confidence_reasons(row[f"width_{h}"], history_count(pid), h))
    return {"pid": pid, "name": row["name"], "pos": row["sub_position"], "age": int(row["age"]),
            "club": row["club_name"], "value": fmt_eur(row["current_value_eur"]),
            "arrow": {"up": "▲", "down": "▼", "flat": "■"}[trend], "trend": trend,
            "change": fmt_change(change), "range": fmt_range(row[f"p10_{h}"], row[f"p90_{h}"]),
            "flag": flagged}


def card_inner(c: dict) -> str:
    """Card body as spans, so it is valid inside a button or a div."""
    e = html.escape
    flag = '<span class="pc-flag" title="Low confidence forecast">⚠</span>' if c["flag"] else ""
    return (f'<span class="pc-top"><span class="pc-pos">{e(c["pos"])}</span>{flag}</span>'
            f'<span class="pc-name">{e(c["name"])}</span>'
            f'<span class="pc-meta">Age {c["age"]} · {e(c["club"])}</span>'
            f'<span class="pc-value">{c["value"]}</span>'
            f'<span class="pc-change pc-{c["trend"]}">{c["arrow"]} {c["change"]}</span>'
            f'<span class="pc-range">Likely {c["range"]}</span>')


def cards_html(cards: list[dict]) -> str:
    """Option B3: real buttons carry data-pid, so Enter and Space click them."""
    if not cards:
        return '<p class="pvf-empty">No players match. Go back and widen the filters.</p>'
    body = "".join(f'<button type="button" class="pvf-card" data-pid="{c["pid"]}" '
                   f'aria-label="Open {html.escape(c["name"])}">{card_inner(c)}</button>'
                   for c in cards)
    return f'<div class="pvf-grid">{body}</div>'


def card_text(c: dict) -> str:
    """Plain multi line label for Radio and Button, which cannot hold markup."""
    flag = FLAG if c["flag"] else ""
    return (f"{flag}{c['name']}\n{c['pos']} · {c['age']} · {c['club']}\n"
            f"{c['value']}   {c['arrow']} {c['change']}\n{c['range']}")


def player_head(pid: int, h: int) -> str:
    """Big name block on the Player screen; data-pid lets tests check the id."""
    p = PLAYER_BY_ID[pid]
    c = card_data(PLAYERS.loc[PLAYERS["player_id"] == pid].iloc[0], h)
    flag = '<span class="pc-flag">⚠ Low confidence</span>' if c["flag"] else ""
    e = html.escape
    return (f'<header class="pvf-player-head" data-pid="{pid}">'
            f'<span class="pc-pos">{e(c["pos"])}</span>{flag}'
            f'<h2>{e(p["name"])}</h2>'
            f'<p>Age {c["age"]} · {e(c["club"])} · {e(p["league_name"])}</p>'
            f'<p class="pvf-big">{c["value"]} <span class="pc-change pc-{c["trend"]}">'
            f'{c["arrow"]} {c["change"]} in {seasons_text(h)}</span></p></header>')


# Scoped JS for option B3: one delegated listener survives every re-render
CARD_JS = """
element.addEventListener('click', (e) => {
  const card = e.target.closest('[data-pid]');
  if (card) trigger('click', {pid: Number(card.dataset.pid)});
});
"""


class ColumnNav:
    """Option A1: one Column per screen, switched with visible updates."""
    def __init__(self):
        self.cols = {}

    def container(self):
        # Screens sit directly in the page, so no wrapper is needed
        return contextlib.nullcontext()

    def screen(self, name):
        self.cols[name] = gr.Column(visible=name == "home", elem_classes=["pvf-screen", f"pvf-{name}"])
        return self.cols[name]

    @property
    def outputs(self):
        return [self.cols[n] for n in SCREENS]

    def to(self, name):
        return [gr.Column(visible=n == name) for n in SCREENS]


class TabsNav:
    """Option A2: one Tab per screen, tab bar hidden, selected set in Python."""
    def __init__(self):
        self.tabs = None

    def container(self):
        self.tabs = gr.Tabs(selected="home", elem_classes="pvf-tabs")
        return self.tabs

    def screen(self, name):
        return gr.Tab(name.title(), id=name, elem_classes=["pvf-screen", f"pvf-{name}"])

    @property
    def outputs(self):
        return [self.tabs]

    def to(self, name):
        return [gr.Tabs(selected=name)]


def build(nav):
    """Four screens plus every card engine, wired to one navigation style."""
    # Documented way to hide the multipage navbar; 6.28 ignores it, CSS hides it
    gr.Navbar(visible=False)
    screen = gr.State("home")
    ids = gr.State([])
    render_cards = gr.State([])

    # One id to scope every CSS rule; an id beats Gradio's class rules anywhere
    with gr.Column(elem_id="pvf"):
        with nav.container():
            with nav.screen("home"):
                gr.HTML('<header class="pvf-hero"><p class="pvf-kicker">Transfermarkt forecasts'
                        '</p><h1>Player value<br>forecaster</h1><p>Where is a player\'s market '
                        'value heading over the next three seasons?</p></header>')
                if MANIFEST.get("is_mock"):
                    gr.HTML('<p class="pvf-mock">Demo data: invented players while the '
                            'models train.</p>')
                with gr.Row(elem_classes="pvf-actions"):
                    start = gr.Button("Search players", variant="primary", size="lg",
                                      elem_classes=["pvf-big-btn", "pvf-fwd"])
                    top = gr.Button("Top 24 now", variant="secondary", size="lg",
                                    elem_classes=["pvf-big-btn", "pvf-fwd"])

            with nav.screen("search"):
                gr.HTML('<h2 class="pvf-h2">Search</h2>')
                search = gr.Textbox(label="Player name", placeholder="Type part of a name",
                                    max_lines=1)
                position = gr.CheckboxGroup(options("position"), label="Position")
                league = gr.Dropdown(options("league_name"), multiselect=True, label="League")
                # Option C1: two core sliders sharing one row as a compact range
                with gr.Row(elem_classes="pvf-age"):
                    age_min = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_FLOOR, step=1,
                                        label="Age from", min_width=120)
                    age_max = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_CEILING, step=1,
                                        label="Age to", min_width=120)
                if RangeSlider is not None:
                    # Option C2, local only: the custom component, for comparison
                    RangeSlider(minimum=AGE_FLOOR, maximum=AGE_CEILING,
                                value=(AGE_FLOOR, AGE_CEILING), label="Age (gradio_rangeslider)")
                with gr.Row():
                    horizon = gr.Radio(list(HORIZONS), value=DEFAULT_HORIZON, label="Look ahead")
                    sort_by = gr.Dropdown(list(SORTS), value=DEFAULT_SORT, label="Sort by")
                engine = gr.Radio(list(ENGINES), value=DEFAULT_ENGINE, label="Card engine (spike only)",
                                  elem_classes="pvf-engine")
                with gr.Row(elem_classes="pvf-actions"):
                    back_search = gr.Button("Back", size="lg", elem_classes=["pvf-big-btn", "pvf-back"])
                    go_btn = gr.Button("Search", variant="primary", size="lg",
                                       elem_classes=["pvf-big-btn", "pvf-fwd", "pvf-go"])

            with nav.screen("results"):
                with gr.Row(elem_classes="pvf-bar"):
                    back_results = gr.Button("Back", size="lg", elem_classes=["pvf-big-btn", "pvf-back"],
                                             scale=0, min_width=110)
                    count = gr.HTML('<h2 class="pvf-h2">Results</h2>')
                with gr.Column(visible=True, elem_classes="pvf-cards-html") as box_html:
                    html_cards = gr.HTML(cards_html([]), js_on_load=CARD_JS, elem_classes="pvf-cards")
                with gr.Column(visible=False, elem_classes="pvf-cards-ds") as box_ds:
                    dataset = gr.Dataset(components=["html"], samples=[], layout="gallery",
                                         samples_per_page=CARD_COUNT, type="index", label="")
                with gr.Column(visible=False, elem_classes="pvf-cards-radio") as box_radio:
                    radio = gr.Radio([], label="Players", show_label=False, container=False)
                with gr.Column(visible=False, elem_classes="pvf-cards-render") as box_render:
                    @gr.render(inputs=render_cards)
                    def draw_cards(cards):
                        with gr.Row(elem_classes="pvf-rgrid"):
                            for c in cards:
                                # Option B1: key keeps components stable across re-renders
                                btn = gr.Button(card_text(c), key=f"card-{c['pid']}",
                                                elem_classes=["pvf-rcard", "pvf-fwd"])
                                btn.click(partial(open_player, c["pid"]), inputs=[horizon],
                                          outputs=open_outputs).then(None, js=AFTER_NAV_JS)

            with nav.screen("player"):
                with gr.Row(elem_classes="pvf-bar"):
                    back_player = gr.Button("Back", size="lg", elem_classes=["pvf-big-btn", "pvf-back"],
                                            scale=0, min_width=110)
                head = gr.HTML(elem_classes="pvf-head-wrap")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        chart = gr.Plot(show_label=False)
                    with gr.Column(scale=2, min_width=280):
                        card = gr.Markdown(elem_classes="pvf-panel")
                        explain = gr.Markdown(elem_classes="pvf-panel")

    # Hidden button the browser Back hook clicks after popstate
    pop = gr.Button("pop", elem_classes="pvf-pop")

    nav_out = nav.outputs + [screen]
    filters = [search, league, position, age_min, age_max, horizon, sort_by]
    boxes = [box_html, box_ds, box_radio, box_render]
    search_outputs = nav_out + [count, ids, html_cards, dataset, radio, render_cards] + boxes
    open_outputs = nav_out + [head, chart, card, explain]

    def go(name):
        return [*nav.to(name), name]

    def go_back(current):
        return go(BACK_TO.get(current, "home"))

    def run_search(search, leagues, positions, age_min, age_max, horizon, sort_by, engine_label):
        """One event: filter, fill only the chosen engine, switch screen."""
        df, total, h = find_players(search, leagues, positions, age_min, age_max, horizon, sort_by)
        cards = [card_data(row, h) for _, row in df.iterrows()]
        which = ENGINES.get(engine_label, "html")
        skip = gr.skip()
        # Unused engines are skipped so their cost never pollutes a timing
        html_out = cards_html(cards) if which == "html" else skip
        ds_out = (gr.Dataset(samples=[[f'<div class="pvf-card">{card_inner(c)}</div>']
                                      for c in cards]) if which == "dataset" else skip)
        radio_out = (gr.Radio(choices=[(card_text(c), str(c["pid"])) for c in cards], value=None)
                     if which == "radio" else skip)
        render_out = cards if which == "render" else skip
        label = (f'<h2 class="pvf-h2">Results <small>{min(total, CARD_COUNT)} of {total} '
                 'matches</small></h2>')
        vis = [gr.Column(visible=ENGINES[k] == which) for k in ENGINES]
        return [*go("results"), label, [c["pid"] for c in cards], html_out, ds_out, radio_out,
                render_out, *vis]

    def open_player(pid, horizon):
        """Every engine ends here with a player id; unknown ids stay put."""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return [gr.skip()] * len(open_outputs)
        if pid not in PLAYER_BY_ID:
            return [gr.skip()] * len(open_outputs)
        h = horizon_of(horizon)
        return [*go("player"), player_head(pid, h), make_chart(pid), make_card(pid, h),
                make_explanation(pid, h)]

    def open_from_html(horizon, evt: gr.EventData):
        # trigger('click', {pid}) arrives as an attribute on EventData
        return open_player(getattr(evt, "pid", None), horizon)

    def open_from_index(index, id_list, horizon):
        # Dataset sends the clicked sample's position on the page
        ok = isinstance(index, int) and 0 <= index < len(id_list)
        return open_player(id_list[index] if ok else None, horizon)

    def open_from_radio(value, horizon):
        # Clearing the value lets the same card fire again after Back
        return [*open_player(value, horizon), gr.Radio(value=None)]

    start.click(lambda: go("search"), None, nav_out).then(None, js=AFTER_NAV_JS)
    top.click(run_search, [gr.State(""), gr.State([]), gr.State([]), gr.State(AGE_FLOOR),
                           gr.State(AGE_CEILING), gr.State(DEFAULT_HORIZON),
                           gr.State(DEFAULT_SORT), engine], search_outputs).then(None, js=AFTER_NAV_JS)
    go_btn.click(run_search, filters + [engine], search_outputs).then(None, js=AFTER_NAV_JS)
    search.submit(run_search, filters + [engine], search_outputs).then(None, js=AFTER_NAV_JS)
    html_cards.click(open_from_html, [horizon], open_outputs).then(None, js=AFTER_NAV_JS)
    dataset.click(open_from_index, [dataset, ids, horizon], open_outputs).then(None, js=AFTER_NAV_JS)
    radio.input(open_from_radio, [radio, horizon], open_outputs + [radio]).then(None, js=AFTER_NAV_JS)
    for btn in (back_search, back_results, back_player, pop):
        btn.click(go_back, screen, nav_out).then(None, js=AFTER_NAV_JS)


# Theme values are set for light and dark alike, so the page is always night
def both(**tokens) -> dict:
    return {**tokens, **{f"{k}_dark": v for k, v in tokens.items()}}


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
    font=["system-ui", "Segoe UI", "sans-serif"], radius_size=gr.themes.sizes.radius_sm,
).set(**both(
    body_background_fill="transparent", body_text_color=TEXT, body_text_color_subdued=TEXT_DIM,
    background_fill_primary=PANEL, background_fill_secondary=PANEL_2,
    block_background_fill=PANEL, block_border_color=LINE, border_color_primary=LINE,
    block_label_text_color=TEXT_DIM, block_title_text_color=TEXT, block_info_text_color=TEXT_DIM,
    input_background_fill="#081710", input_border_color=LINE,
    input_placeholder_color="#93A39A",
    button_primary_background_fill=GOLD, button_primary_background_fill_hover="#E6BD4A",
    button_primary_text_color="#0A140E", button_primary_border_color=GOLD,
    button_secondary_background_fill=PANEL_2, button_secondary_background_fill_hover="#1A3A2A",
    button_secondary_text_color=TEXT, button_secondary_border_color=LINE,
    checkbox_label_background_fill=PANEL_2, checkbox_label_background_fill_selected="#1E4A33",
    checkbox_label_text_color=TEXT, checkbox_label_text_color_selected=TEXT,
    color_accent_soft="#1E4A33", slider_color=GOLD,
))

CSS_FILE = (HERE / "style.css").read_text(encoding="utf-8")
# Page rules sit below this marker; Gradio would prefix them inside .contain
PAGE_MARKER = "/* PAGE LEVEL RULES */"
COMPONENT_CSS, PAGE_CSS = CSS_FILE.split(PAGE_MARKER)

# Runs in the browser after every screen change; no server round trip
AFTER_NAV_JS = "() => { if (window.pvfAfterNav) window.pvfAfterNav(); }"

# Browser Back hook: forward clicks push history, popstate clicks the hidden pop.
# Also saves scroll per screen: forward moves go to the top, Back restores it.
HISTORY_JS = """
<script>
(() => {
  const scrolls = {};
  let goingBack = false;
  const current = () => {
    const el = [...document.querySelectorAll('.pvf-screen')].find(e => e.offsetParent !== null);
    return el ? [...el.classList].find(c => c.startsWith('pvf-') && c !== 'pvf-screen') : null;
  };
  document.addEventListener('click', (e) => {
    const s = current();
    if (s) scrolls[s] = scrollY;
    if (e.target.closest('.pvf-back')) goingBack = true;
  }, true);
  window.addEventListener('popstate', () => { goingBack = true; });
  window.pvfAfterNav = () => {
    const y = goingBack ? (scrolls[current()] || 0) : 0;
    goingBack = false;
    // Wait a frame so the new screen has laid out before scrolling
    requestAnimationFrame(() => window.scrollTo(0, y));
  };
  if (new URLSearchParams(location.search).has('nohist')) return;
  let depth = 0;
  const FWD = '.pvf-fwd, .pvf-card, .pvf-cards-ds button, .pvf-cards-radio label';
  document.addEventListener('click', (e) => {
    const back = e.target.closest('.pvf-back');
    // Capture phase runs first, so in app Back becomes a real history step
    if (back && depth > 0) { e.stopPropagation(); e.preventDefault(); history.back(); return; }
    if (e.target.closest(FWD)) { depth += 1; history.pushState({pvf: depth}, ''); }
  }, true);
  window.addEventListener('popstate', (e) => {
    const target = (e.state && e.state.pvf) || 0;
    if (target >= depth) { depth = target; return; }
    depth = target;
    const pop = document.querySelector('button.pvf-pop, .pvf-pop button');
    if (pop) pop.click();
  });
})();
</script>
"""
HEAD = f"<style>{PAGE_CSS}</style>{HISTORY_JS}"

with gr.Blocks(title="Player value forecaster (spike)") as demo:
    build(ColumnNav())

with demo.route("Tabs", "/tabs", show_in_navbar=False):
    build(TabsNav())

if __name__ == "__main__":
    demo.launch(theme=THEME, css=COMPONENT_CSS, head=HEAD)
