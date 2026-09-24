"""Gradio front end for the player market value forecaster.

Reads only the prediction bundle in ./predictions, never the model,
so the Space stays fast and the pipeline can change independently.
"""
import json
import math
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

EXPLAIN_TITLE = "#### What this means"
EXPLAIN_PROMPT = "Select a player to see what the forecast means."
EXPLAIN_NO_FORECAST = "No forecast is available for this player yet."
# Widest share of ranges, per horizon, that counts as low confidence
LOW_CONFIDENCE_SHARE = 0.2
MIN_HISTORY_ROWS = 3
FLAG = "⚠ "

# Palette: pitch green, chalk lines, gold for the forecast band
PITCH, CHALK, INK, GRASS, GOLD, MUTED = "#14402F", "#F2F3EC", "#17201C", "#3E7A5B", "#D4A72C", "#6B7A72"
# Helper and hint text colour; 7.6:1 on white, unlike stone 400
SUBDUED = "#57534E"


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


def width_cutoffs(players) -> dict[int, float]:
    """Relative range width at which a forecast joins the widest 20%."""
    # Quantile skips NaN, so players without a forecast never shift it
    return {h: players[f"width_{h}"].quantile(1 - LOW_CONFIDENCE_SHARE)
            for h in HORIZONS.values()}


PLAYERS, HISTORY, FORECASTS, MANIFEST = load_bundle()
PLAYERS = add_derived_columns(PLAYERS, FORECASTS)
HISTORY_BY_ID, FORECASTS_BY_ID, PLAYER_BY_ID = build_lookups(PLAYERS, HISTORY, FORECASTS)
# Fixed at startup so a filter never moves what counts as wide
WIDTH_CUTOFF = width_cutoffs(PLAYERS)


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


def seasons_text(h: int) -> str:
    return "1 season" if h == 1 else f"{h} seasons"


def history_count(pid: int) -> int:
    h = HISTORY_BY_ID.get(pid)
    return 0 if h is None else len(h)


def low_confidence_reasons(width, n_history: int, h: int) -> list[str]:
    """Which low confidence rules a forecast trips: 'width', 'history' or none."""
    # NaN width means no usable forecast, so there is nothing to flag
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


def flag_names(df, h: int) -> list[str]:
    """Display names with a warning prefix for low confidence forecasts."""
    # Only the shown rows are checked, so this costs at most MAX_ROWS lookups
    return [FLAG + name if low_confidence_reasons(width, history_count(int(pid)), h) else name
            for pid, name, width in zip(df["player_id"], df["name"], df[f"width_{h}"])]


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
        # Search and examples match df["name"], so the prefix never interferes
        "Player": flag_names(df, h), "Age": df["age"], "Position": df["sub_position"],
        "Club": df["club_name"], "League": df["league_name"], "Nationality": df["nationality"],
        "Value": df["current_value_eur"].map(fmt_eur),
        "Change": df[f"change_{h}"].map(fmt_change),
        "Likely range": [fmt_range(lo, hi) for lo, hi in zip(df[f"p10_{h}"], df[f"p90_{h}"])],
    }, columns=COLUMNS)
    if total:
        shown = (f" (showing the first {MAX_ROWS}, sorted by {sort_by.lower()})"
                 if total > MAX_ROWS else "")
        # Example clicks often leave one match, so get the singular right
        who = "**1 player** matches" if total == 1 else f"**{total} players** match"
        count = f"{who} these filters{shown}. Select a row to see the forecast."
    else:
        count = no_match_message(lo, hi, active)

    if selected_id is None:
        chart_out, card_out, sel_out = None, CARD_PROMPT, None
        explain_out = make_explanation(None, h)
    elif int(selected_id) in matched_ids:
        # Chart stays put; card and explanation redraw for the chosen horizon
        chart_out, card_out, sel_out = gr.skip(), make_card(int(selected_id), h), selected_id
        explain_out = make_explanation(int(selected_id), h)
    else:
        chart_out, card_out, sel_out = None, CARD_GONE, None
        explain_out = make_explanation(None, h)
    return view, df["player_id"].tolist(), count, chart_out, card_out, explain_out, sel_out


def default_filters() -> list:
    """Starting value for every filter, in the same order as the inputs."""
    return ["", [], [], [], [], AGE_FLOOR, AGE_CEILING, DEFAULT_HORIZON, DEFAULT_SORT]


def reset_filters(selected_id):
    """Clear every filter and refresh the results in one single run."""
    defaults = default_filters()
    return (*defaults, *filter_players(*defaults, selected_id))


def open_player(pid, horizon=DEFAULT_HORIZON):
    """Chart, card, explanation and selection for one id, guarding unknown ids."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None, CARD_MISSING, make_explanation(None), None
    if pid not in PLAYER_BY_ID:
        return None, CARD_MISSING, make_explanation(None), None
    h = horizon_of(horizon)
    return make_chart(pid), make_card(pid, h), make_explanation(pid, h), pid


def featured_player_id(players):
    """Highest current value, so a first visit never lands on a blank chart."""
    ranked = players.dropna(subset=["current_value_eur"])
    if ranked.empty:
        return None
    return int(ranked.sort_values(["current_value_eur", "player_id"],
                                  ascending=[False, True])["player_id"].iloc[0])


def initial_view():
    """Page load: default table with the featured player already open."""
    view, ids, count, *_ = filter_players(*default_filters(), None)
    if FEATURED_ID is None:
        return view, ids, count, None, CARD_PROMPT, make_explanation(None), None
    return (view, ids, count, *open_player(FEATURED_ID))


MIN_EXAMPLE_VALUE = 1e6


def is_valued(p):
    return p["current_value_eur"] >= MIN_EXAMPLE_VALUE


# (label, who qualifies, ranking column, take the largest); star goes first
# so it matches the featured player and no other rule can claim it
EXAMPLE_RULES = [
    ("Established star", lambda p: p["current_value_eur"].notna(), "current_value_eur", True),
    ("Rising young player", lambda p: (p["age"] <= 23) & is_valued(p), "change_1", True),
    ("Veteran in decline", lambda p: (p["age"] >= 31) & is_valued(p), "change_1", False),
    ("Hardest to predict", is_valued, "width_1", True),
]


def pick_examples(players) -> list[tuple[str, int, str]]:
    """Apply each rule in turn; players already picked are not eligible again."""
    picks, taken = [], set()
    for label, qualifies, col, largest in EXAMPLE_RULES:
        pool = players[qualifies(players) & ~players["player_id"].isin(taken)].dropna(subset=[col])
        # A rule that finds nobody is skipped rather than faked
        if pool.empty:
            continue
        best = pool.sort_values([col, "player_id"], ascending=[not largest, True]).iloc[0]
        picks.append((label, int(best["player_id"]), best["name"]))
        taken.add(int(best["player_id"]))
    return picks


def open_example(name, player_id):
    """Example click: search that name, reset the rest, open that exact id."""
    values = default_filters()
    values[0] = name or ""
    view, ids, count, *_ = filter_players(*values, None)
    # Names can repeat, so the hidden id decides which player opens
    return (*values, view, ids, count, *open_player(player_id))


FEATURED_ID = featured_player_id(PLAYERS)
EXAMPLES = pick_examples(PLAYERS)


def history_points(pid: int, player: dict):
    """Value history as x and y lists, or the current value alone."""
    h = HISTORY_BY_ID.get(pid)
    if h is None or h.empty:
        # Players with no history still anchor on the players table row
        return [player["value_date"]], [player["current_value_eur"]]
    return h["date"].tolist(), h["value_eur"].tolist()


def money_label(v: float) -> str:
    """Axis tick in the card's € shorthand, without forced decimals: €10m, €500k."""
    if v == 0:
        return "€0"
    return f"€{v / 1e6:g}m" if v >= 1e6 else f"€{v / 1e3:g}k"


def money_ticks(top: float, target: int = 5) -> tuple[list[float], list[str]]:
    """Round tick values from zero to just above top, with € labels."""
    if top is None or pd.isna(top) or top <= 0:
        top = 1e6
    rough = top / target
    magnitude = 10 ** math.floor(math.log10(rough))
    # First 1, 2, 2.5 or 5 step that keeps roughly `target` ticks
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rough)
    # Zero is left unlabelled so it never collides with the first year
    values = [i * step for i in range(1, math.ceil(top / step) + 1)]
    return values, [money_label(v) for v in values]


def make_chart(pid: int):
    """Value history line plus 1 to 3 season forecast with a 10 to 90% band."""
    player = PLAYER_BY_ID.get(pid)
    if player is None:
        return None
    hx, hy = history_points(pid, player)
    f = FORECASTS_BY_ID.get(pid)
    has_forecast = f is not None and not f.empty
    last_date, last_val = hx[-1], hy[-1]

    fig = go.Figure()
    top = max((v for v in hy if pd.notna(v)), default=0)
    if has_forecast:
        tx = f["target_date"].tolist()
        lo, mid, hi = f["p10_eur"].tolist(), f["p50_eur"].tolist(), f["p90_eur"].tolist()
        top = max([top] + [v for v in hi if pd.notna(v)])
        # Band starts at the last known value so it opens from today
        bx = [last_date] + tx
        fig.add_trace(go.Scatter(x=bx + bx[::-1], y=[last_val] + hi + lo[::-1] + [last_val],
                                 fill="toself", mode="lines", fillcolor="rgba(212,167,44,0.22)",
                                 line=dict(width=0), hoverinfo="skip", name="Likely range",
                                 legendrank=3))
    # Hover text reuses the card's formatter so both read identically
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", name="Market value", legendrank=1,
                             line=dict(color=GRASS, width=2.5, shape="hv"), marker=dict(size=5),
                             customdata=[fmt_eur(v) for v in hy],
                             hovertemplate="Market value: %{customdata}<extra></extra>"))
    if has_forecast:
        fig.add_trace(go.Scatter(
            x=tx, y=mid, mode="lines+markers", name="Median forecast", legendrank=2,
            line=dict(color=GOLD, width=2.5, dash="dash"), marker=dict(size=7),
            customdata=[[fmt_eur(med), fmt_range(low, high)] for low, med, high in zip(lo, mid, hi)],
            hovertemplate=("Median forecast: %{customdata[0]}<br>"
                           "Likely range: %{customdata[1]}<extra></extra>")))
        # Separate connector, so hovering today never repeats the current value
        fig.add_trace(go.Scatter(x=[last_date, tx[0]], y=[last_val, mid[0]], mode="lines",
                                 line=dict(color=GOLD, width=2.5, dash="dash"),
                                 hoverinfo="skip", showlegend=False))
        # Dotted rule marks where observed history stops and forecast starts
        if pd.notna(last_date):
            fig.add_vline(x=last_date, line=dict(color=MUTED, width=1, dash="dot"))
    ticks, labels = money_ticks(top)
    fig.update_layout(
        height=380, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", size=13, color=INK),
        # Fixed entry widths stop late web fonts clipping a measured label
        legend=dict(orientation="h", yanchor="top", y=-0.1, x=0, bgcolor="rgba(0,0,0,0)",
                    entrywidth=150, entrywidthmode="pixels"),
        # Light tooltip with dark ink stays readable on either page theme
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=MUTED, font=dict(color=INK)),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=GRASS),
        yaxis=dict(tickvals=ticks, ticktext=labels, range=[0, ticks[-1] * 1.02],
                   gridcolor="rgba(107,122,114,0.25)", zeroline=False),
        xaxis=dict(hoverformat="%b %Y", gridcolor="rgba(107,122,114,0.12)", zeroline=False),
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


def warning_text(reasons: list[str], n_history: int, h: int) -> str:
    """One blockquote line naming every low confidence rule that applied."""
    said = []
    if "width" in reasons:
        said.append(f"this range is among the widest 20% of {seasons_text(h)} forecasts")
    if "history" in reasons:
        said.append("this player has no value history" if n_history == 0 else
                    f"this player has only {n_history} past "
                    f"valuation{'' if n_history == 1 else 's'}")
    # Blockquote gives CSS a single element to hang the gold edge on
    return f"> {FLAG}**Low confidence:** {', and '.join(said)}. Treat it as a rough guide."


def change_text(p50, current) -> str:
    """Median against today's value, rounded exactly like the table's Change column."""
    if current is None or pd.isna(current) or current <= 0:
        return ""
    change = p50 / current - 1
    pct = f"{abs(change):.0%}"
    if pct == "0%":
        return f", about the same as {fmt_eur(current)} today"
    return f", {'up' if change > 0 else 'down'} {pct} from {fmt_eur(current)} today"


def spread_text(rows: dict) -> str:
    """How the range's width moves from the nearest to the furthest horizon."""
    first, last = min(HORIZONS.values()), max(HORIZONS.values())
    if first not in rows or last not in rows:
        return ""
    near = rows[first].p90_eur - rows[first].p10_eur
    far = rows[last].p90_eur - rows[last].p10_eur
    if pd.isna(near) or pd.isna(far) or near <= 0:
        return ""
    ratio = far / near
    span = (f"from {fmt_eur(near)} at {seasons_text(first)} ahead "
            f"to {fmt_eur(far)} at {seasons_text(last)}")
    # Five percent either way reads as no real change to a user
    if ratio >= 1.05:
        how = f"{ratio - 1:.0%} wider" if ratio < 2 else f"{ratio:.1f} times as wide"
        return f"The range widens the further ahead the model looks: its width grows {span}, {how}."
    if ratio <= 0.95:
        return (f"Unusually, the range narrows further ahead: its width shrinks {span}, "
                f"{1 - ratio:.0%} narrower.")
    return f"The range stays about the same width further ahead: its width goes {span}."


def make_explanation(pid, horizon: int = 1) -> str:
    """Fixed template in plain words for one player and horizon; no LLM."""
    try:
        player = PLAYER_BY_ID.get(int(pid))
    except (TypeError, ValueError):
        player = None
    if player is None:
        return f"{EXPLAIN_TITLE}\n\n{EXPLAIN_PROMPT}"
    pid = int(pid)
    f = FORECASTS_BY_ID.get(pid)
    rows = {} if f is None else {int(r.horizon): r for r in f.itertuples()}
    r = rows.get(horizon)
    if r is None or pd.isna(r.p50_eur):
        return f"{EXPLAIN_TITLE}\n\n{EXPLAIN_NO_FORECAST}"

    by = f" (by {r.target_date:%b %Y})" if pd.notna(r.target_date) else ""
    # Same formatters as the card, so both always show identical numbers
    middle = (f"In {seasons_text(horizon)}{by}, the model's middle estimate is "
              f"{fmt_eur(r.p50_eur)}{change_text(r.p50_eur, player['current_value_eur'])}.")
    likely = (f"Its likely range is {fmt_range(r.p10_eur, r.p90_eur)}. The model aims for the "
              "real value to land inside this range 8 times out of 10.")
    value_date = player["value_date"]
    anchor = f"{value_date.day} {value_date:%b %Y}" if pd.notna(value_date) else "its last valuation"
    limits = ("<small>Based only on past Transfermarkt valuations. It does not know about "
              f"injuries, contracts or transfers after {anchor}.</small>")

    parts = [EXPLAIN_TITLE]
    n_history = history_count(pid)
    reasons = low_confidence_reasons(player.get(f"width_{horizon}"), n_history, horizon)
    if reasons:
        parts.append(warning_text(reasons, n_history, horizon))
    parts.append(f"{middle} {likely}")
    spread = spread_text(rows)
    if spread:
        parts.append(spread)
    parts.append(limits)
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
    # Table, ids and count are skipped so every event shares one output list
    keep = (gr.skip(), gr.skip(), gr.skip())
    # A click can land after the table shrank underneath the user
    if not ids or row is None or not 0 <= row < len(ids):
        return (*keep, None, CARD_STALE, make_explanation(None), None)
    return (*keep, *open_player(ids[row], horizon))


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
).set(body_background_fill=CHALK, block_background_fill="#FFFFFF",
      button_primary_background_fill=PITCH, button_primary_text_color=CHALK,
      # Stone 400 defaults reach only 2.5:1 on white; stone 600 passes 4.5:1
      block_info_text_color=SUBDUED, body_text_color_subdued=SUBDUED,
      block_label_text_color=SUBDUED,
      # Placeholders need 4.5:1 on the input fill in both themes too
      input_placeholder_color="#6A645F", input_placeholder_color_dark="#BAB4AE")
CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

APP_TITLE = "Player Value Forecaster"
LOOK_AHEAD_INFO = "How many seasons ahead the forecast looks"
SORT_INFO = "Order the table by value, predicted change or how uncertain the forecast is"
TABLE_NOTE = ("⚠ low-confidence forecast. Likely range: where the model aims for the real "
              "value to land 8 times out of 10.")

with gr.Blocks(title=APP_TITLE) as demo:
    ids_state = gr.State([])
    selected_state = gr.State(None)

    gr.HTML("""
      <header class="pvf-head">
        <h1>Player value forecaster</h1>
        <p>Pick a football player to see where their Transfermarkt value is likely heading
        over the next one to three seasons.</p>
      </header>""")
    if MANIFEST.get("is_mock"):
        gr.HTML('<div class="pvf-mock">Demo data: these players and forecasts are invented '
                'while the models are being trained.</div>')

    # Results are built first so the examples can target them, placed below
    horizon = gr.Radio(list(HORIZONS), value=DEFAULT_HORIZON, label="Look ahead",
                       info=LOOK_AHEAD_INFO, render=False)
    sort_by = gr.Dropdown(list(SORTS), value=DEFAULT_SORT, label="Sort by", info=SORT_INFO,
                          render=False)
    count = gr.Markdown(render=False)
    # Pinning keeps the name in view while the table scrolls sideways on phones
    table = gr.Dataframe(interactive=False, max_height=340, wrap=True, elem_classes="pvf-table",
                         column_widths=["13%", "5%", "12%", "14%", "11%", "12%", "8%", "9%", "16%"],
                         pinned_columns=1, render=False)
    chart = gr.Plot(show_label=False, elem_classes="pvf-chart", render=False)
    card = gr.Markdown(CARD_PROMPT, elem_classes="pvf-card", render=False)
    explain = gr.Markdown(make_explanation(None), elem_classes="pvf-explain", render=False)
    results = [table, ids_state, count, chart, card, explain, selected_state]

    with gr.Row(equal_height=False, elem_classes="pvf-main"):
        with gr.Column(scale=1, min_width=260, elem_classes="pvf-filters"):
            search = gr.Textbox(label="Search player", placeholder="Type part of a name",
                                max_lines=1)
            # Secondary filters start folded so search and results lead the page
            with gr.Accordion("More filters", open=False, elem_classes="pvf-more"):
                position = gr.CheckboxGroup(options("position"), label="Position")
                league = gr.Dropdown(options("league_name"), multiselect=True, label="League")
                country = gr.Dropdown(options("league_country"), multiselect=True,
                                      label="League country")
                nation = gr.Dropdown(options("nationality"), multiselect=True, label="Nationality")
                age_min = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_FLOOR, step=1,
                                    label="Youngest age")
                age_max = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_CEILING, step=1,
                                    label="Oldest age")
            reset = gr.Button("Clear filters", variant="secondary")
            filters = [search, league, country, nation, position, age_min, age_max, horizon, sort_by]

            # Carries the example's player id, since names alone can repeat
            example_id = gr.Number(visible=False, precision=0)
            if EXAMPLES:
                # cache_examples=False matters: Spaces would otherwise cache and skip fn
                gr.Examples([[name, pid] for _label, pid, name in EXAMPLES],
                            inputs=[search, example_id], outputs=filters + results,
                            fn=open_example, run_on_click=True, cache_examples=False,
                            example_labels=[label for label, _pid, _name in EXAMPLES],
                            label="Try an example", api_name="open_example")

        with gr.Column(scale=3, elem_classes="pvf-side"):
            with gr.Row(equal_height=True, elem_classes="pvf-view"):
                horizon.render()
                sort_by.render()
            # Results and detail are siblings so phone CSS can swap their order
            with gr.Column(elem_classes="pvf-results"):
                count.render()
                table.render()
                gr.Markdown(TABLE_NOTE, elem_classes="pvf-note")
            with gr.Row(equal_height=False, elem_classes="pvf-detail"):
                # Explanation sits directly under the chart it describes
                with gr.Column(scale=3):
                    chart.render()
                    explain.render()
                with gr.Column(scale=2, min_width=320):
                    card.render()

    with gr.Accordion("How the forecast works", open=False, elem_classes="pvf-how"):
        gr.Markdown(
            "Each forecast predicts the change in value 1, 2 and 3 seasons after the "
            "1 July snapshot. The shaded band is the range the model expects the value to land "
            "in 8 times out of 10. Models were backtested on later seasons than they were "
            "trained on.")

    # Footer sits outside every container so it stays visible at all times
    gr.HTML(make_footer(MANIFEST), elem_classes="pvf-foot")

    # .input fires on user edits only, so code-set values never refilter
    gr.on(triggers=[comp.input for comp in filters],
          fn=filter_players, inputs=filters + [selected_state], outputs=results,
          trigger_mode="always_last", api_name="filter_players")
    # First paint opens the featured player instead of an empty chart
    demo.load(initial_view, None, results, api_name="initial_view")
    table.select(on_select, [ids_state, horizon], results, api_name="select_player")
    # One run sets every filter and its results together, no cascade
    reset.click(reset_filters, selected_state, filters + results, api_name="reset_filters")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
