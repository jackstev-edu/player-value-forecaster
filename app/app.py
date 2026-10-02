"""Gradio front end for the player market value forecaster.

Reads only the prediction bundle in ./predictions, never the model,
so the Space stays fast and the pipeline can change independently.
"""
import html
import json
import math
import os
import unicodedata
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# Tests point this at a fixed mock bundle; the Space always uses ./predictions
BUNDLE_DIR = Path(os.environ.get("PVF_BUNDLE_DIR", Path(__file__).parent / "predictions"))
MAX_ROWS = 200
AGE_FLOOR, AGE_CEILING = 15, 45

# Columns the table always shows, even when no player matches
COLUMNS = ["Player", "Age", "Position", "Club", "League", "Nationality", "Value",
           "Change", "Likely range"]

# Every horizon this app can label; the bundle decides which are offered
HORIZON_LABELS = {1: "1 season", 2: "2 seasons", 3: "3 seasons"}
# Replaced at startup by the horizons the loaded bundle actually forecasts
HORIZONS = {label: h for h, label in HORIZON_LABELS.items()}
DEFAULT_HORIZON = "1 season"
# Sort label to (column prefix, ascending); prefixes gain the horizon suffix
SORT_RULES = {"Current value": ("current_value_eur", False),
              "Biggest predicted rise": ("change", False),
              "Biggest predicted fall": ("change", True),
              "Most uncertain": ("width", False)}
# Renamed at startup when the bundle predicts no falls for this label to find
FALL_LABEL = "Biggest predicted fall"
NO_FALL_LABEL = "Smallest predicted rise"
SORTS = dict(SORT_RULES)
DEFAULT_SORT = "Current value"

# Text columns shown to the user, where a gap must not read as nan
LABEL_COLUMNS = ["position", "sub_position", "club_name", "league_name",
                 "league_country", "nationality"]
UNKNOWN_LABEL = "Unknown"
# Placeholders the pipeline writes for a field it could not fill
MISSING_TOKENS = {"", "missing", "unknown", "none", "na", "n/a", "nan"}

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
FLAG_TITLE = "Low confidence forecast"
# Cards drawn per page, and how many Load more adds
PAGE = 24
# Screen order, so Back always means the screen before this one
SCREENS = ["home", "search", "results", "player"]
BACK_TO = {"home": "home", "search": "home", "results": "search", "player": "results"}

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


def offered_horizons(forecasts) -> dict[str, int]:
    """Label to seasons for the horizons this bundle really forecasts."""
    present = sorted({int(h) for h in forecasts["horizon"].dropna().unique()
                      if int(h) in HORIZON_LABELS})
    # A bundle with no usable horizon still leaves the control working
    return {HORIZON_LABELS[h]: h for h in present or [1]}


def sort_labels(players, horizon: int) -> dict[str, tuple[str, bool]]:
    """Rename the fall sort when nothing in this bundle is predicted to fall."""
    change = players.get(f"change_{horizon}")
    if change is not None and (change < 0).any():
        return dict(SORT_RULES)
    # Sorting ascending would surface the smallest rises, not any fall
    return {NO_FALL_LABEL if k == FALL_LABEL else k: v for k, v in SORT_RULES.items()}


def falling_share(players, horizon: int) -> float:
    """Share of players the bundle predicts will lose value."""
    change = players[f"change_{horizon}"].dropna()
    return 0.0 if change.empty else float((change < 0).mean())


def clean_labels(players):
    """Nulls and pipeline placeholders become one Unknown, never a shown nan."""
    players = players.copy()
    for col in LABEL_COLUMNS:
        # Strip first, so " " and "Missing " are caught alongside a true null
        text = players[col].astype("string").str.strip()
        unknown = text.isna() | text.str.lower().isin(MISSING_TOKENS)
        players[col] = text.mask(unknown, UNKNOWN_LABEL)
    return players


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
# The model dropped to one horizon, so the control follows the bundle
HORIZONS = offered_horizons(FORECASTS)
DEFAULT_HORIZON = next(iter(HORIZONS))
DEFAULT_H = HORIZONS[DEFAULT_HORIZON]
PLAYERS = add_derived_columns(clean_labels(PLAYERS), FORECASTS)
# Labels are settled here, once the forecasts are known
SORTS = sort_labels(PLAYERS, DEFAULT_H)
FALLING_SHARE = falling_share(PLAYERS, DEFAULT_H)
HISTORY_BY_ID, FORECASTS_BY_ID, PLAYER_BY_ID = build_lookups(PLAYERS, HISTORY, FORECASTS)
# Indexed once so drawing a page of cards never scans the frame
PLAYERS_BY_ID = PLAYERS.set_index("player_id", drop=False)
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


def apply_filters(search, leagues, countries, nations, positions, age_min, age_max):
    """Filter the roster; same rules as before, now returning rows not a table."""
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
    return df[df["age"].between(lo, hi)], lo, hi, active


def card_data(row, h: int) -> dict:
    """Everything one card shows, already formatted the way the table formatted it."""
    pid = int(row["player_id"])
    shown = fmt_change(row[f"change_{h}"])
    # Read the trend off the shown text, so arrow and percent never disagree
    trend = "flat" if shown in ("n/a", "+0%", "-0%") else ("up" if shown.startswith("+") else "down")
    return {"pid": pid, "name": row["name"], "pos": row["sub_position"], "age": int(row["age"]),
            "club": row["club_name"], "value": fmt_eur(row["current_value_eur"]),
            "arrow": {"up": "▲", "down": "▼", "flat": "■"}[trend], "trend": trend,
            "change": shown, "range": fmt_range(row[f"p10_{h}"], row[f"p90_{h}"]),
            "flag": bool(low_confidence_reasons(row[f"width_{h}"], history_count(pid), h))}


def card_markup(c: dict) -> str:
    """One card as a real button, so Tab reaches it and Enter opens it."""
    e = html.escape
    flag = f'<span class="pc-flag" title="{FLAG_TITLE}">⚠</span>' if c["flag"] else ""
    label = f'Open {c["name"]}, {c["value"]}, {c["change"]}'
    return (f'<button type="button" class="pvf-card" data-pid="{c["pid"]}" '
            f'aria-label="{e(label)}">'
            f'<span class="pc-top"><span class="pc-pos">{e(str(c["pos"]))}</span>{flag}</span>'
            f'<span class="pc-name">{e(str(c["name"]))}</span>'
            f'<span class="pc-meta">Age {c["age"]} · {e(str(c["club"]))}</span>'
            f'<span class="pc-value">{c["value"]}</span>'
            f'<span class="pc-change pc-{c["trend"]}">{c["arrow"]} {c["change"]}</span>'
            f'<span class="pc-range">Likely {c["range"]}</span></button>')


def cards_html(ids: list[int], shown: int, h: int) -> str:
    """The first `shown` cards of a result list, as one grid."""
    if not ids:
        return ""
    rows = PLAYERS_BY_ID.loc[list(ids[:shown])]
    body = "".join(card_markup(card_data(row, h)) for _, row in rows.iterrows())
    return f'<div class="pvf-grid">{body}</div>'


def featured_html(h: int) -> str:
    """The four rule-picked examples as cards, each under its truthful label."""
    if not EXAMPLES:
        return ""
    parts = []
    for label, pid, _name in EXAMPLES:
        row = PLAYERS_BY_ID.loc[pid]
        parts.append(f'<div class="pvf-featured-item"><p class="pvf-featured-label">'
                     f'{html.escape(label)}</p>{card_markup(card_data(row, h))}</div>')
    return f'<div class="pvf-featured">{"".join(parts)}</div>'


def player_head(pid: int, h: int) -> str:
    """Name block on the Player screen; data-pid lets a test check the id."""
    c = card_data(PLAYERS_BY_ID.loc[pid], h)
    player = PLAYER_BY_ID[pid]
    e = html.escape
    flag = f'<span class="pc-flag">⚠ Low confidence</span>' if c["flag"] else ""
    return (f'<header class="pvf-player-head" data-pid="{pid}">'
            f'<span class="pc-pos">{e(str(c["pos"]))}</span>{flag}'
            f'<h2>{e(str(player["name"]))}</h2>'
            f'<p>Age {c["age"]} · {e(str(c["club"]))} · {e(str(player["league_name"]))}</p>'
            f'<p class="pvf-big">{c["value"]} <span class="pc-change pc-{c["trend"]}">'
            f'{c["arrow"]} {c["change"]} in {seasons_text(h)}</span></p></header>')


def count_text(total: int, shown: int, sort_by: str) -> str:
    """How many matched and how many are on screen, in the singular when needed."""
    who = "**1 player** matches" if total == 1 else f"**{total} players** match"
    if shown >= total:
        return f"{who} these filters, sorted by {sort_by.lower()}."
    return f"{who} these filters. Showing the first **{shown}**, sorted by {sort_by.lower()}."


def default_filters() -> list:
    """Starting value for every filter, in the same order as the inputs."""
    return ["", [], [], [], [], AGE_FLOOR, AGE_CEILING, DEFAULT_HORIZON, DEFAULT_SORT]


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


MIN_EXAMPLE_VALUE = 1e6


def is_valued(p):
    return p["current_value_eur"] >= MIN_EXAMPLE_VALUE


def directional_label(rising: str, falling: str):
    """Label chosen from the sign of the picked player's own forecast."""
    def choose(row) -> str:
        change = row.get(CHANGE_COL)
        # Only claim a decline when this player really is forecast to fall
        return falling if pd.notna(change) and change < 0 else rising
    return choose


# Ranking columns follow the bundle, since a horizon other than 1 may be all it has
CHANGE_COL, WIDTH_COL = f"change_{DEFAULT_H}", f"width_{DEFAULT_H}"

# (label, who qualifies, ranking column, take the largest); star goes first
# so it matches the featured player and no other rule can claim it. A label
# may be a function of the picked row, for the two that assert a direction.
EXAMPLE_RULES = [
    ("Established star", lambda p: p["current_value_eur"].notna(), "current_value_eur", True),
    (directional_label("Rising young player", "Young player in decline"),
     lambda p: (p["age"] <= 23) & is_valued(p), CHANGE_COL, True),
    (directional_label("Veteran: smallest predicted rise", "Veteran in decline"),
     lambda p: (p["age"] >= 31) & is_valued(p), CHANGE_COL, False),
    ("Hardest to predict", is_valued, WIDTH_COL, True),
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
        text = label(best) if callable(label) else label
        picks.append((text, int(best["player_id"]), best["name"]))
        taken.add(int(best["player_id"]))
    return picks


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
    """Value history line plus every bundled horizon with a 10 to 90% band."""
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


def fmt_day(d) -> str | None:
    """Full date such as 1 Jul 2026, or None when missing."""
    return f"{d.day} {d:%b %Y}" if d is not None and pd.notna(d) else None


def change_text(p50, current, value_date=None) -> str:
    """Median against the latest valuation, rounded like the table's Change column."""
    if current is None or pd.isna(current) or current <= 0:
        return ""
    change = p50 / current - 1
    pct = f"{abs(change):.0%}"
    # Name the valuation's own date; it is not today and not the anchor
    day = fmt_day(value_date)
    latest = f"at the latest valuation ({day})" if day else "at the latest valuation"
    if pct == "0%":
        return f", about the same as {fmt_eur(current)} {latest}"
    return f", {'up' if change > 0 else 'down'} {pct} from {fmt_eur(current)} {latest}"


def spread_text(rows: dict) -> str:
    """How the range's width moves from the nearest to the furthest horizon."""
    first, last = min(HORIZONS.values()), max(HORIZONS.values())
    # One horizon leaves nothing to compare, so the sentence is dropped
    if first == last or first not in rows or last not in rows:
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
    change = change_text(r.p50_eur, player["current_value_eur"], player["value_date"])
    middle = (f"In {seasons_text(horizon)}{by}, the model's middle estimate is "
              f"{fmt_eur(r.p50_eur)}{change}.")
    likely = (f"Its likely range is {fmt_range(r.p10_eur, r.p90_eur)}. The model aims for the "
              "real value to land inside this range 8 times out of 10.")
    # The forecast's anchor is when its knowledge stops, not the last valuation
    anchor = fmt_day(getattr(r, "anchor_date", None)) or "the forecast was made"
    limits = ("<small>Based on past Transfermarkt valuations, age, position and the contract "
              "we looked up by hand. It does not know about injuries, new contracts or "
              f"transfers after {anchor}.</small>")

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


def optimism_note(players, horizon: int) -> str:
    """Warn when forecasts lean one way, so the training bias is visible."""
    change = players[f"change_{horizon}"].dropna()
    if change.empty:
        return ""
    rising = float((change > 0).mean())
    # Only worth saying when the lean is strong enough to mislead
    if rising < 0.6:
        return ""
    typical = float(change.median()) + 1
    return (f"These forecasts lean optimistic: {rising:.0%} of players are predicted to rise, "
            f"typically to {typical:.1f} times their current value. The model learned from "
            "500 players collected because they became valuable, so it has seen few careers "
            "that stall or fade. Read the direction with that in mind.")


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
        '<p>Data: contract dates for 500 players collected by hand by the team; '
        'Transfermarkt values and player profiles, via the Kaggle datasets '
        f'<a href="{kaggle}/davidcariboo/player-scores" target="_blank" rel="noopener">player-scores</a>'
        ' by davidcariboo and '
        f'<a href="{kaggle}/xfkzujqjvx97n/football-datasets" target="_blank" rel="noopener">football-datasets</a>'
        ' by salimt.</p>'
        '<p>Coursework for CMU 24-679 (Yunus Polatoglu &amp; Jack Stevens). '
        'Not for transfer or betting decisions.</p>'
        f'<p>Model {version}, built {built}.</p>'
    )


# Screen callbacks. Each returns the four screen columns plus the screen name, so
# every path through the app writes the same navigation outputs.

def show(name: str) -> list:
    """Visibility updates for the four screens, plus the name to remember."""
    return [gr.Column(visible=n == name) for n in SCREENS] + [name]


def go_back(current: str) -> list:
    return show(BACK_TO.get(current, "home"))


def run_search(search, leagues, countries, nations, positions, age_min, age_max,
               horizon, sort_by):
    """SEARCH: filter, sort, draw the first page of cards, open the Results screen."""
    df, lo, hi, active = apply_filters(search, leagues, countries, nations, positions,
                                       age_min, age_max)
    h = horizon_of(horizon)
    sort_by = sort_by if sort_by in SORTS else DEFAULT_SORT
    ids = sort_rows(df, sort_by, h)["player_id"].tolist()
    total = len(ids)
    shown = min(PAGE, total)
    count = count_text(total, shown, sort_by) if total else no_match_message(lo, hi, active)
    return [*show("results"), cards_html(ids, shown, h), count, ids, shown,
            gr.Button(visible=total > shown)]


def load_more(ids, shown, horizon, sort_by):
    """Add one more page of cards without refiltering or leaving the screen."""
    h = horizon_of(horizon)
    total = len(ids)
    shown = min(shown + PAGE, total)
    sort_by = sort_by if sort_by in SORTS else DEFAULT_SORT
    return (cards_html(ids, shown, h), count_text(total, shown, sort_by), shown,
            gr.Button(visible=total > shown))


# Four screen columns, the screen name, then head, chart, card, explanation and id
PLAYER_OUTPUTS = len(SCREENS) + 6


def open_player_screen(pid, horizon):
    """Everything the Player screen shows, or no change when the id is unknown."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return [gr.skip()] * PLAYER_OUTPUTS
    if pid not in PLAYER_BY_ID:
        return [gr.skip()] * PLAYER_OUTPUTS
    h = horizon_of(horizon)
    return [*show("player"), player_head(pid, h), make_chart(pid), make_card(pid, h),
            make_explanation(pid, h), pid]


def open_from_cards(horizon, evt: gr.EventData):
    """A card click arrives as trigger('click', {pid}) from the browser."""
    return open_player_screen(getattr(evt, "pid", None), horizon)


def clear_filters():
    """Reset every filter tile to its starting value."""
    return default_filters()


def toggle(open_now: bool):
    """Flip a disclosure panel and report its new state."""
    return gr.Column(visible=not open_now), not open_now


# One delegated listener survives every re-render of the card grid
CARD_JS = """
element.addEventListener('click', (e) => {
  const card = e.target.closest('[data-pid]');
  if (card) trigger('click', {pid: Number(card.dataset.pid)});
});
"""

# Night palette: near black, pitch green and gold, shared by the theme and the sheet
NIGHT_BG, PANEL, PANEL_2, LINE = "#05100A", "#0C1D15", "#12291E", "#2C4D3D"
TEXT, TEXT_DIM, ACCENT = "#F2F3EC", "#B5C2BA", "#D4A72C"


def both(**tokens) -> dict:
    """Set each token for light and dark alike, so the page is always night."""
    return {**tokens, **{f"{k}_dark": v for k, v in tokens.items()}}


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Barlow Condensed"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(**both(
    body_background_fill="transparent", body_text_color=TEXT, body_text_color_subdued=TEXT_DIM,
    background_fill_primary=PANEL, background_fill_secondary=PANEL_2,
    block_background_fill=PANEL, block_border_color=LINE, border_color_primary=LINE,
    block_label_text_color=TEXT_DIM, block_title_text_color=TEXT, block_info_text_color=TEXT_DIM,
    input_background_fill="#081710", input_border_color=LINE, input_placeholder_color="#93A39A",
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover="#E6BD4A",
    button_primary_text_color="#0A140E", button_primary_border_color=ACCENT,
    button_secondary_background_fill=PANEL_2, button_secondary_background_fill_hover="#1A3A2A",
    button_secondary_text_color=TEXT, button_secondary_border_color=LINE,
    checkbox_label_background_fill=PANEL_2, checkbox_label_background_fill_selected="#1E4A33",
    checkbox_label_text_color=TEXT, checkbox_label_text_color_selected=TEXT,
    color_accent_soft="#1E4A33", slider_color=ACCENT,
))

SHEET = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
# Rules below this marker style html and body, which css= would scope under .contain
PAGE_MARKER = "/* PAGE LEVEL RULES */"
COMPONENT_CSS, PAGE_CSS = SHEET.split(PAGE_MARKER)

# Runs in the browser after every screen change, with no server round trip
AFTER_NAV_JS = "() => { if (window.pvfAfterNav) window.pvfAfterNav(); }"

# Browser Back walks the screens, and each screen keeps its scroll position
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
  const FWD = '.pvf-fwd, button.pvf-card';
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

APP_TITLE = "Player Value Forecaster"
LOOK_AHEAD_INFO = "How many seasons ahead the forecast looks"
# Horizon wording is written once here, since three places repeat it
HORIZON_SPAN = (seasons_text(max(HORIZONS.values())) if len(HORIZONS) == 1 else
                f"one to {max(HORIZONS.values())} seasons")
# "over the next ..." reads better with a bare noun than with "1 season"
HEAD_SPAN = "season" if list(HORIZONS.values()) == [1] else HORIZON_SPAN
OPTIMISM_NOTE = optimism_note(PLAYERS, DEFAULT_H)
HOW_IT_WORKS = (
    f"Each forecast predicts the change in value {HORIZON_SPAN} after the "
    "1 July snapshot, from the player's age, position, current value and how that value "
    "moved over the past year. The shaded band is where the model expects the real value "
    "to land 8 times out of 10.\n\n"
    "It was trained on 500 contracts we collected by hand, each reused for its other "
    "seasons to reach 1,000 samples, and tested by five-fold cross-validation grouped by "
    "player, so no player appears in both training and test data. A walk-forward check on "
    "five named players put 6 of 10 forecasts inside the band.\n\n"
    "Known limits: those 500 players were chosen because they became valuable, which is "
    "why the forecasts lean upward; the model cannot see injuries, transfers or contract "
    "news after the snapshot; and players much older or cheaper than the training set are "
    "extrapolations.")
SORT_INFO = "Order the cards by value, predicted change or how uncertain the forecast is"
# Naming the share stops the fall sort implying falls are common here
if 0 < FALLING_SHARE < 0.1:
    SORT_INFO += f". Only {FALLING_SHARE:.0%} of players are predicted to fall"
TABLE_NOTE = ("⚠ low-confidence forecast. Likely range: where the model aims for the real "
              "value to land 8 times out of 10.")

APP_TITLE_TEXT = "Player value forecaster"
TAGLINE = f"Where a player’s market value is heading over the next {HEAD_SPAN}."
LEAN_SHORT = "Forecasts lean optimistic"

with gr.Blocks(title=APP_TITLE) as demo:
    screen = gr.State("home")
    ids_state = gr.State([])
    shown_state = gr.State(0)
    selected_state = gr.State(None)
    why_open = gr.State(False)
    how_open = gr.State(False)

    # One id to scope every rule; an id outranks Gradio's own class rules
    with gr.Column(elem_id="pvf"):

        with gr.Column(visible=True, elem_classes=["pvf-screen", "pvf-home"]) as home_screen:
            gr.HTML(f'<header class="pvf-hero"><p class="pvf-kicker">Transfermarkt forecasts'
                    f'</p><h1>{APP_TITLE_TEXT}</h1>'
                    f'<p class="pvf-tagline">{TAGLINE}</p></header>')
            if MANIFEST.get("is_mock"):
                gr.HTML('<p class="pvf-note-line">Demo data: these players and forecasts are '
                        'invented while the models are being trained.</p>')
            elif OPTIMISM_NOTE:
                with gr.Row(elem_classes="pvf-lean"):
                    gr.HTML(f'<p class="pvf-note-line">{LEAN_SHORT}</p>')
                    why_btn = gr.Button("Why?", size="sm", scale=0, min_width=90,
                                        elem_classes="pvf-why")
                with gr.Column(visible=False, elem_classes="pvf-why-panel") as why_panel:
                    gr.Markdown(OPTIMISM_NOTE)
            with gr.Row(elem_classes="pvf-actions"):
                start_btn = gr.Button("Search players", variant="primary", size="lg",
                                      elem_classes=["pvf-big-btn", "pvf-start", "pvf-fwd"])
                how_btn = gr.Button("How it works", size="lg",
                                    elem_classes=["pvf-big-btn", "pvf-how-btn"])
            with gr.Column(visible=False, elem_classes="pvf-how") as how_panel:
                gr.Markdown(HOW_IT_WORKS)
            gr.HTML('<h2 class="pvf-h2">Featured</h2>')
            featured = gr.HTML(featured_html(DEFAULT_H), js_on_load=CARD_JS,
                               elem_classes="pvf-featured-wrap")

        with gr.Column(visible=False, elem_classes=["pvf-screen", "pvf-search"]) as search_screen:
            with gr.Row(elem_classes="pvf-bar"):
                back_search = gr.Button("Back", size="lg", scale=0, min_width=110,
                                        elem_classes=["pvf-big-btn", "pvf-back"])
                gr.HTML('<h2 class="pvf-h2">Search</h2>')
            with gr.Column(elem_classes="pvf-tile"):
                position = gr.CheckboxGroup(options("position"), label="Position")
            with gr.Row(elem_classes="pvf-tile-row"):
                with gr.Column(elem_classes="pvf-tile"):
                    league = gr.Dropdown(options("league_name"), multiselect=True, label="League")
                with gr.Column(elem_classes="pvf-tile"):
                    nation = gr.Dropdown(options("nationality"), multiselect=True, label="Nation")
            with gr.Row(elem_classes="pvf-tile-row"):
                with gr.Column(elem_classes="pvf-tile"):
                    country = gr.Dropdown(options("league_country"), multiselect=True,
                                          label="League country")
                with gr.Column(elem_classes="pvf-tile"):
                    search_box = gr.Textbox(label="Search by name", max_lines=1,
                                            placeholder="Type part of a name")
            with gr.Row(elem_classes=["pvf-tile", "pvf-age"]):
                age_min = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_FLOOR, step=1,
                                    label="Age from", min_width=120)
                age_max = gr.Slider(AGE_FLOOR, AGE_CEILING, value=AGE_CEILING, step=1,
                                    label="Age to", min_width=120)
            # Hidden while the bundle has one horizon, back automatically when it has more
            horizon = gr.Radio(list(HORIZONS), value=DEFAULT_HORIZON, label="Look ahead",
                               info=LOOK_AHEAD_INFO, visible=len(HORIZONS) > 1,
                               elem_classes="pvf-tile")
            with gr.Row(elem_classes="pvf-actions"):
                clear_btn = gr.Button("Clear", size="lg", scale=0, min_width=140,
                                      elem_classes=["pvf-big-btn", "pvf-clear"])
                go_btn = gr.Button("Search", variant="primary", size="lg",
                                   elem_classes=["pvf-big-btn", "pvf-go", "pvf-fwd"])

        with gr.Column(visible=False,
                       elem_classes=["pvf-screen", "pvf-results"]) as results_screen:
            with gr.Row(elem_classes="pvf-bar"):
                back_results = gr.Button("Back", size="lg", scale=0, min_width=110,
                                         elem_classes=["pvf-big-btn", "pvf-back"])
                gr.HTML('<h2 class="pvf-h2">Results</h2>')
            sort_by = gr.Radio(list(SORTS), value=DEFAULT_SORT, label="Sort by", info=SORT_INFO,
                               elem_classes="pvf-chips")
            count = gr.Markdown(elem_classes="pvf-count")
            cards = gr.HTML(js_on_load=CARD_JS, elem_classes="pvf-cards")
            more_btn = gr.Button("Load more", size="lg", visible=False,
                                 elem_classes=["pvf-big-btn", "pvf-more"])
            gr.Markdown(TABLE_NOTE, elem_classes="pvf-note")

        with gr.Column(visible=False, elem_classes=["pvf-screen", "pvf-player"]) as player_screen:
            with gr.Row(elem_classes="pvf-bar"):
                back_player = gr.Button("Back", size="lg", scale=0, min_width=110,
                                        elem_classes=["pvf-big-btn", "pvf-back"])
            head_html = gr.HTML(elem_classes="pvf-head-wrap")
            with gr.Row(equal_height=False, elem_classes="pvf-detail"):
                with gr.Column(scale=3):
                    chart = gr.Plot(show_label=False, elem_classes="pvf-chart")
                    explain = gr.Markdown(make_explanation(None), elem_classes="pvf-explain")
                with gr.Column(scale=2, min_width=300):
                    card = gr.Markdown(CARD_PROMPT, elem_classes="pvf-card-panel")

    # Hidden target the browser Back hook clicks after popstate
    pop = gr.Button("pop", elem_classes="pvf-pop")
    gr.HTML(make_footer(MANIFEST), elem_classes=["pvf-foot", "pvf-foot-dark"])

    filters = [search_box, league, country, nation, position, age_min, age_max, horizon, sort_by]
    nav_out = [home_screen, search_screen, results_screen, player_screen, screen]
    search_out = nav_out + [cards, count, ids_state, shown_state, more_btn]
    player_out = nav_out + [head_html, chart, card, explain, selected_state]

    start_btn.click(lambda: show("search"), None, nav_out).then(None, js=AFTER_NAV_JS)
    go_btn.click(run_search, filters, search_out, api_name="run_search").then(None, js=AFTER_NAV_JS)
    search_box.submit(run_search, filters, search_out).then(None, js=AFTER_NAV_JS)
    # Re-sorting reruns the search, so the chips and the filters cannot disagree
    sort_by.input(run_search, filters, search_out,
                  api_name="sort_results").then(None, js=AFTER_NAV_JS)
    more_btn.click(load_more, [ids_state, shown_state, horizon, sort_by],
                   [cards, count, shown_state, more_btn], api_name="load_more")
    cards.click(open_from_cards, [horizon], player_out,
                api_name="open_player").then(None, js=AFTER_NAV_JS)
    featured.click(open_from_cards, [horizon], player_out,
                   api_name="open_featured").then(None, js=AFTER_NAV_JS)
    clear_btn.click(clear_filters, None, filters, api_name="clear_filters")
    if OPTIMISM_NOTE:
        why_btn.click(toggle, why_open, [why_panel, why_open])
    how_btn.click(toggle, how_open, [how_panel, how_open])
    for btn in (back_search, back_results, back_player, pop):
        btn.click(go_back, screen, nav_out).then(None, js=AFTER_NAV_JS)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=COMPONENT_CSS, head=HEAD)
