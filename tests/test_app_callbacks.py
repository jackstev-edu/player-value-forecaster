"""Callback level tests for the Gradio app, run without launching it."""
import importlib.util
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"
# Invented players with known names and all three horizons; the real bundle changes each export
MOCK_BUNDLE = Path(__file__).resolve().parent / "fixtures" / "mock_bundle"

# A nationality the bundle cannot contain, so these filters match nobody
ABSENT_NATION = "Nowhereland"


@pytest.fixture(scope="module")
def app():
    """Import app.py by path so the Space folder stays self-contained."""
    spec = importlib.util.spec_from_file_location("pvf_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    # The bundle is read at import, so the override only has to hold while it runs
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PVF_BUNDLE_DIR", str(MOCK_BUNDLE))
        spec.loader.exec_module(module)
    return module


def first_player_id(app) -> int:
    return int(app.PLAYERS["player_id"].iloc[0])


def test_importing_the_app_does_not_launch_it(app):
    assert isinstance(app.demo, gr.Blocks)
    assert not app.demo.is_running


def test_chart_and_card_for_a_real_player(app):
    pid = first_player_id(app)
    fig = app.make_chart(pid)
    assert isinstance(fig, go.Figure)
    # Band, history, median and a hover-free connector from today
    assert [t.name for t in fig.data] == ["Likely range", "Market value", "Median forecast", None]
    card = app.make_card(pid)
    assert app.PLAYER_BY_ID[pid]["name"] in card
    assert "Likely range" in card


def test_unknown_player_id_is_handled(app):
    assert app.make_chart(-1) is None
    assert app.make_card(-1) == app.CARD_MISSING


def test_player_without_history_still_renders(app, monkeypatch):
    pid = first_player_id(app)
    without = {k: v for k, v in app.HISTORY_BY_ID.items() if k != pid}
    monkeypatch.setattr(app, "HISTORY_BY_ID", without)
    fig = app.make_chart(pid)
    assert isinstance(fig, go.Figure)
    # Anchored on the players table row, so the value point is still drawn
    assert len(fig.data[1].x) == 1
    assert "No value history is available" in app.make_card(pid)


def test_player_without_forecast_plots_history_only(app, monkeypatch):
    pid = first_player_id(app)
    without = {k: v for k, v in app.FORECASTS_BY_ID.items() if k != pid}
    monkeypatch.setattr(app, "FORECASTS_BY_ID", without)
    fig = app.make_chart(pid)
    assert len(fig.data) == 1
    assert "No forecast is available" in app.make_card(pid)


def test_fmt_eur_formats_and_handles_missing(app):
    assert app.fmt_eur(12_500_000) == "€12.5m"
    assert app.fmt_eur(750_000) == "€750k"
    assert app.fmt_eur(float("nan")) == "unknown"
    assert app.fmt_eur(None) == "unknown"


def test_readme_and_footer_credit_the_same_sources(app):
    """The Space page and the in-app footer must name the same datasets."""
    readme = (Path(__file__).resolve().parents[1] / "app" / "README.md").read_text(encoding="utf-8")
    footer = app.make_footer(app.MANIFEST)
    # Match the Kaggle dataset slug wherever it appears, in markdown or in href
    slugs = lambda text: set(re.findall(r"kaggle\.com/datasets/[\w-]+/([\w-]+)", text))
    assert slugs(footer), "the footer credits no Kaggle dataset at all"
    assert slugs(readme) == slugs(footer), (
        f"README credits {sorted(slugs(readme))}, footer credits {sorted(slugs(footer))}")
    for author in ("davidcariboo", "salimt"):
        if author in footer:
            assert author in readme, f"{author} is credited in the footer but not the README"


def test_footer_falls_back_when_manifest_is_empty(app):
    html = app.make_footer({})
    assert "Model unknown, built unknown" in html
    assert app.make_footer({"created_at": "not a date"}).count("unknown") == 2


def test_search_folds_letters_nfkd_keeps_whole(app):
    assert app.normalise_name("Martin Ødegaard") == "martin odegaard"
    assert app.normalise_name("Łukasz Fabiański") == "lukasz fabianski"
    assert app.normalise_name(None) == ""


def without_forecast(app, monkeypatch, pid, horizon):
    """Drop one player's forecast for one horizon and rebuild derived columns."""
    f = app.FORECASTS
    kept = f[~((f["player_id"] == pid) & (f["horizon"] == horizon))]
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(app.PLAYERS, kept))


def top_value_id(app) -> int:
    return int(app.PLAYERS.sort_values("current_value_eur", ascending=False)["player_id"].iloc[0])


def test_change_and_range_formats(app):
    assert app.fmt_change(0.18) == "+18%"
    assert app.fmt_change(-0.052) == "-5%"
    assert app.fmt_change(float("nan")) == "n/a"
    assert app.fmt_range(8_000_000, 15_000_000) == "€8.0m to €15.0m"
    assert app.fmt_range(None, 1.0) == "n/a"


@pytest.mark.parametrize("h", [1, 2, 3])
def test_card_highlights_only_the_chosen_horizon(app, h):
    pid = first_player_id(app)
    card = app.make_card(pid, h)
    marked = [line for line in card.splitlines() if "▸" in line]
    assert len(marked) == 1
    target = app.FORECASTS_BY_ID[pid].set_index("horizon").loc[h, "target_date"]
    assert f"{target:%b %Y}" in marked[0]


RULE_CHECKS = {
    "Established star": (lambda p: p["current_value_eur"].notna(), "current_value_eur", True),
    "Rising young player": (lambda p: (p["age"] <= 23) & (p["current_value_eur"] >= 1e6),
                            "change_1", True),
    "Veteran in decline": (lambda p: (p["age"] >= 31) & (p["current_value_eur"] >= 1e6),
                           "change_1", False),
    "Hardest to predict": (lambda p: p["current_value_eur"] >= 1e6, "width_1", True),
}


def test_example_picks_are_distinct_and_satisfy_their_rules(app):
    picks = app.EXAMPLES
    assert [label for label, _pid, _name in picks] == list(RULE_CHECKS)
    ids = [pid for _label, pid, _name in picks]
    assert len(set(ids)) == 4
    players = app.PLAYERS
    taken = set()
    for label, pid, name in picks:
        qualifies, col, largest = RULE_CHECKS[label]
        row = players.set_index("player_id").loc[pid]
        assert row["name"] == name
        assert qualifies(players)[players["player_id"] == pid].all()
        # Best among the qualifying players no earlier rule already took
        pool = players[qualifies(players) & ~players["player_id"].isin(taken)][col].dropna()
        assert row[col] == (pool.max() if largest else pool.min())
        taken.add(pid)
    assert picks[0][1] == app.FEATURED_ID


def test_rule_with_no_candidates_is_skipped(app):
    # Cap every age at 30 so no veteran exists
    young = app.PLAYERS.assign(age=app.PLAYERS["age"].clip(upper=30))
    picks = app.pick_examples(young)
    labels = [label for label, _pid, _name in picks]
    assert "Veteran in decline" not in labels and len(labels) == 3
    assert len({pid for _label, pid, _name in picks}) == 3
    assert app.pick_examples(app.PLAYERS.iloc[0:0]) == []


def widen_band(app, monkeypatch, pid, horizon, factor=3.0):
    """Stretch one player's band at one horizon; the startup cutoff stays put."""
    f = app.FORECASTS.copy()
    rows = (f["player_id"] == pid) & (f["horizon"] == horizon)
    f.loc[rows, "p10_eur"] /= factor
    f.loc[rows, "p90_eur"] *= factor
    players = app.add_derived_columns(app.PLAYERS, f)
    _history, by_forecast, by_player = app.build_lookups(players, app.HISTORY, f)
    monkeypatch.setattr(app, "PLAYERS", players)
    monkeypatch.setattr(app, "FORECASTS_BY_ID", by_forecast)
    monkeypatch.setattr(app, "PLAYER_BY_ID", by_player)


def shorten_history(app, monkeypatch, pid, keep=2):
    """Keep only the latest rows, since the mock gives everyone at least 3."""
    shorter = dict(app.HISTORY_BY_ID)
    shorter[pid] = shorter[pid].tail(keep)
    monkeypatch.setattr(app, "HISTORY_BY_ID", shorter)


def card_cells(card: str) -> list[str]:
    """Season, median and range from the card's highlighted row, bold removed."""
    marked = next(line for line in card.splitlines() if "▸" in line)
    cells = [c.strip() for c in marked.strip("| ").split("|")]
    return [c.replace("**", "").removeprefix("▸ ") for c in cells]


@pytest.mark.parametrize("h", [1, 2, 3])
def test_explanation_numbers_match_the_card(app, h):
    pid = first_player_id(app)
    season, median, likely = card_cells(app.make_card(pid, h))
    text = app.make_explanation(pid, h)
    assert f"(by {season})" in text
    assert f"middle estimate is {median}," in text
    assert f"likely range is {likely}." in text
    # Change wording must agree with the table's Change column
    change = app.fmt_change(app.PLAYER_BY_ID[pid][f"change_{h}"])
    if change.lstrip("+-") == "0%":
        assert "about the same as" in text
    else:
        assert f"{'up' if change[0] == '+' else 'down'} {change[1:]} from" in text


def test_explanation_wording_and_limits(app):
    pid = first_player_id(app)
    text = app.make_explanation(pid, 1)
    assert text.startswith(app.EXPLAIN_TITLE)
    assert "aims for the real value to land inside this range 8 times out of 10" in text
    # Limits line is always last, in small text, dated from anchor_date
    anchor = app.FORECASTS_BY_ID[pid].set_index("horizon").loc[1, "anchor_date"]
    valued = app.PLAYER_BY_ID[pid]["value_date"]
    # Premise: the mock dates differ, so the test can tell them apart
    assert anchor != valued
    limits = text.split("\n\n")[-1]
    assert limits.startswith("<small>") and limits.endswith("</small>")
    assert f"transfers after {anchor.day} {anchor:%b %Y}." in limits
    assert f"{valued:%b %Y}" not in limits
    assert "Transfermarkt valuations" in limits


def test_change_is_measured_from_the_dated_latest_valuation(app):
    pid = first_player_id(app)
    text = app.make_explanation(pid, 1)
    d = app.PLAYER_BY_ID[pid]["value_date"]
    current = app.fmt_eur(app.PLAYER_BY_ID[pid]["current_value_eur"])
    assert f"from {current} at the latest valuation ({d.day} {d:%b %Y})." in text
    assert "today" not in text


def test_change_and_limits_fall_back_without_dates(app):
    assert app.change_text(2e6, 1e6, None) == ", up 100% from €1.0m at the latest valuation"
    assert app.change_text(1e6, 1e6, pd.NaT).endswith("€1.0m at the latest valuation")
    assert app.fmt_day(pd.NaT) is None and app.fmt_day(pd.Timestamp("2026-07-01")) == "1 Jul 2026"


def test_how_it_works_states_its_limits(app):
    """A real model now ships, so the text must name its weaknesses, not just its scores."""
    shown = [b for b in app.demo.blocks.values()
             if isinstance(b, gr.Markdown) and b.value == app.HOW_IT_WORKS]
    assert len(shown) == 1, "the How it works text is not rendered exactly once"
    panel = next(b for b in app.demo.blocks.values()
                 if isinstance(b, gr.Column) and "pvf-how" in (b.elem_classes or []))
    # It starts closed behind the HOW IT WORKS button on the Home screen
    assert panel.visible is False
    text = app.HOW_IT_WORKS
    # The old placeholder promised results that have since arrived
    assert "Results will appear here" not in text
    assert "8 times out of 10" in text
    # Every honest limit from decisions #48 and #49 has to survive an edit
    lowered = text.lower()
    assert "lean upward" in lowered, "the upward bias must stay stated"
    assert "cannot see injuries" in lowered
    assert "extrapolation" in lowered
    assert "cross-validation" in lowered


def test_explanation_describes_how_the_range_widens(app):
    pid = first_player_id(app)
    f = app.FORECASTS_BY_ID[pid].set_index("horizon")
    near = f.loc[1, "p90_eur"] - f.loc[1, "p10_eur"]
    far = f.loc[3, "p90_eur"] - f.loc[3, "p10_eur"]
    text = app.make_explanation(pid, 2)
    assert "widens" in text
    assert f"from {app.fmt_eur(near)} at 1 season ahead to {app.fmt_eur(far)} at 3 seasons" in text


def test_spread_text_names_widening_narrowing_and_steady(app):
    from types import SimpleNamespace

    def band(lo, hi):
        return SimpleNamespace(p10_eur=lo, p90_eur=hi)

    assert "50% wider" in app.spread_text({1: band(10e6, 20e6), 3: band(10e6, 25e6)})
    assert "3.0 times as wide" in app.spread_text({1: band(10e6, 20e6), 3: band(10e6, 40e6)})
    assert "narrows" in app.spread_text({1: band(10e6, 20e6), 3: band(10e6, 15e6)})
    assert "about the same width" in app.spread_text({1: band(10e6, 20e6), 3: band(10e6, 20.2e6)})
    # A missing end horizon means there is nothing honest to compare
    assert app.spread_text({1: band(10e6, 20e6)}) == ""


@pytest.mark.parametrize("pid", [None, -1, "abc", float("nan")])
def test_explanation_without_a_known_player_prompts(app, pid):
    assert app.make_explanation(pid, 1) == f"{app.EXPLAIN_TITLE}\n\n{app.EXPLAIN_PROMPT}"


def test_explanation_without_a_forecast_says_so(app, monkeypatch):
    pid = first_player_id(app)
    without = {k: v for k, v in app.FORECASTS_BY_ID.items() if k != pid}
    monkeypatch.setattr(app, "FORECASTS_BY_ID", without)
    assert app.make_explanation(pid, 1) == f"{app.EXPLAIN_TITLE}\n\n{app.EXPLAIN_NO_FORECAST}"


def test_wide_range_is_flagged_with_the_width_reason(app, monkeypatch):
    pid = first_player_id(app)
    widen_band(app, monkeypatch, pid, horizon=1)
    warning = app.make_explanation(pid, 1).split("\n\n")[1]
    assert warning.startswith("> ⚠ **Low confidence:** this range is among the widest 20%")
    assert "past valuation" not in warning
    # The band was only widened at 1 season, so 2 seasons stays clear
    assert "Low confidence" not in app.make_explanation(pid, 2)


def test_short_history_is_flagged_with_the_history_reason(app, monkeypatch):
    pid = first_player_id(app)
    shorten_history(app, monkeypatch, pid, keep=2)
    warning = app.make_explanation(pid, 1).split("\n\n")[1]
    assert warning.startswith("> ⚠ **Low confidence:** this player has only 2 past valuations")
    assert "widest" not in warning


def test_both_reasons_are_named_together(app, monkeypatch):
    pid = first_player_id(app)
    widen_band(app, monkeypatch, pid, horizon=1)
    shorten_history(app, monkeypatch, pid, keep=1)
    warning = app.make_explanation(pid, 1).split("\n\n")[1]
    assert "widest 20%" in warning and "only 1 past valuation." in warning


def test_ties_at_the_cutoff_are_not_flagged(app):
    cutoff = app.WIDTH_CUTOFF[1]
    # Float noise around a shared width must not pick arbitrary players
    assert app.low_confidence_reasons(cutoff * (1 + 1e-12), 10, 1) == []
    assert app.low_confidence_reasons(cutoff * 1.01, 10, 1) == ["width"]
    assert app.low_confidence_reasons(float("nan"), 0, 1) == []


def descendants(block) -> list:
    """Every component nested under a layout block, at any depth."""
    found = []
    for child in getattr(block, "children", []):
        found.append(child)
        found.extend(descendants(child))
    return found


@pytest.mark.parametrize("value, label", [
    (0, "€0"), (500_000, "€500k"), (2_500_000, "€2.5m"), (10_000_000, "€10m"),
    (250_000_000, "€250m")])
def test_money_label_uses_euro_shorthand(app, value, label):
    assert app.money_label(value) == label


@pytest.mark.parametrize("top", [0.4e6, 3.2e6, 18e6, 463e6])
def test_money_ticks_cover_the_top_with_round_steps(app, top):
    values, labels = app.money_ticks(top)
    # Ticks start one step above zero and just reach past the top
    assert values[0] > 0 and values[-1] >= top and values[-2] < top
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    assert len(steps) == 1
    assert labels == [app.money_label(v) for v in values]
    assert 3 <= len(values) <= 7


def test_money_ticks_survive_missing_values(app):
    assert app.money_ticks(float("nan"))[0][-1] >= 1e6
    assert app.money_ticks(0)[0][-1] >= 1e6


def test_chart_hover_matches_the_card(app):
    pid = first_player_id(app)
    fig = app.make_chart(pid)
    history = fig.data[1]
    assert list(history.customdata) == [app.fmt_eur(v) for v in history.y]
    median = fig.data[2]
    f = app.FORECASTS_BY_ID[pid]
    card = app.make_card(pid, 1)
    for (mid, likely), row in zip(median.customdata, f.itertuples()):
        assert mid == app.fmt_eur(row.p50_eur) and likely == app.fmt_range(row.p10_eur, row.p90_eur)
        assert mid in card and likely in card
    # The connector from today and the band never add hover rows
    assert fig.data[0].hoverinfo == "skip" and fig.data[3].hoverinfo == "skip"


def test_chart_axis_legend_and_background(app):
    fig = app.make_chart(first_player_id(app))
    layout = fig.layout
    assert all(t.startswith("€") for t in layout.yaxis.ticktext)
    assert layout.yaxis.range[1] >= max(max(t.y) for t in fig.data)
    # Legend sits below the plot, horizontally, with short full labels
    assert layout.legend.orientation == "h" and layout.legend.y < 0
    assert [t.name for t in fig.data if t.showlegend is not False] == [
        "Likely range", "Market value", "Median forecast"]
    assert layout.plot_bgcolor == layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_browser_tab_title(app):
    assert app.demo.title == "Player Value Forecaster"


def test_helper_text_on_look_ahead_sort_and_table(app):
    assert app.horizon.info == "How many seasons ahead the forecast looks"
    assert app.sort_by.info
    notes = [b.value for b in app.demo.blocks.values() if isinstance(b, gr.Markdown)
             and "pvf-note" in (b.elem_classes or [])]
    assert notes == ["⚠ low-confidence forecast. Likely range: where the model aims for the "
                     "real value to land 8 times out of 10."]


def test_every_control_has_a_visible_label(app):
    for comp in app.filters:
        assert comp.label and comp.show_label is not False
        # Sentence case: only the first letter may be a capital
        assert comp.label[0].isupper() and comp.label[1:] == comp.label[1:].lower()


def test_search_screen_holds_every_filter_tile(app):
    """Filters now live on their own screen as tiles, not behind an accordion."""
    screen = next(b for b in app.demo.blocks.values()
                  if isinstance(b, gr.Column) and "pvf-search" in (b.elem_classes or []))
    inside = descendants(screen)
    for comp in (app.position, app.league, app.country, app.nation, app.age_min,
                 app.age_max, app.search_box, app.horizon, app.go_btn, app.clear_btn):
        assert comp in inside, f"{comp.label!r} is missing from the Search screen"
    # Results controls belong to the Results screen, so SEARCH cannot be skipped
    for comp in (app.sort_by, app.cards, app.more_btn):
        assert comp not in inside
