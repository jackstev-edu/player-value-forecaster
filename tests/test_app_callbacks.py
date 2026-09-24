"""Callback level tests for the Gradio app, run without launching it."""
import importlib.util
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"

# A nationality the bundle cannot contain, so these filters match nobody
ABSENT_NATION = "Nowhereland"


@pytest.fixture(scope="module")
def app():
    """Import app.py by path so the Space folder stays self-contained."""
    spec = importlib.util.spec_from_file_location("pvf_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_filter(app, search=None, leagues=None, countries=None, nations=None, positions=None,
                age_min=None, age_max=None, horizon=None, sort_by=None, selected=None):
    """Call the filter callback with the argument order the listener uses."""
    return app.filter_players(search, leagues, countries, nations, positions,
                              age_min, age_max, horizon, sort_by, selected)


def with_accented_name(app, monkeypatch, name="Kylian Mbappé"):
    """Rename the first player so accent handling can be tested on mock data."""
    players = app.PLAYERS.copy()
    players.loc[players.index[0], "name"] = name
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(players, app.FORECASTS))
    return int(players["player_id"].iloc[0])


class FakeSelect:
    """Stand in for gr.SelectData, which only needs an index here."""

    def __init__(self, row):
        self.index = [row, 0]


def first_player_id(app) -> int:
    return int(app.PLAYERS["player_id"].iloc[0])


def test_importing_the_app_does_not_launch_it(app):
    assert isinstance(app.demo, gr.Blocks)
    assert not app.demo.is_running


def test_no_filters_returns_capped_table_with_matching_ids(app):
    view, ids, count, _chart, _card, _explain, _selected = call_filter(app)
    assert len(view) <= app.MAX_ROWS
    assert len(ids) == len(view)
    # The id list must line up row for row with what the table shows
    shown = [name.removeprefix(app.FLAG) for name in view["Player"]]
    assert shown == [app.PLAYER_BY_ID[pid]["name"] for pid in ids]
    assert "players" in count


def test_impossible_filters_give_empty_table_and_advice(app):
    view, ids, count, chart, _card, _explain, _selected = call_filter(
        app, leagues=[app.PLAYERS["league_name"].iloc[0]], nations=[ABSENT_NATION])
    # Headers are pinned by name so a dropped or renamed column fails here
    assert list(view.columns) == ["Player", "Age", "Position", "Club", "League", "Nationality",
                                  "Value", "Change", "Likely range"]
    assert app.COLUMNS == list(view.columns)
    assert len(view) == 0 and ids == []
    assert "No players match" in count
    assert "League" in count and "Nationality" in count
    assert chart is None


def test_no_match_message_names_the_age_range(app):
    _view, _ids, count, _chart, _card, _explain, _selected = call_filter(
        app, nations=[ABSENT_NATION], age_min=30, age_max=31)
    assert "currently 30 to 31" in count


def test_swapped_age_bounds_match_ordered_bounds(app):
    swapped = call_filter(app, age_min=30, age_max=24)
    ordered = call_filter(app, age_min=24, age_max=30)
    pd.testing.assert_frame_equal(swapped[0], ordered[0])
    assert swapped[1] == ordered[1]
    assert swapped[2] == ordered[2]


def test_all_none_inputs_do_not_raise(app):
    view, ids, count, _chart, card, _explain, selected = call_filter(app)
    assert len(view) > 0 and ids and count
    assert card == app.CARD_PROMPT and selected is None


def test_selection_survives_matching_filters(app):
    pid = first_player_id(app)
    league = app.PLAYER_BY_ID[pid]["league_name"]
    _view, _ids, _count, chart, card, _explain, selected = call_filter(
        app, leagues=[league], selected=pid)
    # Chart is left alone; the card is redrawn for the chosen horizon
    assert chart == gr.skip() and card == app.make_card(pid, 1)
    assert selected == pid


def test_selection_outside_the_shown_rows_is_kept(app):
    cheapest = int(app.PLAYERS.sort_values("current_value_eur")["player_id"].iloc[0])
    _view, ids, _count, chart, card, _explain, selected = call_filter(app, selected=cheapest)
    # Premise of the case: this player ranks below the visible top rows
    assert cheapest not in ids
    assert chart == gr.skip() and card == app.make_card(cheapest, 1)
    assert selected == cheapest


def test_selection_cleared_when_filtered_out(app):
    pid = first_player_id(app)
    _view, _ids, _count, chart, card, _explain, selected = call_filter(
        app, nations=[ABSENT_NATION], selected=pid)
    assert chart is None
    assert card == app.CARD_GONE
    assert selected is None


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


def test_on_select_returns_the_clicked_player(app):
    ids = call_filter(app)[1]
    *_, chart, card, _explain, selected = app.on_select(ids, "1 season", FakeSelect(2))
    assert isinstance(chart, go.Figure)
    assert selected == ids[2]
    assert app.PLAYER_BY_ID[ids[2]]["name"] in card


def test_on_select_guards_empty_and_out_of_range_rows(app):
    *_, chart, card, _explain, selected = app.on_select([], "1 season", FakeSelect(0))
    assert chart is None and card == app.CARD_STALE and selected is None
    *_, chart, card, _explain, selected = app.on_select([1, 2], "1 season", FakeSelect(9))
    assert chart is None and card == app.CARD_STALE and selected is None


def test_fmt_eur_formats_and_handles_missing(app):
    assert app.fmt_eur(12_500_000) == "€12.5m"
    assert app.fmt_eur(750_000) == "€750k"
    assert app.fmt_eur(float("nan")) == "unknown"
    assert app.fmt_eur(None) == "unknown"


def test_footer_falls_back_when_manifest_is_empty(app):
    html = app.make_footer({})
    assert "Model unknown, built unknown" in html
    assert app.make_footer({"created_at": "not a date"}).count("unknown") == 2


def test_every_filter_path_returns_seven_outputs(app):
    pid = first_player_id(app)
    league = app.PLAYER_BY_ID[pid]["league_name"]
    cases = [
        call_filter(app),
        call_filter(app, selected=pid),
        call_filter(app, leagues=[league], selected=pid),
        call_filter(app, nations=[ABSENT_NATION]),
        call_filter(app, nations=[ABSENT_NATION], selected=pid),
        call_filter(app, age_min=45, age_max=15, selected=pid),
        call_filter(app, positions=[app.PLAYERS["position"].iloc[0]], age_min=None, age_max=None),
    ]
    for outputs in cases:
        assert len(outputs) == 7


@pytest.mark.parametrize("query", ["mbappe", "MBAPPE", "Mbappé", "mBaPpÉ", "  kylian   mbap "])
def test_search_ignores_case_and_accents(app, monkeypatch, query):
    pid = with_accented_name(app, monkeypatch)
    _view, ids, count, _chart, _card, _explain, _selected = call_filter(app, search=query)
    assert ids == [pid]
    assert "**1 player** matches" in count


def test_search_folds_letters_nfkd_keeps_whole(app):
    assert app.normalise_name("Martin Ødegaard") == "martin odegaard"
    assert app.normalise_name("Łukasz Fabiański") == "lukasz fabianski"
    assert app.normalise_name(None) == ""


def test_unmatched_search_names_the_search_in_the_advice(app):
    view, ids, count, _chart, _card, _explain, _selected = call_filter(app, search="zzqx")
    assert len(view) == 0 and ids == []
    assert "No players match" in count
    assert 'Search "zzqx"' in count


def test_blank_search_is_no_filter(app):
    assert call_filter(app, search="   ")[1] == call_filter(app)[1]


@pytest.mark.parametrize("selected", [None, "first"])
def test_reset_returns_defaults_and_matching_outputs(app, selected):
    pid = first_player_id(app) if selected else None
    out = app.reset_filters(pid)
    defaults = app.default_filters()
    n = len(defaults)
    assert list(out[:n]) == defaults
    assert len(out) == n + 7
    expected = app.filter_players(*defaults, pid)
    pd.testing.assert_frame_equal(out[n], expected[0])
    assert out[n + 1:] == expected[1:]


def without_forecast(app, monkeypatch, pid, horizon):
    """Drop one player's forecast for one horizon and rebuild derived columns."""
    f = app.FORECASTS
    kept = f[~((f["player_id"] == pid) & (f["horizon"] == horizon))]
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(app.PLAYERS, kept))


def top_value_id(app) -> int:
    return int(app.PLAYERS.sort_values("current_value_eur", ascending=False)["player_id"].iloc[0])


@pytest.mark.parametrize("horizon", ["1 season", "2 seasons", "3 seasons"])
@pytest.mark.parametrize("sort_by, column, descending", [
    ("Current value", "current_value_eur", True),
    ("Biggest predicted rise", "change", True),
    ("Biggest predicted fall", "change", False),
    ("Most uncertain", "width", True),
])
def test_sort_orders_on_underlying_numbers(app, sort_by, column, descending, horizon):
    h = app.HORIZONS[horizon]
    col = column if column == "current_value_eur" else f"{column}_{h}"
    _view, ids, _count, _chart, _card, _explain, _selected = call_filter(app, horizon=horizon, sort_by=sort_by)
    numbers = app.PLAYERS.set_index("player_id").loc[ids, col]
    assert numbers.notna().all()
    # Every adjacent pair must respect the requested direction
    pairs = list(zip(numbers, numbers[1:]))
    assert all((a >= b) if descending else (a <= b) for a, b in pairs)
    # First row must be the true extreme across every matched player
    extreme = app.PLAYERS[col].max() if descending else app.PLAYERS[col].min()
    assert numbers.iloc[0] == extreme


def test_rise_order_is_numeric_not_text(app):
    view = call_filter(app, sort_by="Biggest predicted rise")[0]
    # Text sorting would rank "+9%" above "+18%"; numbers must not
    parsed = view["Change"].str.rstrip("%").astype(float).tolist()
    assert parsed == sorted(parsed, reverse=True)


def test_horizon_changes_the_change_column(app):
    pid = first_player_id(app)
    name = app.PLAYER_BY_ID[pid]["name"]
    seen = {}
    for label, h in app.HORIZONS.items():
        view, ids, *_ = call_filter(app, search=name, horizon=label)
        row = ids.index(pid)
        row_data = app.PLAYERS.set_index("player_id").loc[pid]
        assert view["Change"].iloc[row] == app.fmt_change(row_data[f"change_{h}"])
        assert view["Likely range"].iloc[row] == app.fmt_range(row_data[f"p10_{h}"],
                                                               row_data[f"p90_{h}"])
        seen[label] = view["Change"].iloc[row]
    assert len(set(seen.values())) > 1


def test_change_and_range_formats(app):
    assert app.fmt_change(0.18) == "+18%"
    assert app.fmt_change(-0.052) == "-5%"
    assert app.fmt_change(float("nan")) == "n/a"
    assert app.fmt_range(8_000_000, 15_000_000) == "€8.0m to €15.0m"
    assert app.fmt_range(None, 1.0) == "n/a"


@pytest.mark.parametrize("sort_by", ["Current value", "Biggest predicted rise",
                                     "Biggest predicted fall", "Most uncertain"])
def test_missing_forecast_sorts_last_and_shows_na(app, monkeypatch, sort_by):
    pid = top_value_id(app)
    without_forecast(app, monkeypatch, pid, horizon=2)
    # Lift the row cap so the true last place is visible in the table
    monkeypatch.setattr(app, "MAX_ROWS", len(app.PLAYERS))
    view, ids, *_ = call_filter(app, horizon="2 seasons", sort_by=sort_by)
    assert ids[-1] == pid
    assert view["Change"].iloc[-1] == "n/a"
    assert view["Likely range"].iloc[-1] == "n/a"
    assert (view["Change"].iloc[:-1] != "n/a").all()
    # Other horizons still have this forecast, so it ranks normally there
    if sort_by == "Current value":
        assert call_filter(app, horizon="1 season", sort_by=sort_by)[1][0] == pid


def test_unknown_sort_and_horizon_fall_back_to_defaults(app):
    odd = call_filter(app, horizon="9 seasons", sort_by="Alphabetical")
    default = call_filter(app, horizon="1 season", sort_by="Current value")
    pd.testing.assert_frame_equal(odd[0], default[0])
    assert odd[1] == default[1]


@pytest.mark.parametrize("h", [1, 2, 3])
def test_card_highlights_only_the_chosen_horizon(app, h):
    pid = first_player_id(app)
    card = app.make_card(pid, h)
    marked = [line for line in card.splitlines() if "▸" in line]
    assert len(marked) == 1
    target = app.FORECASTS_BY_ID[pid].set_index("horizon").loc[h, "target_date"]
    assert f"{target:%b %Y}" in marked[0]


def test_on_select_uses_the_chosen_horizon(app):
    ids = call_filter(app)[1]
    *_, _chart, card, _explain, _selected = app.on_select(ids, "3 seasons", FakeSelect(0))
    assert card == app.make_card(ids[0], 3)


def test_initial_view_opens_the_featured_player(app):
    view, ids, count, chart, card, _explain, selected = app.initial_view()
    assert selected == app.FEATURED_ID == top_value_id(app)
    assert isinstance(chart, go.Figure)
    assert app.PLAYER_BY_ID[selected]["name"] in card
    # Table and count match a plain default filter run
    expected = app.filter_players(*app.default_filters(), None)
    pd.testing.assert_frame_equal(view, expected[0])
    assert ids == expected[1] and count == expected[2]


def test_load_is_not_a_filter_trigger(app):
    # demo.load must only run initial_view, never filter_players as well
    load_fns = [fn.fn.__name__ for fn in app.demo.fns.values()
                if any(t[1] == "load" for t in fn.targets)]
    assert load_fns == ["initial_view"]


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


@pytest.mark.parametrize("index", range(4))
def test_clicking_an_example_opens_that_player(app, index):
    label, pid, name = app.EXAMPLES[index]
    out = app.open_example(name, pid)
    n = len(app.default_filters())
    assert len(out) == n + 7
    # Search holds the name and every other filter is back to default
    assert out[0] == name
    assert list(out[1:n]) == app.default_filters()[1:]
    view, ids, _count, chart, card, _explain, selected = out[n:]
    # A flagged example shows the prefix, but the search itself used the plain name
    assert pid in ids and {n.removeprefix(app.FLAG) for n in view["Player"]} == {name}
    assert selected == pid
    assert isinstance(chart, go.Figure) and name in card


def test_example_with_repeated_name_opens_the_exact_id(app, monkeypatch):
    first, second = app.PLAYERS["player_id"].iloc[:2].astype(int).tolist()
    players = app.PLAYERS.copy()
    players.loc[players.index[:2], "name"] = "Same Name"
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(players, app.FORECASTS))
    for pid in (first, second):
        # Hidden Number may deliver a float, which must still resolve
        out = app.open_example("Same Name", float(pid))
        assert sorted(out[-6]) == sorted([first, second])
        assert out[-1] == pid


def test_example_with_unknown_id_shows_missing_card(app):
    out = app.open_example("anything", None)
    assert out[-4] is None and out[-3] == app.CARD_MISSING and out[-1] is None
    assert app.EXPLAIN_PROMPT in out[-2]


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
    # Limits line is always last, in small text, dated from value_date
    d = app.PLAYER_BY_ID[pid]["value_date"]
    limits = text.split("\n\n")[-1]
    assert limits.startswith("<small>") and limits.endswith("</small>")
    assert f"transfers after {d.day} {d:%b %Y}." in limits
    assert "Transfermarkt valuations" in limits


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


def test_ordinary_player_is_not_flagged(app):
    pid = first_player_id(app)
    assert app.history_count(pid) >= app.MIN_HISTORY_ROWS
    assert "Low confidence" not in app.make_explanation(pid, 1)
    view, ids, *_ = call_filter(app, search=app.PLAYER_BY_ID[pid]["name"])
    assert not view["Player"].iloc[ids.index(pid)].startswith(app.FLAG)


def test_ties_at_the_cutoff_are_not_flagged(app):
    cutoff = app.WIDTH_CUTOFF[1]
    # Float noise around a shared width must not pick arbitrary players
    assert app.low_confidence_reasons(cutoff * (1 + 1e-12), 10, 1) == []
    assert app.low_confidence_reasons(cutoff * 1.01, 10, 1) == ["width"]
    assert app.low_confidence_reasons(float("nan"), 0, 1) == []


def test_flag_prefix_marks_only_flagged_players(app, monkeypatch):
    wide, short = app.PLAYERS["player_id"].iloc[:2].astype(int).tolist()
    widen_band(app, monkeypatch, wide, horizon=1)
    shorten_history(app, monkeypatch, short, keep=2)
    monkeypatch.setattr(app, "MAX_ROWS", len(app.PLAYERS))
    view, ids, *_ = call_filter(app, horizon="1 season")
    flagged = {pid for pid, name in zip(ids, view["Player"]) if name.startswith(app.FLAG)}
    assert flagged == {wide, short}
    # Width flags follow the horizon; the history flag applies to all of them
    view, ids, *_ = call_filter(app, horizon="2 seasons")
    flagged = {pid for pid, name in zip(ids, view["Player"]) if name.startswith(app.FLAG)}
    assert flagged == {short}


def test_search_finds_flagged_players_by_plain_name(app, monkeypatch):
    pid = first_player_id(app)
    name = app.PLAYER_BY_ID[pid]["name"]
    shorten_history(app, monkeypatch, pid, keep=2)
    view, ids, *_ = call_filter(app, search=name)
    assert pid in ids
    assert view["Player"].iloc[ids.index(pid)] == app.FLAG + name
    # An example click feeds the plain name and still lands on the player
    out = app.open_example(name, pid)
    assert pid in out[len(app.default_filters()) + 1] and out[-1] == pid


def test_horizon_change_rerenders_card_and_explanation(app):
    pid = first_player_id(app)
    seen = set()
    for label, h in app.HORIZONS.items():
        _view, _ids, _count, chart, card, explain, selected = call_filter(
            app, horizon=label, selected=pid)
        assert chart == gr.skip() and selected == pid
        # Neither may be skipped, or the old horizon would stay on screen
        assert card == app.make_card(pid, h)
        assert explain == app.make_explanation(pid, h)
        seen.add(explain)
    assert len(seen) == len(app.HORIZONS)


def test_selecting_a_row_explains_the_chosen_horizon(app):
    ids = call_filter(app)[1]
    out = app.on_select(ids, "3 seasons", FakeSelect(0))
    # Table, ids and count are left alone by a row click
    assert out[:3] == (gr.skip(), gr.skip(), gr.skip())
    assert out[5] == app.make_explanation(ids[0], 3)


def test_every_callback_path_returns_seven_results(app):
    pid = first_player_id(app)
    ids = call_filter(app)[1]
    n = len(app.default_filters())
    results = [
        call_filter(app), call_filter(app, selected=pid),
        call_filter(app, nations=[ABSENT_NATION], selected=pid),
        app.initial_view(),
        app.on_select(ids, "1 season", FakeSelect(0)),
        app.on_select([], "1 season", FakeSelect(0)),
        app.on_select(ids, "1 season", None),
        app.reset_filters(None)[n:], app.reset_filters(pid)[n:],
        app.open_example("anything", pid)[n:], app.open_example("anything", None)[n:],
    ]
    for out in results:
        assert len(out) == 7
        assert app.EXPLAIN_TITLE in out[5]
    # Listener output lists must match what each callback returns
    outputs = {fn.api_name: len(fn.outputs) for fn in app.demo.fns.values()}
    assert outputs["filter_players"] == outputs["initial_view"] == outputs["select_player"] == 7
    assert outputs["reset_filters"] == n + 7
    # Examples register a fill step too; the one running open_example has n + 7
    assert n + 7 in [count for name, count in outputs.items() if name.startswith("open_example")]


def descendants(block) -> list:
    """Every component nested under a layout block, at any depth."""
    found = []
    for child in getattr(block, "children", []):
        found.append(child)
        found.extend(descendants(child))
    return found


def accordion(app, label):
    return next(b for b in app.demo.blocks.values()
                if isinstance(b, gr.Accordion) and b.label == label)


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


def test_secondary_filters_sit_in_a_collapsed_accordion(app):
    more = accordion(app, "More filters")
    assert more.open is False
    inside = descendants(more)
    for comp in (app.position, app.league, app.country, app.nation, app.age_min, app.age_max):
        assert comp in inside
    # Search, look ahead, sort and clear stay visible outside it
    for comp in (app.search, app.horizon, app.sort_by, app.reset, app.count):
        assert comp not in inside


def test_every_control_has_a_visible_label(app):
    for comp in app.filters:
        assert comp.label and comp.show_label is not False
        # Sentence case: only the first letter may be a capital
        assert comp.label[0].isupper() and comp.label[1:] == comp.label[1:].lower()


def test_table_pins_the_player_column(app):
    assert app.table.pinned_columns == 1
    assert app.COLUMNS[0] == "Player"
