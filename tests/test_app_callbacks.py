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
                age_min=None, age_max=None, selected=None):
    """Call the filter callback with the argument order the listener uses."""
    return app.filter_players(search, leagues, countries, nations, positions,
                              age_min, age_max, selected)


def with_accented_name(app, monkeypatch, name="Kylian Mbappé"):
    """Rename the first player so accent handling can be tested on mock data."""
    players = app.PLAYERS.copy()
    players.loc[players.index[0], "name"] = name
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(players.drop(columns="name_key")))
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
    view, ids, count, _chart, _card, _selected = call_filter(app)
    assert len(view) <= app.MAX_ROWS
    assert len(ids) == len(view)
    # The id list must line up row for row with what the table shows
    assert view["Player"].tolist() == [app.PLAYER_BY_ID[pid]["name"] for pid in ids]
    assert "players" in count


def test_impossible_filters_give_empty_table_and_advice(app):
    view, ids, count, chart, _card, _selected = call_filter(
        app, leagues=[app.PLAYERS["league_name"].iloc[0]], nations=[ABSENT_NATION])
    assert list(view.columns) == app.COLUMNS
    assert len(view) == 0 and ids == []
    assert "No players match" in count
    assert "League" in count and "Nationality" in count
    assert chart is None


def test_no_match_message_names_the_age_range(app):
    _view, _ids, count, _chart, _card, _selected = call_filter(
        app, nations=[ABSENT_NATION], age_min=30, age_max=31)
    assert "currently 30 to 31" in count


def test_swapped_age_bounds_match_ordered_bounds(app):
    swapped = call_filter(app, age_min=30, age_max=24)
    ordered = call_filter(app, age_min=24, age_max=30)
    pd.testing.assert_frame_equal(swapped[0], ordered[0])
    assert swapped[1] == ordered[1]
    assert swapped[2] == ordered[2]


def test_all_none_inputs_do_not_raise(app):
    view, ids, count, _chart, card, selected = call_filter(app)
    assert len(view) > 0 and ids and count
    assert card == app.CARD_PROMPT and selected is None


def test_selection_survives_matching_filters(app):
    pid = first_player_id(app)
    league = app.PLAYER_BY_ID[pid]["league_name"]
    _view, _ids, _count, chart, card, selected = call_filter(app, leagues=[league], selected=pid)
    assert chart == gr.skip() and card == gr.skip()
    assert selected == pid


def test_selection_outside_the_shown_rows_is_kept(app):
    cheapest = int(app.PLAYERS.sort_values("current_value_eur")["player_id"].iloc[0])
    _view, ids, _count, chart, card, selected = call_filter(app, selected=cheapest)
    # Premise of the case: this player ranks below the visible top rows
    assert cheapest not in ids
    assert chart == gr.skip() and card == gr.skip()
    assert selected == cheapest


def test_selection_cleared_when_filtered_out(app):
    pid = first_player_id(app)
    _view, _ids, _count, chart, card, selected = call_filter(
        app, nations=[ABSENT_NATION], selected=pid)
    assert chart is None
    assert card == app.CARD_GONE
    assert selected is None


def test_chart_and_card_for_a_real_player(app):
    pid = first_player_id(app)
    fig = app.make_chart(pid)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
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
    chart, card, selected = app.on_select(ids, FakeSelect(2))
    assert isinstance(chart, go.Figure)
    assert selected == ids[2]
    assert app.PLAYER_BY_ID[ids[2]]["name"] in card


def test_on_select_guards_empty_and_out_of_range_rows(app):
    chart, card, selected = app.on_select([], FakeSelect(0))
    assert chart is None and card == app.CARD_STALE and selected is None
    chart, card, selected = app.on_select([1, 2], FakeSelect(9))
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


def test_every_filter_path_returns_six_outputs(app):
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
        assert len(outputs) == 6


@pytest.mark.parametrize("query", ["mbappe", "MBAPPE", "Mbappé", "mBaPpÉ", "  kylian   mbap "])
def test_search_ignores_case_and_accents(app, monkeypatch, query):
    pid = with_accented_name(app, monkeypatch)
    _view, ids, count, _chart, _card, _selected = call_filter(app, search=query)
    assert ids == [pid]
    assert "**1 players**" in count


def test_search_folds_letters_nfkd_keeps_whole(app):
    assert app.normalise_name("Martin Ødegaard") == "martin odegaard"
    assert app.normalise_name("Łukasz Fabiański") == "lukasz fabianski"
    assert app.normalise_name(None) == ""


def test_unmatched_search_names_the_search_in_the_advice(app):
    view, ids, count, _chart, _card, _selected = call_filter(app, search="zzqx")
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
    assert len(out) == n + 6
    expected = app.filter_players(*defaults, pid)
    pd.testing.assert_frame_equal(out[n], expected[0])
    assert out[n + 1:] == expected[1:]
