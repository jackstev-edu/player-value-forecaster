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
    view, ids, count, _chart, _card, _selected = call_filter(app)
    assert len(view) <= app.MAX_ROWS
    assert len(ids) == len(view)
    # The id list must line up row for row with what the table shows
    assert view["Player"].tolist() == [app.PLAYER_BY_ID[pid]["name"] for pid in ids]
    assert "players" in count


def test_impossible_filters_give_empty_table_and_advice(app):
    view, ids, count, chart, _card, _selected = call_filter(
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
    # Chart is left alone; the card is redrawn for the chosen horizon
    assert chart == gr.skip() and card == app.make_card(pid, 1)
    assert selected == pid


def test_selection_outside_the_shown_rows_is_kept(app):
    cheapest = int(app.PLAYERS.sort_values("current_value_eur")["player_id"].iloc[0])
    _view, ids, _count, chart, card, selected = call_filter(app, selected=cheapest)
    # Premise of the case: this player ranks below the visible top rows
    assert cheapest not in ids
    assert chart == gr.skip() and card == app.make_card(cheapest, 1)
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
    chart, card, selected = app.on_select(ids, "1 season", FakeSelect(2))
    assert isinstance(chart, go.Figure)
    assert selected == ids[2]
    assert app.PLAYER_BY_ID[ids[2]]["name"] in card


def test_on_select_guards_empty_and_out_of_range_rows(app):
    chart, card, selected = app.on_select([], "1 season", FakeSelect(0))
    assert chart is None and card == app.CARD_STALE and selected is None
    chart, card, selected = app.on_select([1, 2], "1 season", FakeSelect(9))
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
    _view, ids, _count, _chart, _card, _selected = call_filter(app, horizon=horizon, sort_by=sort_by)
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
    _chart, card, _selected = app.on_select(ids, "3 seasons", FakeSelect(0))
    assert card == app.make_card(ids[0], 3)
