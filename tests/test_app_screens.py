"""Screen and wiring tests for the four screen layout, run without a browser.

These replace the table era wiring tests. They use the fixed mock bundle, so
they can assert exact players and counts. Anything that only the browser can
show, such as the restored scroll position, is covered by
scripts/ui_probes/back_button.py instead.
"""
import html
import importlib.util
import re
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"
MOCK_BUNDLE = Path(__file__).resolve().parent / "fixtures" / "mock_bundle"

# A nationality the bundle cannot contain, so these filters match nobody
ABSENT_NATION = "Nowhereland"


@pytest.fixture(scope="module")
def app():
    """Import app.py by path so the Space folder stays self-contained."""
    spec = importlib.util.spec_from_file_location("pvf_app_screens", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    # The bundle is read at import, so the override only has to hold while it runs
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PVF_BUNDLE_DIR", str(MOCK_BUNDLE))
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def defaults(app):
    return app.default_filters()


def search_with(app, defaults, **changes):
    """Run SEARCH with the default filters, overriding the named slots."""
    slots = {"search": 0, "leagues": 1, "countries": 2, "nations": 3, "positions": 4,
             "age_min": 5, "age_max": 6, "horizon": 7, "sort_by": 8}
    values = list(defaults)
    for key, value in changes.items():
        values[slots[key]] = value
    return app.run_search(*values)


def visible_screen(outputs, app) -> str:
    """Which screen a navigation output list turns on."""
    on = [name for name, update in zip(app.SCREENS, outputs[:len(app.SCREENS)])
          if update.visible is True]
    assert len(on) == 1, f"expected one visible screen, got {on}"
    return on[0]


def parts(app, outputs):
    """The named pieces of a run_search or load_more result."""
    n = len(app.SCREENS) + 1
    return {"cards": outputs[n], "count": outputs[n + 1], "ids": outputs[n + 2],
            "shown": outputs[n + 3], "more": outputs[n + 4]}


def card_count(markup: str) -> int:
    return markup.count('class="pvf-card"')


def card_ids(markup: str) -> list[int]:
    """Player ids in the order the cards are drawn."""
    return [int(chunk.split('"')[0])
            for chunk in markup.split('data-pid="')[1:]]


# Screen switching

def test_show_turns_on_exactly_one_screen(app):
    for name in app.SCREENS:
        out = app.show(name)
        assert visible_screen(out, app) == name
        assert out[-1] == name


def test_back_walks_one_screen_at_a_time(app):
    assert visible_screen(app.go_back("player"), app) == "results"
    assert visible_screen(app.go_back("results"), app) == "search"
    assert visible_screen(app.go_back("search"), app) == "home"
    # Home is the floor, so Back there cannot leave the app by itself
    assert visible_screen(app.go_back("home"), app) == "home"


def test_unknown_screen_falls_back_to_home(app):
    assert visible_screen(app.go_back("nonsense"), app) == "home"


def test_search_opens_the_results_screen(app, defaults):
    assert visible_screen(search_with(app, defaults), app) == "results"


def test_opening_a_player_opens_the_player_screen(app, defaults):
    ids = parts(app, search_with(app, defaults))["ids"]
    assert visible_screen(app.open_player_screen(ids[0], app.DEFAULT_HORIZON), app) == "player"


# Featured cards

def test_featured_cards_use_the_example_rules(app):
    markup = app.featured_html(app.DEFAULT_H)
    assert card_count(markup) == len(app.EXAMPLES)
    assert card_ids(markup) == [pid for _label, pid, _name in app.EXAMPLES]
    for label, _pid, _name in app.EXAMPLES:
        assert html.escape(label) in markup


def test_featured_card_opens_that_exact_player(app):
    for _label, pid, _name in app.EXAMPLES:
        out = app.open_player_screen(pid, app.DEFAULT_HORIZON)
        assert visible_screen(out, app) == "player"
        head = out[len(app.SCREENS) + 1]
        assert f'data-pid="{pid}"' in head
        assert out[-1] == pid
        assert app.PLAYER_BY_ID[pid]["name"] in head


def test_opening_an_unknown_id_changes_nothing(app):
    for bad in (-1, None, "not a number"):
        out = app.open_player_screen(bad, app.DEFAULT_HORIZON)
        assert len(out) == app.PLAYER_OUTPUTS
        assert all(isinstance(o, type(gr.skip())) for o in out)


# SEARCH applies the filters

def test_search_with_no_filters_returns_everyone(app, defaults):
    got = parts(app, search_with(app, defaults))
    assert len(got["ids"]) == len(app.PLAYERS)
    assert got["shown"] == app.PAGE
    assert card_count(got["cards"]) == app.PAGE


def test_each_filter_narrows_to_matching_players(app, defaults):
    for key, column in (("leagues", "league_name"), ("countries", "league_country"),
                        ("nations", "nationality"), ("positions", "position")):
        choice = app.options(column)[0]
        got = parts(app, search_with(app, defaults, **{key: [choice]}))
        assert got["ids"], f"{column}={choice} matched nobody"
        assert all(app.PLAYER_BY_ID[int(p)][column] == choice for p in got["ids"])


def test_age_filter_keeps_only_that_range(app, defaults):
    got = parts(app, search_with(app, defaults, age_min=20, age_max=22))
    assert got["ids"]
    assert all(20 <= app.PLAYER_BY_ID[int(p)]["age"] <= 22 for p in got["ids"])


def test_swapped_age_bounds_are_ordered(app, defaults):
    wide = parts(app, search_with(app, defaults, age_min=20, age_max=22))
    swapped = parts(app, search_with(app, defaults, age_min=22, age_max=20))
    assert swapped["ids"] == wide["ids"]


def test_search_ignores_case_and_accents(app, defaults, monkeypatch):
    players = app.PLAYERS.copy()
    players.loc[players.index[0], "name"] = "Kylian Mbappé"
    monkeypatch.setattr(app, "PLAYERS", app.add_derived_columns(players, app.FORECASTS))
    pid = int(players["player_id"].iloc[0])
    for typed in ("mbappe", "MBAPPE", "Mbappé"):
        got = parts(app, search_with(app, defaults, search=typed))
        assert pid in [int(p) for p in got["ids"]], f"{typed!r} did not find the player"


def test_blank_search_filters_nothing(app, defaults):
    assert (parts(app, search_with(app, defaults, search="   "))["ids"]
            == parts(app, search_with(app, defaults))["ids"])


def test_no_match_gives_advice_naming_the_live_filters(app, defaults):
    got = parts(app, search_with(app, defaults, nations=[ABSENT_NATION], search="zzzz"))
    assert got["ids"] == []
    assert card_count(got["cards"]) == 0
    assert "No players match" in got["count"]
    assert "Nationality" in got["count"] and 'Search "zzzz"' in got["count"]
    # Nothing left to page through, so the button stays away
    assert got["more"].visible is False


def test_filters_combine_rather_than_replace(app, defaults):
    league = app.options("league_name")[0]
    position = app.options("position")[0]
    both = parts(app, search_with(app, defaults, leagues=[league], positions=[position]))
    for pid in both["ids"]:
        row = app.PLAYER_BY_ID[int(pid)]
        assert row["league_name"] == league and row["position"] == position


def test_all_none_inputs_do_not_raise(app):
    got = parts(app, app.run_search(None, None, None, None, None, None, None, None, None))
    assert len(got["ids"]) == len(app.PLAYERS)


# Load more

def test_load_more_adds_one_page(app, defaults):
    got = parts(app, search_with(app, defaults))
    assert got["shown"] == app.PAGE
    cards, count, shown, more = app.load_more(got["ids"], got["shown"], app.DEFAULT_HORIZON,
                                              app.DEFAULT_SORT)
    assert shown == app.PAGE * 2
    assert card_count(cards) == app.PAGE * 2
    # The extra page continues the same order, it does not reshuffle
    assert card_ids(cards)[:app.PAGE] == card_ids(got["cards"])
    assert str(shown) in count


def test_load_more_button_hides_on_the_last_page(app, defaults):
    got = parts(app, search_with(app, defaults))
    ids, shown = got["ids"], got["shown"]
    assert got["more"].visible is True
    while shown < len(ids):
        _cards, _count, shown, more = app.load_more(ids, shown, app.DEFAULT_HORIZON,
                                                    app.DEFAULT_SORT)
    assert shown == len(ids)
    assert more.visible is False


def test_load_more_never_runs_past_the_end(app, defaults):
    got = parts(app, search_with(app, defaults, nations=[app.options("nationality")[0]]))
    ids = got["ids"]
    _cards, _count, shown, _more = app.load_more(ids, len(ids), app.DEFAULT_HORIZON,
                                                 app.DEFAULT_SORT)
    assert shown == len(ids)


def test_a_short_result_set_offers_no_load_more(app, defaults, monkeypatch):
    small = app.PLAYERS.head(5)
    monkeypatch.setattr(app, "PLAYERS", small)
    got = parts(app, search_with(app, defaults))
    assert got["shown"] == 5
    assert got["more"].visible is False
    assert "Showing the first" not in got["count"]


# Sort chips

def test_every_sort_chip_orders_on_its_own_column(app, defaults):
    h = app.DEFAULT_H
    for label, (prefix, ascending) in app.SORTS.items():
        got = parts(app, search_with(app, defaults, sort_by=label))
        column = prefix if prefix == "current_value_eur" else f"{prefix}_{h}"
        values = [app.PLAYERS_BY_ID.loc[int(p)][column] for p in got["ids"][:30]]
        clean = [v for v in values if v == v]
        assert clean == sorted(clean, reverse=not ascending), f"{label} came back unordered"


def test_rise_and_fall_are_opposite_ends(app, defaults):
    rise = parts(app, search_with(app, defaults, sort_by="Biggest predicted rise"))["ids"]
    fall_label = app.FALL_LABEL if app.FALL_LABEL in app.SORTS else app.NO_FALL_LABEL
    fall = parts(app, search_with(app, defaults, sort_by=fall_label))["ids"]
    assert rise[:10] != fall[:10]


def test_sorting_keeps_the_same_players(app, defaults):
    base = set(parts(app, search_with(app, defaults))["ids"])
    for label in app.SORTS:
        assert set(parts(app, search_with(app, defaults, sort_by=label))["ids"]) == base


def test_sorting_respects_the_live_filters(app, defaults):
    league = app.options("league_name")[0]
    got = parts(app, search_with(app, defaults, leagues=[league], sort_by="Most uncertain"))
    assert all(app.PLAYER_BY_ID[int(p)]["league_name"] == league for p in got["ids"])


def test_unknown_sort_falls_back_to_the_default(app, defaults):
    assert (parts(app, search_with(app, defaults, sort_by="Nonsense"))["ids"]
            == parts(app, search_with(app, defaults, sort_by=app.DEFAULT_SORT))["ids"])


def test_unknown_horizon_falls_back_to_the_default(app, defaults):
    assert (parts(app, search_with(app, defaults, horizon="99 seasons"))["ids"]
            == parts(app, search_with(app, defaults))["ids"])


# Back keeps what the user had

def test_back_from_player_returns_to_results(app):
    assert visible_screen(app.go_back("player"), app) == "results"


def test_back_does_not_touch_the_cards_or_the_filters(app, defaults):
    """Back only writes navigation outputs, so results and filters survive it."""
    out = app.go_back("player")
    # Back writes only the four screen columns and the remembered name
    assert len(out) == len(app.SCREENS) + 1
    assert all(isinstance(o, gr.Column) for o in out[:len(app.SCREENS)])
    assert out[-1] == "results"


def test_clear_resets_every_filter_tile(app):
    assert app.clear_filters() == app.default_filters()


# Card markup

def test_cards_are_buttons_a_keyboard_can_reach(app, defaults):
    markup = parts(app, search_with(app, defaults))["cards"]
    assert markup.count("<button type=\"button\"") == app.PAGE
    assert markup.count("aria-label=") == app.PAGE
    assert "data-pid=" in markup


def test_card_shows_every_required_field(app, defaults):
    got = parts(app, search_with(app, defaults))
    pid = int(got["ids"][0])
    row = app.PLAYERS_BY_ID.loc[pid]
    data = app.card_data(row, app.DEFAULT_H)
    one = app.card_markup(data)
    assert html.escape(str(row["name"])) in one
    assert html.escape(str(row["sub_position"])) in one
    assert f"Age {int(row['age'])}" in one
    assert html.escape(str(row["club_name"])) in one
    assert app.fmt_eur(row["current_value_eur"]) in one
    assert data["change"] in one and data["arrow"] in one
    assert app.fmt_range(row[f"p10_{app.DEFAULT_H}"], row[f"p90_{app.DEFAULT_H}"]) in one


def test_card_escapes_a_name_that_looks_like_markup(app, monkeypatch):
    row = app.PLAYERS_BY_ID.iloc[0].copy()
    row["name"] = '<img src=x onerror="alert(1)"> & co'
    row["club_name"] = "A & B <FC>"
    one = app.card_markup(app.card_data(row, app.DEFAULT_H))
    # Only the card's own elements may survive; the name cannot introduce a tag
    assert sorted(set(re.findall(r"<\s*([a-zA-Z][\w-]*)", one))) == ["button", "span"]
    assert "&lt;img" in one and "&amp;" in one


def test_card_arrow_agrees_with_the_shown_percent(app):
    h = app.DEFAULT_H
    for _, row in app.PLAYERS.head(60).iterrows():
        data = app.card_data(row, h)
        if data["change"].startswith("+") and data["change"] != "+0%":
            assert data["trend"] == "up"
        elif data["change"].startswith("-") and data["change"] != "-0%":
            assert data["trend"] == "down"
        else:
            assert data["trend"] == "flat"


def test_flagged_players_carry_the_warning_mark(app, monkeypatch):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    h = app.DEFAULT_H
    players = app.PLAYERS.copy()
    wide = players[f"width_{h}"].max() * 10
    players.loc[players["player_id"] == pid, f"width_{h}"] = wide
    monkeypatch.setattr(app, "PLAYERS", players)
    monkeypatch.setattr(app, "PLAYERS_BY_ID", players.set_index("player_id", drop=False))
    data = app.card_data(players[players["player_id"] == pid].iloc[0], h)
    assert data["flag"] is True
    assert "pc-flag" in app.card_markup(data)
    # An ordinary player keeps the mark off
    plain = app.card_data(app.PLAYERS_BY_ID.loc[int(app.PLAYERS["player_id"].iloc[-1])], h)
    assert "pc-flag" not in app.card_markup(plain) or plain["flag"]


def test_player_head_names_the_open_player(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    head = app.player_head(pid, app.DEFAULT_H)
    assert f'data-pid="{pid}"' in head
    assert html.escape(str(app.PLAYER_BY_ID[pid]["name"])) in head
    assert app.seasons_text(app.DEFAULT_H) in head


# Output shapes

def test_every_navigation_path_returns_the_same_shape(app, defaults):
    assert len(app.show("home")) == len(app.SCREENS) + 1
    assert len(app.go_back("results")) == len(app.SCREENS) + 1
    assert len(search_with(app, defaults)) == len(app.SCREENS) + 6
    assert len(app.open_player_screen(int(app.PLAYERS["player_id"].iloc[0]),
                                      app.DEFAULT_HORIZON)) == app.PLAYER_OUTPUTS


def test_player_screen_draws_a_chart_card_and_explanation(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    out = app.open_player_screen(pid, app.DEFAULT_HORIZON)
    _head, chart, card, explain, selected = out[len(app.SCREENS) + 1:]
    assert isinstance(chart, go.Figure)
    assert app.PLAYER_BY_ID[pid]["name"] in card
    assert "What this means" in explain
    assert selected == pid


def test_count_text_matches_what_is_drawn(app, defaults):
    got = parts(app, search_with(app, defaults))
    assert str(len(app.PLAYERS)) in got["count"]
    assert str(got["shown"]) in got["count"]


def test_count_text_uses_the_singular_for_one_player(app, defaults):
    name = str(app.PLAYERS["name"].iloc[0])
    got = parts(app, search_with(app, defaults, search=name))
    if len(got["ids"]) == 1:
        assert "1 player** matches" in got["count"]


# The horizon control follows the bundle

def test_look_ahead_is_offered_only_when_the_bundle_has_choices(app):
    radios = [c for c in app.demo.blocks.values()
              if isinstance(c, gr.Radio) and c.label == "Look ahead"]
    assert len(radios) == 1
    assert radios[0].visible == (len(app.HORIZONS) > 1)
    assert list(radios[0].choices) == [(label, label) for label in app.HORIZONS]
