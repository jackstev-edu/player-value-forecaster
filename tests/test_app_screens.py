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


def player_parts(app, outputs):
    """The named pieces of an open_player_screen result."""
    n = len(app.SCREENS) + 1
    panels_at = n + 6
    return {"hero": outputs[n], "chart": outputs[n + 1], "means": outputs[n + 2],
            "sure": outputs[n + 3], "details": outputs[n + 4], "season": outputs[n + 5],
            "panels": outputs[panels_at:panels_at + len(app.PANELS)],
            "open": outputs[-2], "selected": outputs[-1]}


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
    got = player_parts(app, app.open_player_screen(pid, app.DEFAULT_HORIZON))
    assert isinstance(got["chart"], go.Figure)
    assert app.PLAYER_BY_ID[pid]["name"] in got["hero"]
    assert "What this means" in got["means"]
    assert got["selected"] == pid


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


# Player screen: hero card, range bar and the three panels

def hero_of(app, pid):
    return app.hero_card(pid, app.DEFAULT_H)


def aria_of(markup: str) -> str:
    return re.search(r'aria-label="([^"]+)"', markup).group(1)


def test_hero_numbers_match_the_explanation(app):
    """The hero and the explanation read the same forecast, so they cannot disagree."""
    h = app.DEFAULT_H
    checked = 0
    for pid in [int(p) for p in app.PLAYERS["player_id"].head(25)]:
        row = app.PLAYERS_BY_ID.loc[pid]
        explanation = app.make_explanation(pid, h)
        hero = hero_of(app, pid)
        if "No forecast" in explanation:
            continue
        # Middle estimate, today's value and the percent all appear in both
        assert app.fmt_eur(row[f"p50_{h}"]) in hero
        assert app.fmt_eur(row[f"p50_{h}"]) in explanation
        assert app.fmt_eur(row["current_value_eur"]) in hero
        assert app.fmt_eur(row["current_value_eur"]) in explanation
        percent = app.fmt_change(row[f"change_{h}"]).lstrip("+-")
        assert percent in hero and percent in explanation
        checked += 1
    assert checked, "no player had a forecast to compare"


def test_range_bar_aria_label_states_all_four_numbers(app):
    h = app.DEFAULT_H
    for pid in [int(p) for p in app.PLAYERS["player_id"].head(15)]:
        row = app.PLAYERS_BY_ID.loc[pid]
        bar = app.range_bar(pid, h)
        if not bar:
            continue
        label = aria_of(bar)
        for value in (row[f"p10_{h}"], row[f"p90_{h}"], row[f"p50_{h}"],
                      row["current_value_eur"]):
            assert app.fmt_eur(value) in label, f"{app.fmt_eur(value)} missing from {label!r}"
        assert "Likely range" in label and "middle estimate" in label


def test_range_bar_marks_sit_inside_the_track(app):
    """Every marker lands on the bar, even when today's value is outside the band."""
    h = app.DEFAULT_H
    for pid in [int(p) for p in app.PLAYERS["player_id"].head(40)]:
        bar = app.range_bar(pid, h)
        if not bar:
            continue
        for percent in re.findall(r"left:([\d.]+)%", bar):
            assert 0.0 <= float(percent) <= 100.0
        band = re.search(r'pvf-bar-band" style="left:([\d.]+)%;width:([\d.]+)%', bar)
        left, width = float(band.group(1)), float(band.group(2))
        assert 0 < width < 100, "the band should read as a band, not fill the track"
        assert left + width <= 100.0


def test_range_bar_is_empty_without_a_forecast(app, monkeypatch):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    players = app.PLAYERS.copy()
    players.loc[players["player_id"] == pid, f"p50_{app.DEFAULT_H}"] = float("nan")
    monkeypatch.setattr(app, "PLAYERS_BY_ID", players.set_index("player_id", drop=False))
    assert app.range_bar(pid, app.DEFAULT_H) == ""


def test_hero_names_the_profile_fields(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    player = app.PLAYER_BY_ID[pid]
    hero = hero_of(app, pid)
    for field in ("club_name", "league_name", "nationality", "sub_position"):
        assert html.escape(str(player[field])) in hero
    assert f'data-pid="{pid}"' in hero
    assert app.seasons_text(app.DEFAULT_H) in hero


def test_hero_escapes_every_bundle_string(app):
    """A name or club carrying markup cannot introduce a tag into the hero."""
    pid = int(app.PLAYERS["player_id"].iloc[0])
    nasty = "<script>alert(1)</script> & co"
    players = app.PLAYERS.copy()
    for column in ("name", "club_name", "league_name", "nationality", "sub_position"):
        players.loc[players["player_id"] == pid, column] = nasty
    by_id = players.set_index("player_id", drop=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app, "PLAYERS_BY_ID", by_id)
        mp.setattr(app, "PLAYER_BY_ID", {pid: by_id.loc[pid].to_dict()})
        hero = app.hero_card(pid, app.DEFAULT_H)
    # Only the hero's own elements survive, so the name cannot open a tag
    tags = sorted(set(re.findall(r"<\s*([a-zA-Z][\w-]*)", hero)))
    assert tags == ["div", "h2", "header", "p", "span"], f"unexpected tags {tags}"
    assert "&lt;script&gt;" in hero and "&amp;" in hero


def test_badge_appears_only_for_flagged_players(app, monkeypatch):
    h = app.DEFAULT_H
    pid = int(app.PLAYERS["player_id"].iloc[0])
    if not app.low_confidence_reasons(app.PLAYERS_BY_ID.loc[pid][f"width_{h}"],
                                      app.history_count(pid), h):
        assert "pvf-badge" not in hero_of(app, pid)
    # Widening the band past the cutoff must bring the badge out
    players = app.PLAYERS.copy()
    players.loc[players["player_id"] == pid, f"width_{h}"] = players[f"width_{h}"].max() * 10
    monkeypatch.setattr(app, "PLAYERS_BY_ID", players.set_index("player_id", drop=False))
    assert "pvf-badge" in app.hero_card(pid, h)
    assert app.flag_reason_short(pid, h)


def test_flag_reason_names_the_rule_that_fired(app, monkeypatch):
    h = app.DEFAULT_H
    pid = int(app.PLAYERS["player_id"].iloc[0])
    players = app.PLAYERS.copy()
    players.loc[players["player_id"] == pid, f"width_{h}"] = players[f"width_{h}"].max() * 10
    monkeypatch.setattr(app, "PLAYERS_BY_ID", players.set_index("player_id", drop=False))
    assert app.flag_reason_short(pid, h) == "unusually wide range"
    # With no history recorded either, the reason names both rules
    monkeypatch.setattr(app, "HISTORY_BY_ID", {})
    assert app.flag_reason_short(pid, h) == "wide range, little history"


def test_unflagged_player_has_no_reason_text(app):
    h = app.DEFAULT_H
    for pid in [int(p) for p in app.PLAYERS["player_id"].head(40)]:
        flagged = bool(app.low_confidence_reasons(app.PLAYERS_BY_ID.loc[pid][f"width_{h}"],
                                                  app.history_count(pid), h))
        assert bool(app.flag_reason_short(pid, h)) == flagged


def test_panels_arrive_closed(app):
    got = player_parts(app, app.open_player_screen(
        int(app.PLAYERS["player_id"].iloc[0]), app.DEFAULT_HORIZON))
    assert [c.visible for c in got["panels"]] == [False] * len(app.PANELS)
    assert got["open"] == "", "no panel should be remembered as open on arrival"


def test_each_button_opens_only_its_own_panel(app):
    for i, name in enumerate(app.PANELS):
        out = app.open_panel(name, "")
        shown = [c.visible for c in out[:len(app.PANELS)]]
        assert shown == [j == i for j in range(len(app.PANELS))]
        assert out[-1] == name


def test_opening_one_panel_closes_the_others(app):
    out = app.open_panel("details", "means")
    assert [c.visible for c in out[:3]] == [False, False, True]
    assert out[-1] == "details"


def test_clicking_the_open_panel_closes_it(app):
    out = app.open_panel("sure", "sure")
    assert [c.visible for c in out[:3]] == [False, False, False]
    assert out[-1] == ""


def test_panel_texts_are_filled_for_the_open_player(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    got = player_parts(app, app.open_player_screen(pid, app.DEFAULT_HORIZON))
    assert "What this means" in got["means"]
    assert "8 times out of 10" in got["sure"]
    assert app.PLAYER_BY_ID[pid]["club_name"] in got["details"]


def test_how_sure_adds_no_claim_the_app_does_not_already_make(app):
    """Every paragraph comes from wording the app already shows elsewhere."""
    pid = int(app.PLAYERS["player_id"].iloc[0])
    text = app.how_sure_text(pid, app.DEFAULT_H)
    for paragraph in app.HOW_IT_WORKS.split("\n\n")[1:]:
        assert paragraph in text
    assert app.LIKELY_MEANING in text


def test_how_sure_names_the_flag_reasons_when_flagged(app, monkeypatch):
    h = app.DEFAULT_H
    pid = int(app.PLAYERS["player_id"].iloc[0])
    assert "not flagged low confidence" in app.how_sure_text(pid, h)
    players = app.PLAYERS.copy()
    players.loc[players["player_id"] == pid, f"width_{h}"] = players[f"width_{h}"].max() * 10
    monkeypatch.setattr(app, "PLAYERS_BY_ID", players.set_index("player_id", drop=False))
    flagged = app.how_sure_text(pid, h)
    assert "Low confidence" in flagged and "widest" in flagged


def test_player_details_lists_every_fact(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    player = app.PLAYER_BY_ID[pid]
    text = app.player_details(pid)
    for label in ("Club", "League", "League country", "Nationality", "Position",
                  "Latest valuation", "Past valuations"):
        assert f"| {label} |" in text
    for field in ("club_name", "league_name", "league_country", "nationality", "sub_position"):
        assert html.escape(str(player[field])) in text
    assert str(app.history_count(pid)) in text


def test_player_details_escapes_its_values(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    by_id = {pid: dict(app.PLAYER_BY_ID[pid], club_name="<b>A & B</b>")}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app, "PLAYER_BY_ID", by_id)
        text = app.player_details(pid)
    assert "<b>" not in text and "&lt;b&gt;" in text and "&amp;" in text


def test_chart_is_styled_for_the_dark_page(app):
    """Axis text and hovers must be light, since the page behind them is near black."""
    fig = app.make_chart(int(app.PLAYERS["player_id"].iloc[0]))
    layout = fig.layout
    assert layout.font.color == app.CHART_TEXT
    assert layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert layout.hoverlabel.font.color == app.CHART_TEXT
    # The legend sits below the plot, so the margin has to leave room for it
    assert layout.margin.b >= 48


# The three horizon fallback. The mock fixture carries horizons 1, 2 and 3, so this
# file's app fixture is itself the multi horizon case; the shipped bundle has one.

def seasons_block(app):
    """The Markdown that holds the per season table, if the layout built one."""
    found = [b for b in app.demo.blocks.values()
             if getattr(b, "elem_classes", None) and "pvf-seasons" in b.elem_classes]
    assert len(found) == 1, "the season table should exist exactly once"
    return found[0]


def test_fixture_really_has_three_horizons(app):
    """Guards the tests below: they mean nothing against a one horizon bundle."""
    assert len(app.HORIZONS) == 3
    assert sorted(app.HORIZONS.values()) == [1, 2, 3]


def test_season_table_is_shown_for_a_multi_horizon_bundle(app):
    assert seasons_block(app).visible is True


def test_season_table_lists_every_horizon(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    got = player_parts(app, app.open_player_screen(pid, app.DEFAULT_HORIZON))
    text = got["season"]
    assert text, "a three horizon bundle should fill the season table"
    assert "Every season" in text
    forecasts = app.FORECASTS_BY_ID[pid]
    # One row per horizon, each carrying that horizon's own numbers
    for row in forecasts.itertuples():
        assert app.fmt_eur(row.p50_eur) in text
        assert app.fmt_range(row.p10_eur, row.p90_eur) in text


def test_season_table_marks_the_chosen_horizon(app):
    pid = int(app.PLAYERS["player_id"].iloc[0])
    for label, h in app.HORIZONS.items():
        got = player_parts(app, app.open_player_screen(pid, label))
        target = app.FORECASTS_BY_ID[pid].set_index("horizon").loc[h, "target_date"]
        # The arrow marks the row for the season the user chose
        assert f"▸ **{target:%b %Y}**" in got["season"]


def test_season_table_hidden_when_the_bundle_has_one_horizon(app, monkeypatch):
    """With a single season the table would only repeat the hero, so it goes."""
    monkeypatch.setattr(app, "HORIZONS", {app.DEFAULT_HORIZON: app.DEFAULT_H})
    assert app.season_card(int(app.PLAYERS["player_id"].iloc[0]), app.DEFAULT_H) == ""


def test_look_ahead_control_is_offered_for_three_horizons(app):
    radios = [c for c in app.demo.blocks.values()
              if isinstance(c, gr.Radio) and c.label == "Look ahead"]
    assert len(radios) == 1 and radios[0].visible is True
    assert len(radios[0].choices) == 3


def test_changing_horizon_changes_the_hero_numbers(app):
    """The hero follows the chosen season, not just the default one."""
    pid = int(app.PLAYERS["player_id"].iloc[0])
    heroes = {label: player_parts(app, app.open_player_screen(pid, label))["hero"]
              for label in app.HORIZONS}
    assert len(set(heroes.values())) == len(app.HORIZONS), "every horizon should differ"
    for label, h in app.HORIZONS.items():
        row = app.PLAYERS_BY_ID.loc[pid]
        assert app.fmt_eur(row[f"p50_{h}"]) in heroes[label]
        assert app.seasons_text(h) in heroes[label]
