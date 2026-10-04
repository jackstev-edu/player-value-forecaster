"""Smoke test: every app callback against the real bundle in app/predictions/.

test_app_callbacks.py pins a fixed mock bundle so its assertions stay exact.
This file deliberately does the opposite: it loads whatever bundle ships with
the app and checks each callback survives the real data and shows nothing
broken. It asserts shape and sanity, never specific players, so a re-export
cannot break it.
"""
import importlib.util
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"
BUNDLE_DIR = APP_PATH.parent / "predictions"

# Anything that reaches the screen looking like an unfilled value
BROKEN_TEXT = re.compile(r"\bnan\b|\bnat\b|\bnone\b|<na>", re.IGNORECASE)


@pytest.fixture(scope="module")
def app():
    """Import app.py with no override, so it reads app/predictions/."""
    spec = importlib.util.spec_from_file_location("pvf_app_smoke", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    # No PVF_BUNDLE_DIR here on purpose; the shipped bundle is the point
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def defaults(app):
    return app.default_filters()


def cells(view) -> list[str]:
    """Every table cell as text, for the broken value scan."""
    return [str(v) for col in view.columns for v in view[col]]


def test_bundle_is_the_shipped_one(app):
    assert app.BUNDLE_DIR == BUNDLE_DIR
    assert len(app.PLAYERS) > 0
    assert set(app.HORIZONS.values()) <= set(app.HORIZON_LABELS)


def test_fall_sort_label_matches_the_data(app):
    """The fall sort may only be offered when something is really predicted to fall."""
    horizon = app.horizon_of(app.DEFAULT_HORIZON)
    any_fall = (app.PLAYERS[f"change_{horizon}"] < 0).any()
    if any_fall:
        assert app.FALL_LABEL in app.SORTS
    else:
        assert app.FALL_LABEL not in app.SORTS
        assert app.NO_FALL_LABEL in app.SORTS


def test_example_labels_do_not_invent_a_decline(app):
    """A label saying decline must belong to a player the model predicts will fall."""
    horizon = app.horizon_of(app.DEFAULT_HORIZON)
    for label, pid, _name in app.EXAMPLES:
        if "decline" not in label.lower():
            continue
        change = app.PLAYER_BY_ID[int(pid)].get(f"change_{horizon}")
        assert pd.notna(change) and change < 0, f"{label} names a player at {change}"


def test_sort_info_quotes_the_real_falling_share(app):
    """The helper text may only quote a share that matches the bundle."""
    if "predicted to fall" not in app.SORT_INFO:
        return
    quoted = int(app.SORT_INFO.rsplit("Only ", 1)[1].split("%")[0])
    assert quoted == round(app.FALLING_SHARE * 100)


def test_cards_and_explanations_for_a_sample_of_players(app):
    """Walk a spread of players so odd rows cannot hide behind the top of the table."""
    sample = app.PLAYERS.sort_values("player_id")["player_id"].iloc[::311]
    horizon = app.horizon_of(app.DEFAULT_HORIZON)
    for pid in sample:
        pid = int(pid)
        card = app.make_card(pid, horizon)
        explain = app.make_explanation(pid, horizon)
        assert app.PLAYER_BY_ID[pid]["name"] in card
        assert not BROKEN_TEXT.search(card), f"player {pid} card: {card[:120]}"
        assert not BROKEN_TEXT.search(explain), f"player {pid} explain: {explain[:120]}"
        assert isinstance(app.make_chart(pid), go.Figure)


def test_flagged_share_is_plausible(app):
    """The warning must stay meaningful, so it cannot cover almost everyone."""
    horizon = app.horizon_of(app.DEFAULT_HORIZON)
    flagged = sum(
        bool(app.low_confidence_reasons(width, app.history_count(int(pid)), horizon))
        for pid, width in zip(app.PLAYERS["player_id"], app.PLAYERS[f"width_{horizon}"]))
    share = flagged / len(app.PLAYERS)
    assert 0 < share < 0.5, f"{share:.0%} of players are flagged low confidence"


def test_footer_reports_the_loaded_manifest(app):
    footer = app.make_footer(app.MANIFEST)
    assert str(app.MANIFEST.get("model_version")) in footer
    assert "unknown" not in footer.lower()


def test_real_bundle_shows_no_demo_banner(app):
    """A shipped real bundle must not advertise itself as invented data."""
    if app.MANIFEST.get("is_mock"):
        pytest.skip("this bundle is the mock one")
    assert app.OPTIMISM_NOTE == "" or "lean optimistic" in app.OPTIMISM_NOTE


def test_values_and_dates_are_within_reason(app):
    """Catch an export that writes cents, or a date the app would render oddly."""
    players = app.PLAYERS
    assert players["current_value_eur"].min() > 0
    assert players["current_value_eur"].max() < 5e8
    assert players["age"].between(app.AGE_FLOOR, app.AGE_CEILING).all(), \
        "ages outside the slider range cannot be reached by any filter"
    assert pd.notna(players["value_date"]).all()


# Screen callbacks against the shipped bundle, replacing the table era checks

def defaults_of(app):
    return app.default_filters()


def search_parts(app, values):
    """The named pieces of a run_search result."""
    out = app.run_search(*values)
    n = len(app.SCREENS) + 1
    return {"cards": out[n], "count": out[n + 1], "ids": out[n + 2], "shown": out[n + 3],
            "more": out[n + 4]}


def test_search_returns_cards_with_no_broken_text(app):
    got = search_parts(app, defaults_of(app))
    assert len(got["ids"]) > 0
    assert got["cards"].count('class="pvf-card"') == got["shown"] == app.PAGE
    assert not BROKEN_TEXT.search(got["cards"]), "an unfilled value reached a card"


def test_every_sort_chip_works_on_the_real_bundle(app):
    base = None
    for label in app.SORTS:
        values = defaults_of(app)
        values[8] = label
        got = search_parts(app, values)
        assert len(got["ids"]) > 0, f"{label} returned nothing"
        assert set(got["ids"]) == (base if base is not None else set(got["ids"]))
        base = set(got["ids"])


def test_rise_and_fall_chips_differ(app):
    def top(label):
        values = defaults_of(app)
        values[8] = label
        return search_parts(app, values)["ids"][:10]
    fall = app.FALL_LABEL if app.FALL_LABEL in app.SORTS else app.NO_FALL_LABEL
    assert top("Biggest predicted rise") != top(fall)
    assert top("Most uncertain") != top("Current value")


def test_each_filter_narrows_the_real_bundle(app):
    for slot, column in ((1, "league_name"), (2, "league_country"),
                         (3, "nationality"), (4, "position")):
        choice = app.options(column)[0]
        values = defaults_of(app)
        values[slot] = [choice]
        got = search_parts(app, values)
        assert got["ids"], f"{column}={choice} matched nobody"
        assert all(app.PLAYER_BY_ID[int(p)][column] == choice for p in got["ids"])


def test_age_filter_bounds_on_the_real_bundle(app):
    values = defaults_of(app)
    values[5], values[6] = 30, 32
    got = search_parts(app, values)
    assert got["ids"]
    assert all(30 <= app.PLAYER_BY_ID[int(p)]["age"] <= 32 for p in got["ids"])


def test_accented_name_is_found_without_accents(app):
    accented = app.PLAYERS[app.PLAYERS["name"].str.contains(r"[^\x00-\x7F]", regex=True)]
    if accented.empty:
        pytest.skip("this bundle has no accented names")
    values = defaults_of(app)
    values[0] = app.normalise_name(accented["name"].iloc[0])
    got = search_parts(app, values)
    assert int(accented["player_id"].iloc[0]) in [int(p) for p in got["ids"]]


def test_no_match_advice_on_the_real_bundle(app):
    values = defaults_of(app)
    values[0] = "zzzzznotaplayer"
    got = search_parts(app, values)
    assert got["ids"] == []
    assert "No players match" in got["count"]


def test_load_more_pages_through_the_real_bundle(app):
    got = search_parts(app, defaults_of(app))
    cards, _count, shown, _more = app.load_more(got["ids"], got["shown"], app.DEFAULT_HORIZON,
                                                app.DEFAULT_SORT)
    assert shown == app.PAGE * 2
    assert cards.count('class="pvf-card"') == app.PAGE * 2
    assert not BROKEN_TEXT.search(cards)


def test_opening_a_card_opens_that_player(app):
    got = search_parts(app, defaults_of(app))
    pid = int(got["ids"][5])
    out = app.open_player_screen(pid, app.DEFAULT_HORIZON)
    n = len(app.SCREENS) + 1
    hero, chart, means, sure, details = out[n:n + 5]
    assert f'data-pid="{pid}"' in hero
    assert out[-1] == pid
    assert isinstance(chart, go.Figure)
    assert app.PLAYER_BY_ID[pid]["name"] in hero
    assert "What this means" in means
    assert "8 times out of 10" in sure
    assert app.PLAYER_BY_ID[pid]["club_name"] in details
    for text in (hero, means, sure, details):
        assert not BROKEN_TEXT.search(text)


def test_every_featured_card_opens_its_own_player(app):
    assert app.EXAMPLES
    markup = app.featured_html(app.DEFAULT_H)
    assert markup.count('class="pvf-card"') == len(app.EXAMPLES)
    for _label, pid, _name in app.EXAMPLES:
        out = app.open_player_screen(pid, app.DEFAULT_HORIZON)
        assert out[-1] == pid
        assert f'data-pid="{pid}"' in out[len(app.SCREENS) + 1]


def test_clear_restores_the_default_filters(app):
    assert app.clear_filters() == app.default_filters()
