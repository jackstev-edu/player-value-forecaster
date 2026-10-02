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


def test_initial_view_opens_a_player(app):
    view, ids, count, chart, card, explain, selected = app.initial_view()
    assert list(view.columns) == app.COLUMNS
    assert len(ids) > 0
    # A first visit must land on a drawn chart, not the empty prompt
    assert isinstance(chart, go.Figure)
    assert selected in app.PLAYER_BY_ID
    assert app.CARD_PROMPT not in card
    assert "What this means" in explain


def test_default_filter_shows_rows_and_no_broken_text(app, defaults):
    view, ids, count, *_ = app.filter_players(*defaults, None)
    assert len(ids) == len(view) > 0
    assert len(view) <= app.MAX_ROWS
    bad = [c for c in cells(view) if BROKEN_TEXT.search(c)]
    assert not bad, f"unfilled values reached the table: {bad[:5]}"
    assert "player" in count.lower()


def test_every_horizon_has_usable_numbers(app, defaults):
    """Each offered horizon must produce real forecasts, not a column of n/a."""
    for label in app.HORIZONS:
        values = list(defaults)
        values[7] = label
        view, ids, *_ = app.filter_players(*values, None)
        assert len(ids) > 0, f"{label} returned no rows"
        real = [c for c in view["Change"] if c != "n/a"]
        assert real, f"{label} shows no forecast at all, so it should not be offered"


def test_every_sort_orders_without_error(app, defaults):
    for sort_by in app.SORTS:
        values = list(defaults)
        values[8] = sort_by
        view, ids, *_ = app.filter_players(*values, None)
        assert len(ids) == len(view) > 0, f"{sort_by} returned nothing"


def test_sorts_actually_differ(app, defaults):
    """Rise and fall must not return the same players, as they did with flat mock data."""
    def top(sort_by):
        values = list(defaults)
        values[8] = sort_by
        _, ids, *_ = app.filter_players(*values, None)
        return ids[:10]
    assert top("Biggest predicted rise") != top("Biggest predicted fall")
    assert top("Most uncertain") != top("Current value")


def test_every_filter_narrows_the_table(app, defaults):
    """One value from each filter must return only rows carrying that value."""
    total = len(app.PLAYERS)
    for slot, column in ((1, "league_name"), (2, "league_country"),
                         (3, "nationality"), (4, "position")):
        choice = app.options(column)[0]
        values = list(defaults)
        values[slot] = [choice]
        _, ids, *_ = app.filter_players(*values, None)
        matched = app.PLAYERS[app.PLAYERS[column] == choice]
        assert len(ids) > 0, f"{column}={choice} matched nobody"
        assert len(matched) < total, f"{column}={choice} did not narrow anything"
        # Every returned row really carries the chosen value, not just fewer rows
        assert all(app.PLAYER_BY_ID[int(p)][column] == choice for p in ids)


def test_age_filter_bounds(app, defaults):
    values = list(defaults)
    values[5], values[6] = 30, 32
    view, ids, *_ = app.filter_players(*values, None)
    assert len(ids) > 0
    assert all(30 <= app.PLAYER_BY_ID[int(p)]["age"] <= 32 for p in ids)


def test_search_finds_an_accented_name_without_accents(app, defaults):
    """A real bundle carries accents, so plain typing must still match."""
    accented = app.PLAYERS[app.PLAYERS["name"].str.contains(r"[^\x00-\x7F]", regex=True)]
    if accented.empty:
        pytest.skip("this bundle has no accented names")
    name = accented["name"].iloc[0]
    plain = app.normalise_name(name)
    values = list(defaults)
    values[0] = plain
    _, ids, *_ = app.filter_players(*values, None)
    assert int(accented["player_id"].iloc[0]) in [int(i) for i in ids]


def test_no_match_message_names_the_filters(app, defaults):
    values = list(defaults)
    values[0] = "zzzzznotaplayer"
    view, ids, count, *_ = app.filter_players(*values, None)
    assert len(ids) == 0
    assert list(view.columns) == app.COLUMNS
    assert "No players match" in count


def test_select_row_opens_that_player(app, defaults):
    view, ids, *_ = app.filter_players(*defaults, None)

    class FakeSelect:
        index = [3, 0]

    outputs = app.on_select(ids, app.DEFAULT_HORIZON, FakeSelect())
    chart, card, explain, selected = outputs[3:]
    assert selected == ids[3]
    assert isinstance(chart, go.Figure)
    assert app.PLAYER_BY_ID[int(selected)]["name"] in card


def test_open_every_example(app):
    """Each example must open its own player and draw a chart."""
    assert app.EXAMPLES, "no examples were picked from this bundle"
    assert len({pid for _, pid, _ in app.EXAMPLES}) == len(app.EXAMPLES)
    for _label, pid, name in app.EXAMPLES:
        out = app.open_example(name, pid)
        chart, card, explain, selected = out[-4:]
        assert selected == pid
        assert isinstance(chart, go.Figure)
        assert not BROKEN_TEXT.search(card), f"{name} card shows an unfilled value"


def test_fall_sort_label_matches_the_data(app):
    """The fall sort may only be offered when something is really predicted to fall."""
    horizon = app.horizon_of(app.DEFAULT_HORIZON)
    any_fall = (app.PLAYERS[f"change_{horizon}"] < 0).any()
    if any_fall:
        assert app.FALL_LABEL in app.SORTS
    else:
        assert app.FALL_LABEL not in app.SORTS
        assert app.NO_FALL_LABEL in app.SORTS


def test_fall_sort_really_returns_falls(app, defaults):
    """Whatever the fall sort is called, its top rows must match its name."""
    label = app.FALL_LABEL if app.FALL_LABEL in app.SORTS else app.NO_FALL_LABEL
    values = list(defaults)
    values[8] = label
    view, ids, *_ = app.filter_players(*values, None)
    top = list(view["Change"])[:5]
    if label == app.FALL_LABEL:
        assert all(c.startswith("-") for c in top), f"fall sort led with {top}"


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


def test_reset_filters_restores_defaults(app, defaults):
    out = app.reset_filters(None)
    assert list(out[:len(defaults)]) == defaults


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
