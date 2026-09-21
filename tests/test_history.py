import numpy as np
import pandas as pd

from pvf.features.history import add_value_history

MAX_STALENESS = 365


def _valuations():
    """Player 1 peaks in 2018 then halves; player 2 debuts three months before the
    anchor; player 3 has one ancient valuation and one recent one."""
    rows = [
        (1, "2016-06-01", 5e6),
        (1, "2017-06-01", 12e6),
        (1, "2018-06-01", 20e6),   # the peak, and the value a year before the anchor
        (1, "2019-06-01", 10e6),   # the value at the anchor
        (1, "2019-08-01", 30e6),   # after the anchor: must stay invisible
        (2, "2019-04-01", 3e6),
        (3, "2014-01-01", 1e6),    # far too old to anchor a 12-month comparison
        (3, "2019-06-01", 2e6),
    ]
    return pd.DataFrame(rows, columns=["player_id", "date", "market_value_in_eur"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def _panel():
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "anchor_date": pd.to_datetime(["2019-07-01"] * 3),
        "value_eur": [10e6, 3e6, 2e6],
    })


def _out():
    return add_value_history(_panel(), _valuations(), MAX_STALENESS).set_index("player_id")


def test_peak_is_the_highest_value_on_or_before_the_anchor():
    # 20M in 2018, not the 30M he is worth that August
    assert _out().loc[1, "peak_value_eur"] == 20e6


def test_value_versus_peak_measures_the_fall_from_the_high():
    assert _out().loc[1, "value_vs_peak"] == 0.5


def test_a_player_at_his_peak_sits_at_one():
    assert _out().loc[2, "value_vs_peak"] == 1.0


def test_twelve_month_change_is_a_log_ratio():
    assert _out().loc[1, "value_change_12m"] == np.log(0.5)


def test_a_newcomer_has_no_twelve_month_change():
    # Player 2's first valuation is three months old, so there is nothing to compare to
    out = _out()
    assert pd.isna(out.loc[2, "value_12m_ago_eur"])
    assert pd.isna(out.loc[2, "value_change_12m"])


def test_a_stale_lookback_does_not_count_as_a_year_ago():
    # Player 3's only earlier valuation is from 2014, five years before the lookback date
    out = _out()
    assert pd.isna(out.loc[3, "value_12m_ago_eur"])
    assert pd.isna(out.loc[3, "value_change_12m"])


def test_years_of_history_runs_from_the_first_valuation():
    assert _out().loc[1, "years_of_history"] == (
        pd.Timestamp("2019-07-01") - pd.Timestamp("2016-06-01")).days / 365.25


def test_history_length_is_not_confused_by_a_gap():
    # Player 3 has two valuations five years apart; history is measured from the first
    assert _out().loc[3, "years_of_history"] > 5


def test_valuations_counted_are_only_those_on_or_before_the_anchor():
    # Four for player 1; the August revaluation is not one of them
    assert _out().loc[1, "n_valuations_so_far"] == 4


def test_value_history_does_not_multiply_panel_rows():
    panel = _panel()
    out = add_value_history(panel, _valuations(), MAX_STALENESS)
    assert len(out) == len(panel)


def _full_tables():
    """One anchor at 2020-07-01, drawn from season 2019. Player 1 doubles over the year."""
    return {
        "games": pd.DataFrame({
            "game_id": [1], "competition_id": ["GB1"], "season": [2019],
            "home_club_id": [10], "away_club_id": [20],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2], "club_id": [10, 20],
            "type": ["starting_lineup", "starting_lineup"],
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2],
            "player_club_id": [10, 20], "minutes_played": [90, 90],
        }),
        "players": pd.DataFrame({
            "player_id": [1, 2],
            "date_of_birth": pd.to_datetime(["1995-01-01", "1996-01-01"]),
            "position": ["Attack", "Defender"],
        }),
        "player_valuations": pd.DataFrame({
            "player_id": [1, 1, 2],
            "date": pd.to_datetime(["2019-06-01", "2020-06-01", "2020-06-01"]),
            "market_value_in_eur": [5e6, 10e6, 3e6],
        }),
    }


def _full_cfg():
    return {
        "panel": {"anchor_month_day": "07-01", "min_anchor_season": 2020,
                  "max_value_staleness_days": 365, "leagues": ["GB1"]},
        "target": {"horizons": [1]},
        "split": {"values_available_until": "2020-06-12"},
    }


def test_build_panel_attaches_value_history():
    from pvf.features.build_panel import build_panel
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    assert panel.loc[1, "value_change_12m"] == np.log(2)
    assert panel.loc[1, "peak_value_eur"] == 10e6
    assert panel.loc[1, "n_valuations_so_far"] == 2
