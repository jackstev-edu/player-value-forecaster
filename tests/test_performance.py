import numpy as np
import pandas as pd

from pvf.features.performance import add_performance_features


def _games():
    """Season 2019 games 1-5; game 99 belongs to season 2012, before lineups exist."""
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 99],
        "season": [2019, 2019, 2019, 2019, 2019, 2012],
    })


def _game_lineups():
    """Player 1 starts three and is an unused sub twice. Player 2 is only ever a sub.
    Player 3 played in 2012, which `game_lineups` does not cover at all."""
    rows = [
        (1, 1, 10, "starting_lineup"),
        (2, 1, 10, "starting_lineup"),
        (3, 1, 10, "starting_lineup"),
        (4, 1, 10, "substitutes"),
        (5, 1, 10, "substitutes"),
        (1, 2, 20, "substitutes"),
        (2, 2, 20, "substitutes"),
        # Player 4 moves at the winter break: two clubs, one season
        (1, 4, 10, "starting_lineup"),
        (4, 4, 20, "starting_lineup"),
        (5, 4, 20, "substitutes"),
    ]
    return pd.DataFrame(rows, columns=["game_id", "player_id", "club_id", "type"])


def _appearances():
    """Minutes, goals and assists for whoever actually took the field."""
    rows = [
        (1, 1, 10, 90, 1, 0),
        (2, 1, 10, 90, 1, 1),
        (3, 1, 10, 90, 0, 0),
        (1, 2, 20, 30, 0, 0),
        (99, 3, 30, 90, 2, 1),
        (1, 4, 10, 90, 1, 0),
        (4, 4, 20, 90, 0, 2),
    ]
    return pd.DataFrame(rows, columns=["game_id", "player_id", "player_club_id",
                                       "minutes_played", "goals", "assists"])


def _panel():
    """One anchor. Player 5 was named in squads but never took the field."""
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 4, 5],
        "club_id": [10, 20, 30, 10, 20, 10],
        "season": [2019, 2019, 2012, 2019, 2019, 2019],
        "squad_games": [5, 2, 1, 1, 2, 4],
        "minutes": [270, 30, 90, 90, 90, 0],
        "anchor_date": pd.to_datetime(["2020-07-01"] * 6),
    })


def _tables():
    return {"game_lineups": _game_lineups(), "appearances": _appearances(),
            "games": _games()}


def _out():
    return add_performance_features(_panel(), _tables()).set_index(["player_id", "club_id"])


def test_starts_count_only_starting_lineup_rows():
    # Player 1 is in five squads but starts three of them
    assert _out().loc[(1, 10), "starts"] == 3


def test_a_squad_player_who_never_started_has_zero_starts():
    # Player 2 was named twice, both times on the bench: a measured zero, not a gap
    assert _out().loc[(2, 20), "starts"] == 0


def test_start_share_is_starts_over_squad_games():
    assert _out().loc[(1, 10), "start_share"] == 3 / 5


def test_a_season_without_lineup_coverage_has_no_starts():
    # `game_lineups` begins in July 2013, so season 2012 has no starting XI to count.
    # Zero here would read as "never started" instead of "not recorded" (decision #15).
    out = _out()
    assert pd.isna(out.loc[(3, 30), "starts"])
    assert pd.isna(out.loc[(3, 30), "start_share"])


def test_goals_and_assists_are_season_totals():
    out = _out()
    assert out.loc[(1, 10), "goals"] == 2
    assert out.loc[(1, 10), "assists"] == 1


def test_goals_assists_per_ninety_scales_by_minutes():
    # 3 involvements in 270 minutes is one per game
    assert _out().loc[(1, 10), "goals_assists_per90"] == 1.0


def test_a_player_who_never_took_the_field_has_no_per_ninety():
    # Player 5 has zero minutes: the rate is undefined, not infinite and not zero
    out = _out()
    assert out.loc[(5, 10), "goals"] == 0
    assert pd.isna(out.loc[(5, 10), "goals_assists_per90"])
    assert not np.isinf(out.loc[(5, 10), "goals_assists_per90"] or 0)


def test_a_midseason_move_splits_the_season_by_club():
    # Player 4 scored for club 10 and assisted twice for club 20; neither row sees both
    out = _out()
    assert out.loc[(4, 10), "goals"] == 1
    assert out.loc[(4, 10), "assists"] == 0
    assert out.loc[(4, 20), "goals"] == 0
    assert out.loc[(4, 20), "assists"] == 2


def test_starts_are_counted_per_club_not_per_season():
    out = _out()
    assert out.loc[(4, 10), "starts"] == 1
    assert out.loc[(4, 20), "starts"] == 1


def test_performance_does_not_multiply_panel_rows():
    panel = _panel()
    assert len(add_performance_features(panel, _tables())) == len(panel)


def _full_tables():
    """One anchor at 2020-07-01, drawn from season 2019."""
    return {
        "games": pd.DataFrame({
            "game_id": [1, 2], "competition_id": ["GB1", "GB1"], "season": [2019, 2019],
            "home_club_id": [10, 10], "away_club_id": [20, 20],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 2, 1, 2], "player_id": [1, 1, 2, 2],
            "club_id": [10, 10, 20, 20],
            "type": ["starting_lineup", "substitutes", "starting_lineup",
                     "starting_lineup"],
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 2, 1, 2], "player_id": [1, 1, 2, 2],
            "player_club_id": [10, 10, 20, 20],
            "minutes_played": [90, 0, 90, 90], "goals": [2, 0, 0, 0],
            "assists": [0, 0, 1, 0],
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


def test_build_panel_attaches_last_season_performance():
    from pvf.features.build_panel import build_panel
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    assert panel.loc[1, "starts"] == 1
    assert panel.loc[1, "start_share"] == 0.5
    assert panel.loc[1, "goals"] == 2
    assert panel.loc[1, "goals_assists_per90"] == 2.0
    assert panel.loc[2, "starts"] == 2
    assert panel.loc[2, "start_share"] == 1.0
