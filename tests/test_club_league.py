import pandas as pd

from pvf.features.club_league import (
    club_domestic_league,
    club_european_participation,
    player_club_season,
    players_at_anchor,
)

LEAGUES = ["GB1", "NL1"]


def _games():
    """Club 10 is promoted from NL1 to GB1 and plays the Champions League in 2019."""
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4],
        "competition_id": ["NL1", "GB1", "CL", "EL"],
        "season": [2018, 2019, 2019, 2019],
        "home_club_id": [10, 10, 10, 20],
        "away_club_id": [20, 20, 30, 30],
    })


def test_club_league_changes_between_seasons():
    out = club_domestic_league(_games(), LEAGUES)
    club10 = out[out.club_id == 10].set_index("season")["competition_id"]
    assert club10.loc[2018] == "NL1"
    assert club10.loc[2019] == "GB1"


def test_european_game_does_not_create_a_second_domestic_league():
    out = club_domestic_league(_games(), LEAGUES)
    # Club 10 played GB1 and the Champions League in 2019, but sits in one league
    assert len(out[(out.club_id == 10) & (out.season == 2019)]) == 1


def test_european_participation_flags_each_competition():
    out = club_european_participation(_games()).set_index(["club_id", "season"])
    assert bool(out.loc[(10, 2019), "played_ucl"]) is True
    assert bool(out.loc[(10, 2019), "played_uel"]) is False
    assert bool(out.loc[(20, 2019), "played_uel"]) is True


def _lineups():
    """Player 1 starts game 2; player 2 is an unused substitute in the same game."""
    return pd.DataFrame({
        "game_id": [2, 2],
        "player_id": [1, 2],
        "club_id": [10, 10],
        "type": ["starting_lineup", "substitutes"],
    })


def _appearances():
    """Only player 1 took the field, so player 2 has no appearance row."""
    return pd.DataFrame({
        "game_id": [2],
        "player_id": [1],
        "player_club_id": [10],
        "minutes_played": [90],
    })


def test_unused_substitute_counts_as_squad_membership():
    out = player_club_season(_lineups(), _appearances(), _games())
    sub = out[out.player_id == 2].iloc[0]
    assert sub.squad_games == 1
    assert sub.minutes == 0


def test_season_without_lineups_falls_back_to_appearances():
    # game_lineups starts in July 2013, so season 2012 has appearances only
    games = pd.DataFrame({
        "game_id": [9], "competition_id": ["GB1"], "season": [2012],
        "home_club_id": [10], "away_club_id": [20],
    })
    apps = pd.DataFrame({
        "game_id": [9], "player_id": [7], "player_club_id": [10], "minutes_played": [90],
    })
    empty_lineups = _lineups().iloc[0:0]

    out = player_club_season(empty_lineups, apps, games)

    assert out.player_id.tolist() == [7]
    assert out.iloc[0].minutes == 90


def test_squad_games_never_undercounts_games_actually_played():
    # game_lineups is missing rows for some games: 1,010 player-club-seasons in the real
    # data have more appearances than lineup entries. Playing implies being in the squad.
    games = pd.DataFrame({
        "game_id": [1, 2], "competition_id": ["GB1", "GB1"], "season": [2019, 2019],
        "home_club_id": [10, 10], "away_club_id": [20, 20],
    })
    lineups = pd.DataFrame({
        "game_id": [1], "player_id": [1], "club_id": [10], "type": ["starting_lineup"],
    })
    apps = pd.DataFrame({
        "game_id": [1, 2], "player_id": [1, 1], "player_club_id": [10, 10],
        "minutes_played": [90, 90],
    })

    out = player_club_season(lineups, apps, games)

    assert out.iloc[0].played_games == 2
    assert out.iloc[0].squad_games == 2


def test_mid_season_transfer_keeps_both_clubs_and_marks_the_busier_one_primary():
    games = pd.DataFrame({
        "game_id": [1, 2, 3], "competition_id": ["GB1"] * 3, "season": [2019] * 3,
        "home_club_id": [10, 10, 20], "away_club_id": [30, 30, 30],
    })
    lineups = pd.DataFrame({
        "game_id": [1, 2, 3],
        "player_id": [1, 1, 1],
        "club_id": [10, 10, 20],
        "type": ["starting_lineup"] * 3,
    })
    out = player_club_season(lineups, _appearances().iloc[0:0], games)

    assert len(out) == 2
    assert out.set_index("club_id").loc[10, "is_primary"]
    assert not out.set_index("club_id").loc[20, "is_primary"]


def test_anchor_reads_only_the_previous_season():
    pcs = pd.DataFrame({
        "player_id": [1, 2],
        "season": [2018, 2019],
        "club_id": [10, 10],
        "squad_games": [5, 5],
        "minutes": [450, 450],
        "is_primary": [True, True],
    })
    cdl = pd.DataFrame({
        "club_id": [10, 10],
        "season": [2018, 2019],
        "competition_id": ["NL1", "GB1"],
    })

    out = players_at_anchor(pcs, cdl, anchor_year=2020)

    assert out.player_id.tolist() == [2]
    assert out.iloc[0].competition_id == "GB1"


def test_anchor_drops_players_whose_club_has_no_tracked_league():
    pcs = pd.DataFrame({
        "player_id": [1], "season": [2019], "club_id": [99],
        "squad_games": [5], "minutes": [450], "is_primary": [True],
    })
    cdl = pd.DataFrame({"club_id": [10], "season": [2019], "competition_id": ["GB1"]})

    assert players_at_anchor(pcs, cdl, anchor_year=2020).empty
