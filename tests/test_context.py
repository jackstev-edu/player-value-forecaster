import pandas as pd
import pytest

from pvf.features.build_panel import build_panel
from pvf.features.context import (
    add_context_features,
    club_strength,
    league_strength,
    positional_ranks,
)

CFG = {
    "panel": {
        "anchor_month_day": "07-01",
        "max_value_staleness_days": 365,
        "leagues": ["GB1", "NL1"],
        "min_anchor_season": 2018,
    }
}


def _valuations():
    """Six players priced just before the 2018 season ends, one stale, one priced after.

    Player 1 is revalued on 2019-08-01, after the 2019-07-01 anchor. That number is the
    summer transfer window talking and must not reach a season-2018 feature.
    Player 7's only valuation is from 2016, far outside the 365-day staleness cap.
    """
    rows = [
        (1, "2019-06-01", 10_000_000),
        (1, "2019-08-01", 99_000_000),
        (2, "2019-06-01", 8_000_000),
        (3, "2019-06-01", 6_000_000),
        (4, "2019-06-01", 4_000_000),
        (5, "2019-06-01", 2_000_000),
        (6, "2019-06-01", 1_000_000),
        (7, "2016-01-01", 50_000_000),
        (8, "2019-06-01", 5_000_000),
    ]
    return pd.DataFrame(rows, columns=["player_id", "date", "market_value_in_eur"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def _involvement():
    """Players 1-7 at club 10 in 2018; player 8 at club 20; player 1 stays on into 2019."""
    rows = [(p, 10, 2018) for p in range(1, 8)]
    rows += [(8, 20, 2018), (1, 10, 2019)]
    return pd.DataFrame(rows, columns=["player_id", "club_id", "season"])


def _games():
    """Club 10 plays the Champions League in 2018; club 20 plays only its league."""
    return pd.DataFrame({
        "game_id": [1, 2, 3],
        "competition_id": ["GB1", "CL", "GB1"],
        "season": [2018, 2018, 2019],
        "home_club_id": [10, 10, 10],
        "away_club_id": [20, 30, 20],
    })


def _panel():
    return pd.DataFrame({
        "player_id": [1, 8],
        "club_id": [10, 20],
        "season": [2018, 2018],
        "competition_id": ["GB1", "GB1"],
        "anchor_date": pd.to_datetime(["2019-07-01", "2019-07-01"]),
    })


def _tables():
    return {"player_valuations": _valuations(), "games": _games(),
            "involvement": _involvement()}


def test_club_value_ignores_valuations_dated_after_the_season_ends():
    out = club_strength(_involvement(), _valuations(), CFG["panel"]).set_index(
        ["club_id", "season"])
    # 10+8+6+4+2+1; player 1 counts at 10M, not the 99M he is worth in August
    assert out.loc[(10, 2018), "club_squad_value_eur"] == 31_000_000


def test_stale_valuations_are_left_out_of_club_value():
    out = club_strength(_involvement(), _valuations(), CFG["panel"]).set_index(
        ["club_id", "season"])
    # Player 7's 50M is from 2016, well past the 365-day cap
    assert out.loc[(10, 2018), "club_squad_size"] == 6


def test_club_median_value_is_the_squad_median():
    out = club_strength(_involvement(), _valuations(), CFG["panel"]).set_index(
        ["club_id", "season"])
    assert out.loc[(10, 2018), "club_median_value_eur"] == 5_000_000


def test_club_top5_value_averages_the_five_most_valuable():
    out = club_strength(_involvement(), _valuations(), CFG["panel"]).set_index(
        ["club_id", "season"])
    # (10+8+6+4+2)/5, so the 1M squad player does not dilute the star measure
    assert out.loc[(10, 2018), "club_top5_value_eur"] == 6_000_000


def test_a_club_gets_one_row_per_season():
    out = club_strength(_involvement(), _valuations(), CFG["panel"])
    assert not out.duplicated(["club_id", "season"]).any()


def test_context_features_describe_the_season_before_the_anchor():
    out = add_context_features(_panel(), _tables(), CFG).set_index("player_id")
    # The 2019-07-01 anchor reads club 10's 2018 squad, not its 2019 one
    assert out.loc[1, "club_squad_value_eur"] == 31_000_000


def test_european_participation_reaches_the_panel():
    out = add_context_features(_panel(), _tables(), CFG).set_index("player_id")
    assert bool(out.loc[1, "played_ucl"]) is True


def test_a_club_with_no_european_games_is_false_not_null():
    out = add_context_features(_panel(), _tables(), CFG).set_index("player_id")
    assert bool(out.loc[8, "played_ucl"]) is False


def test_context_features_do_not_multiply_panel_rows():
    panel = _panel()
    out = add_context_features(panel, _tables(), CFG)
    assert len(out) == len(panel)


def _full_tables():
    """Club 10 plays GB1 and the Champions League in 2019; club 20 plays only GB1."""
    return {
        "games": pd.DataFrame({
            "game_id": [1, 2],
            "competition_id": ["GB1", "CL"],
            "season": [2019, 2019],
            "home_club_id": [10, 10],
            "away_club_id": [20, 30],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2], "club_id": [10, 20],
            "type": ["starting_lineup", "starting_lineup"],
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2],
            "player_club_id": [10, 20], "minutes_played": [90, 90],
            "goals": [0, 0], "assists": [0, 0],
        }),
        "players": pd.DataFrame({
            "player_id": [1, 2],
            "date_of_birth": pd.to_datetime(["1995-01-01", "1996-01-01"]),
            "position": ["Attack", "Defender"],
        }),
        "player_valuations": pd.DataFrame({
            "player_id": [1, 2],
            "date": pd.to_datetime(["2020-06-01", "2020-06-01"]),
            "market_value_in_eur": [10_000_000, 3_000_000],
        }),
        # Slot 5 reads these two; empty means nobody is tracked, so its columns
        # come back null rather than zero.
        "player_injuries": pd.DataFrame({
            "player_id": pd.Series(dtype="int64"),
            "from_date": pd.Series(dtype="datetime64[ns]"),
            "end_date": pd.Series(dtype="datetime64[ns]"),
            "days_missed": pd.Series(dtype="float64"),
        }),
        "transfer_history": pd.DataFrame({
            "player_id": pd.Series(dtype="int64"),
            "transfer_date": pd.Series(dtype="datetime64[ns]"),
            "transfer_type": pd.Series(dtype="object"),
            "transfer_fee": pd.Series(dtype="int64"),
        }),
    }


def _full_cfg():
    return {
        "panel": {"anchor_month_day": "07-01", "min_anchor_season": 2020,
                  "max_value_staleness_days": 365, "leagues": ["GB1"]},
        "target": {"horizons": [1]},
        "split": {"values_available_until": "2020-06-12"},
    }


def test_build_panel_attaches_club_strength():
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    assert panel.loc[1, "club_squad_value_eur"] == 10_000_000


def test_build_panel_attaches_european_participation():
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    assert bool(panel.loc[1, "played_ucl"]) is True
    assert bool(panel.loc[2, "played_ucl"]) is False


def _leagues():
    """club_domestic_league output for the fixture: clubs 10 and 20 both play GB1."""
    return pd.DataFrame({
        "club_id": [10, 20, 10, 20],
        "season": [2018, 2018, 2019, 2019],
        "competition_id": ["GB1", "GB1", "GB1", "GB1"],
    })


def test_league_value_totals_the_clubs_that_played_in_it():
    strength = club_strength(_involvement(), _valuations(), CFG["panel"])
    out = league_strength(strength, _leagues()).set_index(["competition_id", "season"])
    # club 10 is worth 31M and club 20 is worth 5M
    assert out.loc[("GB1", 2018), "league_total_value_eur"] == 36_000_000


def test_league_median_is_taken_across_clubs_not_players():
    strength = club_strength(_involvement(), _valuations(), CFG["panel"])
    out = league_strength(strength, _leagues()).set_index(["competition_id", "season"])
    # (31M + 5M) / 2 clubs, not the median of the nine individual players
    assert out.loc[("GB1", 2018), "league_median_club_value_eur"] == 18_000_000


def test_league_counts_only_clubs_with_a_strength_row():
    strength = club_strength(_involvement(), _valuations(), CFG["panel"])
    out = league_strength(strength, _leagues()).set_index(["competition_id", "season"])
    # Club 20 fielded nobody in 2019, so GB1 that season is club 10 alone
    assert out.loc[("GB1", 2019), "league_club_count"] == 1


def test_league_strength_is_one_row_per_league_season():
    strength = club_strength(_involvement(), _valuations(), CFG["panel"])
    out = league_strength(strength, _leagues())
    assert not out.duplicated(["competition_id", "season"]).any()


def test_club_share_of_league_is_its_slice_of_the_total():
    out = add_context_features(_panel(), _tables(), CFG).set_index("player_id")
    # Club 10 is 31M of GB1's 36M in 2018 - a big fish in a small pond
    assert out.loc[1, "club_value_share_of_league"] == pytest.approx(31 / 36)


def test_build_panel_attaches_league_strength():
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    # GB1 2019 is club 10 at 10M plus club 20 at 3M
    assert panel.loc[1, "league_total_value_eur"] == 13_000_000


def _rank_panel():
    """Two GB1 clubs at one anchor, plus a lone player at a second anchor.

    Player 6 exists so that a rank computed across anchors instead of within one shows
    up as a peer count of 3 at club 10 rather than 2.
    """
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6],
        "club_id": [10, 10, 10, 20, 20, 10],
        "competition_id": ["GB1"] * 6,
        "position": ["Attack", "Attack", "Defender", "Attack", "Attack", "Attack"],
        "value_eur": [10e6, 4e6, 7e6, 20e6, 1e6, 99e6],
        "anchor_date": pd.to_datetime(["2019-07-01"] * 5 + ["2020-07-01"]),
    })


def test_most_valuable_player_in_his_position_ranks_first_at_his_club():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    assert out.loc[1, "position_rank_at_club"] == 1
    assert out.loc[2, "position_rank_at_club"] == 2


def test_each_position_is_ranked_on_its_own():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    # Player 3 is the only defender at club 10, so he leads his position despite
    # player 1 being worth more overall
    assert out.loc[3, "position_rank_at_club"] == 1


def test_club_ranks_do_not_pool_players_from_other_clubs():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    # Player 5 is the cheapest attacker in the league but only the second at his club
    assert out.loc[5, "position_rank_at_club"] == 2
    assert out.loc[5, "position_rank_in_league"] == 4


def test_league_ranks_pool_every_club_in_that_league():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    # Behind player 4's 20M, ahead of players 2 and 5
    assert out.loc[1, "position_rank_in_league"] == 2


def test_peer_counts_are_reported_for_both_scopes():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    assert out.loc[1, "position_peers_at_club"] == 2
    assert out.loc[1, "position_peers_in_league"] == 4


def test_percentile_is_one_for_the_most_valuable_in_the_group():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    assert out.loc[4, "position_pctile_in_league"] == pytest.approx(1.0)
    assert out.loc[5, "position_pctile_in_league"] == pytest.approx(0.25)


def test_ranks_are_computed_within_one_anchor():
    out = positional_ranks(_rank_panel()).set_index("player_id")
    # Player 6 is at club 10 too, but a season later
    assert out.loc[6, "position_peers_at_club"] == 1


def test_players_worth_the_same_share_a_rank():
    panel = pd.DataFrame({
        "player_id": [1, 2, 3],
        "club_id": [10, 10, 10],
        "competition_id": ["GB1"] * 3,
        "position": ["Attack"] * 3,
        "value_eur": [5e6, 5e6, 1e6],
        "anchor_date": pd.to_datetime(["2019-07-01"] * 3),
    })
    out = positional_ranks(panel).set_index("player_id")
    assert out.loc[1, "position_rank_at_club"] == 1
    assert out.loc[2, "position_rank_at_club"] == 1
    # The tie consumes both places, so the next player is third
    assert out.loc[3, "position_rank_at_club"] == 3


def test_build_panel_attaches_positional_ranks():
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    # One attacker and one defender, each alone in his position at his own club
    assert panel.loc[1, "position_rank_at_club"] == 1
    assert panel.loc[1, "position_peers_in_league"] == 1


def test_a_player_with_an_unknown_position_gets_no_rank():
    """`players.position` uses the string "Missing", not null, for 586 players.

    Ranking them puts each in a group of one or two and hands back rank 1 and a
    percentile of 1.0, which reads to a model as "the best in his position" when it
    actually means "we do not know what he plays".
    """
    panel = pd.DataFrame({
        "player_id": [1, 2],
        "club_id": [10, 10],
        "competition_id": ["GB1", "GB1"],
        "position": ["Missing", "Attack"],
        "value_eur": [5e6, 5e6],
        "anchor_date": pd.to_datetime(["2019-07-01"] * 2),
    })
    out = positional_ranks(panel).set_index("player_id")
    assert pd.isna(out.loc[1, "position_rank_at_club"])
    assert pd.isna(out.loc[1, "position_pctile_in_league"])
    assert pd.isna(out.loc[1, "position_peers_at_club"])
    # The known-position player is unaffected
    assert out.loc[2, "position_rank_at_club"] == 1
