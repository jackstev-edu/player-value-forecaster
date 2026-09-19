import numpy as np
import pandas as pd

from pvf.features.build_panel import add_player_features, anchor_population, value_asof


def _players():
    return pd.DataFrame({
        "player_id": [1, 2],
        "date_of_birth": pd.to_datetime(["2000-07-01", "1990-01-01"]),
        "position": ["Attack", "Goalkeeper"],
        "sub_position": ["Centre-Forward", "Goalkeeper"],
        "foot": ["right", "left"],
        # 0 means unknown in this dataset, not a 0 cm player
        "height_in_cm": [185, 0],
        "country_of_citizenship": ["Brazil", "Spain"],
    })


def _profiles():
    return pd.DataFrame({"player_id": [1], "is_eu": [False]})


def _panel():
    return pd.DataFrame({
        "player_id": [1, 2],
        "anchor_date": pd.to_datetime(["2020-07-01", "2020-07-01"]),
    })


def test_age_at_anchor_is_measured_from_birth_date():
    out = add_player_features(_panel(), _players(), _profiles())
    ages = out.set_index("player_id")["age"]
    assert np.isclose(ages.loc[1], 20.0, atol=0.01)
    assert np.isclose(ages.loc[2], 30.5, atol=0.05)


def test_zero_height_becomes_unknown_rather_than_zero():
    out = add_player_features(_panel(), _players(), _profiles()).set_index("player_id")
    assert out.loc[1, "height_in_cm"] == 185
    assert pd.isna(out.loc[2, "height_in_cm"])


def test_eu_flag_comes_from_profiles():
    out = add_player_features(_panel(), _players(), _profiles()).set_index("player_id")
    assert out.loc[1, "is_eu"] == False  # noqa: E712 - distinguishing False from NaN


def test_player_without_a_profile_gets_a_null_eu_flag():
    out = add_player_features(_panel(), _players(), _profiles()).set_index("player_id")
    assert pd.isna(out.loc[2, "is_eu"])


def test_leaky_snapshot_columns_are_not_carried_through():
    players = _players().assign(market_value_in_eur=[1e6, 2e6], current_club_id=[10, 20])
    out = add_player_features(_panel(), players, _profiles())
    assert "market_value_in_eur" not in out.columns
    assert "current_club_id" not in out.columns


def _tables():
    return {
        "games": pd.DataFrame({
            "game_id": [1, 2],
            "competition_id": ["GB1", "TS1"],   # TS1 is not a tracked league
            "season": [2019, 2019],
            "home_club_id": [10, 99],
            "away_club_id": [20, 98],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 2],
            "player_id": [1, 2],
            "club_id": [10, 99],
            "type": ["starting_lineup", "starting_lineup"],
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 2], "player_id": [1, 2],
            "player_club_id": [10, 99], "minutes_played": [90, 90],
        }),
    }


def _cfg():
    return {"panel": {"anchor_month_day": "07-01", "min_anchor_season": 2020,
                      "leagues": ["GB1"]}}


def test_population_keeps_only_players_from_a_tracked_league():
    pop = anchor_population(_tables(), _cfg(), last_anchor_season=2020)
    assert pop.player_id.tolist() == [1]
    assert pop.iloc[0].competition_id == "GB1"


def test_population_anchors_on_1_july_of_the_season_after():
    pop = anchor_population(_tables(), _cfg(), last_anchor_season=2020)
    assert pop.iloc[0].anchor_date == pd.Timestamp("2020-07-01")


def test_population_is_one_row_per_player_per_anchor():
    pop = anchor_population(_tables(), _cfg(), last_anchor_season=2020)
    assert not pop.duplicated(["player_id", "anchor_date"]).any()


def test_anchor_dates_are_nanosecond_resolution():
    # pd.Timestamp("2020-07-01") is datetime64[s] in pandas 2, while to_datetime on the
    # parquet gives [ns]. merge_asof refuses to join across resolutions, so value_asof
    # blows up on real data while passing tests that build both sides the same way.
    pop = anchor_population(_tables(), _cfg(), last_anchor_season=2020)
    assert pop["anchor_date"].dtype == "datetime64[ns]"


def test_value_asof_joins_anchors_of_any_datetime_resolution():
    valuations = pd.DataFrame({
        "player_id": [1],
        "date": pd.to_datetime(["2020-01-01"]),
        "market_value_in_eur": [1e6],
    })
    anchors = pd.DataFrame({
        "player_id": [1],
        "anchor_date": pd.Series([pd.Timestamp("2020-07-01")]).astype("datetime64[s]"),
    })

    out = value_asof(valuations, anchors)

    assert out["value_eur"].iloc[0] == 1e6
