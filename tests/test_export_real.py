import numpy as np
import pandas as pd
import pytest

from pvf.export.real import forecasts_table, history_table, players_table, universe
from pvf.export.schema import validate
from pvf.features.samples import core_features_at

ANCHOR = pd.Timestamp("2026-07-01")


def valuations():
    return pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 3],
        "date": pd.to_datetime(["2025-06-01", "2026-01-15", "2026-08-01",
                                "2025-07-01", "2026-05-01", "2024-01-01"]),
        "market_value_in_eur": [10e6, 20e6, 99e6, 4e6, 2e6, 1e6],
    })


def kaggle_players():
    return pd.DataFrame({
        "player_id": [1, 2, 3], "name": ["Ann Striker", "Bo Keeper", "Cy Stale"],
        "date_of_birth": pd.to_datetime(["2004-07-01", "1996-07-01", "2000-01-01"]),
        "position": ["Attack", "Goalkeeper", "Defender"],
        "country_of_citizenship": ["Spain", "Brazil", "Japan"],
    })


def panel():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 1, 4], "club_id": [10, 20, 20, 10, 20],
        "competition_id": ["ES1", "TR1", "TR1", "ES1", "TR1"],
        "anchor_date": pd.to_datetime(["2026-07-01"] * 3 + ["2025-07-01", "2026-07-01"]),
        "is_primary": [True, True, True, True, False],
        "sub_position": ["Centre-Forward", "Goalkeeper", "Centre-Back", "Centre-Forward", "Right-Back"],
    })


def clubs():
    return pd.DataFrame({"club_id": [10, 20], "name": ["Club Ten", "Club Twenty"]})


def test_core_features_use_only_values_on_or_before_the_anchor():
    f = core_features_at(pd.Series([1, 2]), pd.Series([ANCHOR, ANCHOR]), valuations(), kaggle_players())
    assert f.loc[0, "value_now"] == 20e6  # the August 2026 value is after the anchor
    assert f.loc[0, "value_change_12m"] == pytest.approx(np.log(20e6 / 10e6))
    assert f.loc[0, "age"] == pytest.approx(22.0, abs=0.01)
    assert f.loc[1, "position"] == "Goalkeeper"


def test_stale_value_gives_no_current_value():
    f = core_features_at(pd.Series([3]), pd.Series([ANCHOR]), valuations(), kaggle_players())
    assert pd.isna(f.loc[0, "value_now"])  # last valued January 2024


def test_universe_is_primary_panel_rows_at_the_anchor():
    u = universe(panel(), ANCHOR)
    assert sorted(u["player_id"]) == [1, 2, 3]


def test_players_table_drops_players_without_a_current_value():
    u = universe(panel(), ANCHOR)
    core = core_features_at(u["player_id"], pd.Series(ANCHOR, index=u.index), valuations(), kaggle_players())
    p = players_table(u, core, kaggle_players(), clubs(), ANCHOR)
    assert sorted(p["player_id"]) == [1, 2]
    row = p.set_index("player_id").loc[1]
    assert row["name"] == "Ann Striker" and row["club_name"] == "Club Ten"
    assert row["league_name"] == "LaLiga" and row["league_country"] == "Spain"
    assert row["current_value_eur"] == 20e6 and row["value_date"] == pd.Timestamp("2026-01-15")
    assert row["nationality"] == "Spain" and row["age"] == 22


def test_history_stops_at_the_anchor():
    h = history_table(valuations(), [1, 2], ANCHOR)
    assert h["date"].max() <= ANCHOR
    assert set(h["player_id"]) == {1, 2}
    assert list(h.columns) == ["player_id", "date", "value_eur"]


def test_forecasts_turn_log_ratio_quantiles_into_euros():
    current = pd.Series([20e6, 2e6], index=[1, 2])
    q = pd.DataFrame({"q10": [-0.5, -0.2], "q50": [0.0, 0.1], "q90": [0.5, 0.4]}, index=[1, 2])
    f = forecasts_table(current, q, ANCHOR, model="lgbm-core")
    row = f.set_index("player_id").loc[1]
    assert row["p50_eur"] == pytest.approx(20e6)
    assert row["p10_eur"] == pytest.approx(20e6 * np.exp(-0.5))
    assert row["horizon"] == 1 and row["target_date"] == pd.Timestamp("2027-07-01")
    assert row["model"] == "lgbm-core"


def test_tables_pass_the_bundle_schema():
    u = universe(panel(), ANCHOR)
    core = core_features_at(u["player_id"], pd.Series(ANCHOR, index=u.index), valuations(), kaggle_players())
    p = players_table(u, core, kaggle_players(), clubs(), ANCHOR)
    current = p.set_index("player_id")["current_value_eur"]
    q = pd.DataFrame({"q10": -0.3, "q50": 0.0, "q90": 0.3}, index=current.index)
    validate(p, history_table(valuations(), p["player_id"], ANCHOR), forecasts_table(current, q, ANCHOR, "m"))
