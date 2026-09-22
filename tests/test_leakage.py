import pytest

from pvf.features.leakage import assert_no_leakage, is_leaky


def test_snapshot_columns_flagged():
    assert is_leaky("players", "market_value_in_eur")
    assert is_leaky("player_profiles", "current_club_name")


def test_static_columns_allowed():
    assert not is_leaky("players", "date_of_birth")
    assert not is_leaky("player_profiles", "height")
    assert not is_leaky("player_valuations", "market_value_in_eur")


def test_assert_raises():
    with pytest.raises(ValueError):
        assert_no_leakage({"players": ["foot", "agent_name"]})


# --- Task 12: every slot declares what it reads, and nothing reads the future ---

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from pvf.features.leakage import FEATURE_SOURCES

ANCHOR = pd.Timestamp("2020-07-01")


def test_every_declared_source_column_is_safe():
    """The literal task-12 assertion: run every slot's inputs past the blocklist."""
    for slot, used in FEATURE_SOURCES.items():
        assert_no_leakage(used)


def test_every_feature_module_declares_its_sources():
    """A new slot cannot be added without saying what it reads."""
    package = Path(__file__).resolve().parents[1] / "src" / "pvf" / "features"
    modules = {p.stem for p in package.glob("*.py")}
    # leakage declares nothing itself; club_league is a helper whose outputs are
    # declared by build_panel, which reads it to pick the population.
    modules -= {"__init__", "leakage", "club_league"}
    assert modules == set(FEATURE_SOURCES), (
        f"undeclared: {modules - set(FEATURE_SOURCES)}, "
        f"stale: {set(FEATURE_SOURCES) - modules}")


def test_a_leaky_column_in_a_declaration_is_caught():
    with pytest.raises(ValueError):
        assert_no_leakage({"players": ["position", "contract_expiration_date"]})


def _base_tables():
    """Two players, one anchor at 2020-07-01, drawn from season 2019."""
    return {
        "games": pd.DataFrame({
            "game_id": [1, 2], "competition_id": ["GB1", "GB1"], "season": [2019, 2019],
            "home_club_id": [10, 10], "away_club_id": [20, 20],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 2, 1, 2], "player_id": [1, 1, 2, 2],
            "club_id": [10, 10, 20, 20],
            "type": ["starting_lineup"] * 4,
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 2, 1, 2], "player_id": [1, 1, 2, 2],
            "player_club_id": [10, 10, 20, 20],
            "minutes_played": [90, 90, 90, 90], "goals": [1, 0, 0, 0],
            "assists": [0, 1, 0, 0],
        }),
        "players": pd.DataFrame({
            "player_id": [1, 2],
            "date_of_birth": pd.to_datetime(["1995-01-01", "1996-01-01"]),
            "position": ["Attack", "Defender"],
        }),
        "player_valuations": pd.DataFrame({
            "player_id": [1, 1, 2, 2],
            "date": pd.to_datetime(["2019-06-01", "2020-06-01",
                                    "2019-06-01", "2020-06-01"]),
            "market_value_in_eur": [5e6, 10e6, 3e6, 4e6],
        }),
        "player_injuries": pd.DataFrame({
            "player_id": [1], "from_date": pd.to_datetime(["2019-09-01"]),
            "end_date": pd.to_datetime(["2019-10-01"]), "days_missed": [30.0],
        }),
        "transfer_history": pd.DataFrame({
            "player_id": [1], "transfer_date": pd.to_datetime(["2019-08-01"]),
            "transfer_type": ["Transfer"], "transfer_fee": [5_000_000],
        }),
    }


def _polluted_tables():
    """The same world, plus everything that happens AFTER the anchor.

    A big revaluation, a record transfer, a long injury and another season of
    football. None of it was knowable on 1 July 2020.
    """
    t = {k: v.copy() for k, v in _base_tables().items()}
    t["player_valuations"] = pd.concat([t["player_valuations"], pd.DataFrame({
        "player_id": [1, 2], "date": pd.to_datetime(["2020-08-01", "2020-08-01"]),
        "market_value_in_eur": [90e6, 50e6]})], ignore_index=True)
    t["transfer_history"] = pd.concat([t["transfer_history"], pd.DataFrame({
        "player_id": [1, 2], "transfer_date": pd.to_datetime(["2020-08-01", "2020-09-01"]),
        "transfer_type": ["Transfer", "Loan"],
        "transfer_fee": [200_000_000, 0]})], ignore_index=True)
    t["player_injuries"] = pd.concat([t["player_injuries"], pd.DataFrame({
        "player_id": [1, 2], "from_date": pd.to_datetime(["2020-08-01", "2020-08-15"]),
        "end_date": pd.to_datetime(["2021-06-01", "2021-01-01"]),
        "days_missed": [304.0, 139.0]})], ignore_index=True)
    t["games"] = pd.concat([t["games"], pd.DataFrame({
        "game_id": [3], "competition_id": ["GB1"], "season": [2020],
        "home_club_id": [10], "away_club_id": [20]})], ignore_index=True)
    t["game_lineups"] = pd.concat([t["game_lineups"], pd.DataFrame({
        "game_id": [3, 3], "player_id": [1, 2], "club_id": [10, 20],
        "type": ["starting_lineup", "starting_lineup"]})], ignore_index=True)
    t["appearances"] = pd.concat([t["appearances"], pd.DataFrame({
        "game_id": [3, 3], "player_id": [1, 2], "player_club_id": [10, 20],
        "minutes_played": [90, 90], "goals": [5, 4],
        "assists": [5, 4]})], ignore_index=True)
    return t


def _cfg():
    return {
        "panel": {"anchor_month_day": "07-01", "min_anchor_season": 2020,
                  "max_value_staleness_days": 365, "leagues": ["GB1"]},
        "target": {"horizons": [1]},
        "split": {"values_available_until": "2020-06-12"},
    }


def _is_target(column: str) -> bool:
    return column.startswith("y_h") or column.startswith("value_h")


def test_the_pollution_actually_lands_somewhere():
    """Without this the leakage test could pass by doing nothing at all."""
    from pvf.features.build_panel import build_panel
    clean = build_panel(_base_tables(), _cfg()).set_index("player_id")
    dirty = build_panel(_polluted_tables(), _cfg()).set_index("player_id")
    assert pd.isna(clean.loc[1, "y_h1"])
    assert not pd.isna(dirty.loc[1, "y_h1"])


def test_no_feature_column_can_see_past_the_anchor():
    from pvf.features.build_panel import build_panel
    clean = build_panel(_base_tables(), _cfg()).set_index("player_id").sort_index()
    dirty = build_panel(_polluted_tables(), _cfg()).set_index("player_id").sort_index()

    assert list(clean.columns) == list(dirty.columns)
    assert len(clean) == len(dirty)

    features = [c for c in clean.columns if not _is_target(c)]
    differing = [c for c in features
                 if not clean[c].equals(dirty[c])]
    assert not differing, f"these feature columns saw the future: {differing}"
