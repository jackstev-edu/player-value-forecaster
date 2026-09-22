import numpy as np
import pandas as pd
import pytest

from pvf.data.load import read_table


def _write(folder, table, df):
    df.to_parquet(folder / f"{table}.parquet", index=False)


def test_transfer_history_dates_are_parsed(tmp_path):
    """`transfer_history` lives in football-datasets and `transfers` in player-scores.
    Only the second was listed, so the first came back as strings — and a string date
    compares lexicographically without raising, so the gap fails silently."""
    _write(tmp_path, "transfer_history", pd.DataFrame({
        "player_id": [1, 2],
        "transfer_date": ["2019-07-01", "2020-01-31"],
        "transfer_fee": [0, 1000],
    }))
    out = read_table(tmp_path, "transfer_history")
    assert pd.api.types.is_datetime64_any_dtype(out["transfer_date"])
    assert out["transfer_date"].iloc[0] == pd.Timestamp("2019-07-01")


def test_a_malformed_date_becomes_nat_rather_than_raising(tmp_path):
    _write(tmp_path, "transfer_history", pd.DataFrame({
        "player_id": [1, 2],
        "transfer_date": ["2019-07-01", "not a date"],
    }))
    out = read_table(tmp_path, "transfer_history")
    assert pd.isna(out["transfer_date"].iloc[1])


def test_injury_dates_are_parsed(tmp_path):
    _write(tmp_path, "player_injuries", pd.DataFrame({
        "player_id": [1],
        "from_date": ["2019-08-01"],
        "end_date": ["2019-09-15"],
        "days_missed": [45.0],
    }))
    out = read_table(tmp_path, "player_injuries")
    assert out["from_date"].iloc[0] == pd.Timestamp("2019-08-01")
    assert out["end_date"].iloc[0] == pd.Timestamp("2019-09-15")


def test_zero_heights_become_unknown(tmp_path):
    _write(tmp_path, "player_profiles", pd.DataFrame({
        "player_id": [1, 2], "height": [180.0, 0.0],
        "date_of_birth": ["1995-01-01", "1996-01-01"],
    }))
    out = read_table(tmp_path, "player_profiles")
    assert out["height"].iloc[0] == 180.0
    assert pd.isna(out["height"].iloc[1])
