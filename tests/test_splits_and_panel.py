import numpy as np
import pandas as pd

from pvf.evaluation.splits import rolling_origins, time_split
from pvf.features.build_panel import add_targets, value_asof


def _valuations():
    return pd.DataFrame({
        "player_id": [1, 1, 1, 1],
        "date": pd.to_datetime(["2019-06-01", "2020-06-01", "2021-06-01", "2022-06-01"]),
        "market_value_in_eur": [1e6, 2e6, 4e6, 4e6],
    })


def test_value_asof_never_looks_forward():
    anchors = pd.DataFrame({"player_id": [1], "anchor_date": pd.to_datetime(["2020-05-31"])})
    out = value_asof(_valuations(), anchors)
    assert out["value_eur"].iloc[0] == 1e6


def test_targets_are_log_ratios():
    panel = value_asof(_valuations(), pd.DataFrame(
        {"player_id": [1], "anchor_date": pd.to_datetime(["2020-07-01"])}))
    panel = add_targets(panel, _valuations(), [1, 2], max_staleness_days=365)
    assert np.isclose(panel["y_h1"].iloc[0], np.log(2))
    assert np.isclose(panel["y_h2"].iloc[0], np.log(2))


def test_time_split_purges_overlap():
    panel = pd.DataFrame({
        "anchor_date": pd.to_datetime([f"{y}-07-01" for y in range(2015, 2024)]),
        "y_h3": 0.1,
    })
    s = time_split(panel, horizon=3, train_last_season=2020, val_season=2021, test_season=2023)
    # Train targets must resolve by the 2023 anchor, so the last train anchor is 2020
    assert s["train"]["anchor_date"].dt.year.max() == 2020
    assert s["val"].empty  # 2021 + 3 > 2023, correctly purged


def test_rolling_origins_respect_data_end():
    assert rolling_origins([2022, 2023, 2024, 2025], 3, "2026-06-12") == [2022, 2023]
