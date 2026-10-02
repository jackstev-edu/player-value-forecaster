import numpy as np
import pandas as pd

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
