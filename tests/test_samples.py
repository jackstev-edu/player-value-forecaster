"""Turning contract rows plus Kaggle valuations into model samples."""
import numpy as np
import pandas as pd
import pytest

from pvf.features.samples import FEATURES, build_samples

T = pd.Timestamp


def contracts(start="2020-07-01", end="2024-06-30", anchor="2021-07-01", **extra):
    row = {"rank": 1, "player_id": 1, "anchor_date": T(anchor),
           "contract_start": T(start), "contract_end": T(end)}
    return pd.DataFrame([{**row, **extra}])


def valuations(*points, player_id=1):
    return pd.DataFrame([{"player_id": player_id, "date": T(d), "market_value_in_eur": v}
                         for d, v in points])


PLAYERS = pd.DataFrame([{"player_id": 1, "date_of_birth": T("2000-01-01"), "position": "Attack"}])
BASE = valuations(("2020-06-01", 10e6), ("2021-06-01", 20e6), ("2022-06-01", 40e6))


def build(c=None, v=BASE, **kw):
    return build_samples(contracts() if c is None else c, v, PLAYERS, **kw).iloc[0]


def test_value_now_is_the_latest_valuation_on_or_before_the_anchor():
    assert build()["value_now"] == 20e6


def test_target_is_log_ratio_one_season_later():
    assert build()["y_h1"] == pytest.approx(np.log(40e6 / 20e6))


def test_summer_signing_moves_the_anchor_to_the_signing_date():
    row = build(contracts(start="2021-08-15", end="2026-06-30"))
    assert row["anchor_date"] == T("2021-08-15")
    assert row["season"] == 2021


def test_contract_features_are_measured_in_years_at_the_anchor():
    row = build()
    assert row["years_left"] == pytest.approx((T("2024-06-30") - T("2021-07-01")).days / 365.25)
    assert row["contract_years"] == pytest.approx((T("2024-06-30") - T("2020-07-01")).days / 365.25)
    assert row["years_into_contract"] == pytest.approx(1.0, abs=0.01)


def test_age_and_position_come_from_the_player_table():
    row = build()
    assert row["age"] == pytest.approx(21.5, abs=0.01)
    assert row["position"] == "Attack"


def test_twelve_month_change_uses_the_value_a_year_before():
    assert build()["value_change_12m"] == pytest.approx(np.log(20e6 / 10e6))


def test_no_target_when_no_new_valuation_arrives():
    v = valuations(("2020-06-01", 10e6), ("2021-06-01", 20e6))
    assert np.isnan(build(v=v)["y_h1"])


def test_no_target_when_the_horizon_passes_the_last_valuation_date():
    assert np.isnan(build(values_until="2022-05-01")["y_h1"])


def test_stale_anchor_value_gives_no_sample_value():
    v = valuations(("2019-01-01", 10e6), ("2022-06-01", 40e6))
    row = build(v=v)
    assert np.isnan(row["value_now"]) and np.isnan(row["y_h1"])


def test_origin_defaults_to_manual():
    assert build()["origin"] == "manual"


def test_features_do_not_move_when_later_valuations_change():
    later = valuations(("2021-09-01", 99e6), ("2021-12-01", 1e6))
    before = build_samples(contracts(), BASE, PLAYERS)
    after = build_samples(contracts(), pd.concat([BASE, later]), PLAYERS)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
