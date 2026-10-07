"""Chronos wrapper: the context never reads past the anchor, and missing history falls back."""
import numpy as np
import pandas as pd
import pytest

from pvf.models.foundation import ChronosForecaster, conformal_shift, contexts_for, monthly_context

torch = pytest.importorskip("torch")


def _vals():
    return pd.DataFrame({
        "player_id": [1, 1, 1, 2],
        "date": pd.to_datetime(["2023-01-15", "2023-06-10", "2024-03-01", "2023-05-01"]),
        "market_value_in_eur": [10e6, 20e6, 99e6, 5e6],
    })


def test_context_is_a_monthly_step_series_ending_on_the_anchor():
    ctx = monthly_context(_vals(), 1, pd.Timestamp("2023-07-01"))
    # 1 Feb..1 Jun hold the January value; 1 Jul takes the June one
    assert ctx.tolist() == [10e6] * 5 + [20e6]


def test_context_never_reads_a_valuation_after_the_anchor():
    ctx = monthly_context(_vals(), 1, pd.Timestamp("2024-02-29"))
    assert 99e6 not in ctx


def test_no_history_gives_an_empty_context():
    assert monthly_context(_vals(), 1, pd.Timestamp("2022-07-01")).size == 0
    assert contexts_for(_vals(), [3], [pd.Timestamp("2023-07-01")])[0].size == 0


class _FakePipeline:
    """Forecasts 0.5x, 1x and 2x the last context value at every step."""

    def predict_quantiles(self, batch, prediction_length, quantile_levels):
        last = torch.stack([c[-1] for c in batch])
        q = torch.stack([last * 0.5, last, last * 2.0], dim=-1)
        return q[:, None, :].repeat(1, prediction_length, 1), None


def test_predict_returns_log_ratios_against_value_now():
    f = ChronosForecaster(pipeline=_FakePipeline())
    out = f.predict([np.array([10e6, 20e6])], np.array([20e6]))
    assert out[["q10", "q50", "q90"]].iloc[0].tolist() == pytest.approx(
        [np.log(0.5), 0.0, np.log(2.0)])
    assert not out["fallback"].iloc[0]


def test_missing_history_falls_back_to_no_change_without_a_band():
    f = ChronosForecaster(pipeline=_FakePipeline())
    out = f.predict([np.array([]), np.array([4e6])], np.array([3e6, np.nan]))
    assert out["fallback"].tolist() == [True, True]
    assert out["q50"].tolist() == [0.0, 0.0]
    assert out["q10"].isna().all()


def test_conformal_shift_widens_a_band_that_is_too_narrow():
    y = np.linspace(-1, 1, 101)
    shift = conformal_shift(np.full(101, -0.1), np.full(101, 0.1), y, level=0.8)
    covered = ((y >= -0.1 - shift) & (y <= 0.1 + shift)).mean()
    assert shift > 0 and covered >= 0.8
