"""Second rubric model type: pretrained time-series model on value histories.

Candidate: Amazon Chronos (off-the-shelf zero-shot first, fine-tune if time allows).
Only the player's own value history is used, so it doubles as a
"history-only" comparison against the context-aware GBM.
"""
import pandas as pd


def to_yearly_series(valuations: pd.DataFrame, player_id: int, anchor: pd.Timestamp) -> pd.Series:
    """Resample one player's values before the anchor to a regular grid."""
    s = valuations.loc[valuations["player_id"] == player_id].set_index("date")["market_value_in_eur"]
    # Only history strictly before the anchor is visible
    s = s[s.index < anchor].sort_index()
    # >>> 1. RESAMPLE to a fixed step (quarterly?) with forward fill, log transform <<<
    return s


class ChronosForecaster:
    """Wrapper exposing the same predict() shape as QuantileGBM."""

    def __init__(self, model_name: str = "amazon/chronos-bolt-small", device: str = "cpu"):
        self.model_name, self.device = model_name, device
        self.pipeline_ = None

    def load(self) -> "ChronosForecaster":
        # >>> 2. LOAD pipeline from chronos package, keep torch import local <<<
        raise NotImplementedError

    def predict(self, panel: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
        # >>> 3. BATCH series per anchor row, forecast enough steps to reach 3 seasons <<<

        # >>> 4. CONVERT quantile paths to log ratios vs anchor value at each horizon <<<
        raise NotImplementedError
