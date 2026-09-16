"""Trained-from-scratch model: LightGBM quantile regressors, one set per horizon."""
import pandas as pd


class QuantileGBM:
    """Holds one LightGBM model per (horizon, quantile) pair."""

    def __init__(self, horizons: list[int], quantiles: list[float], params: dict | None = None,
                 seed: int = 42):
        self.horizons, self.quantiles = horizons, quantiles
        self.params = params or {}
        self.seed = seed
        self.models_: dict[tuple[int, float], object] = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> "QuantileGBM":
        # >>> 1. LOOP HORIZONS x QUANTILES, drop rows with NaN y_h{h} for that horizon <<<

        # >>> 2. BUILD lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed) <<<

        # >>> 3. EARLY STOP ON val, store in self.models_[(h, q)] <<<
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # >>> 4. RETURN long frame: index, horizon, q10, q50, q90 in log-ratio space <<<

        # >>> 5. SORT quantiles per row so q10 <= q50 <= q90 (quantile crossing fix) <<<
        raise NotImplementedError
