"""Trained-from-scratch model: LightGBM quantile regressors, one set per horizon."""
import lightgbm as lgb
import numpy as np
import pandas as pd


class QuantileGBM:
    """Holds one LightGBM model per (horizon, quantile) pair."""

    def __init__(self, horizons: list[int], quantiles: list[float], params: dict | None = None,
                 seed: int = 42):
        self.horizons, self.quantiles = horizons, quantiles
        self.params = params or {}
        self.seed = seed
        self.models_: dict[tuple[int, float], object] = {}

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numeric features as float, the rest as categoricals fixed at fit time."""
        out = X[self.features_].copy()
        for c in self.features_:
            if c in self.categories_:
                # Unseen levels become NaN, which LightGBM routes like a missing value
                out[c] = pd.Categorical(out[c], categories=self.categories_[c])
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None,
            features: list[str]) -> "QuantileGBM":
        self.features_ = list(features)
        # Object, bool and category columns are categorical; levels come from training only
        self.categories_ = {
            c: sorted(train[c].dropna().unique().tolist(), key=str)
            for c in self.features_ if not pd.api.types.is_numeric_dtype(train[c])
            or pd.api.types.is_bool_dtype(train[c])}
        params = dict(self.params)
        stop_rounds = params.pop("early_stopping_rounds", None)
        self.models_ = {}
        for h in self.horizons:
            y_col = f"y_h{h}"
            tr = train[train[y_col].notna()]
            fit_kw = {}
            if val is not None and stop_rounds:
                va = val[val[y_col].notna()]
                fit_kw = {"eval_X": self._prepare(va), "eval_y": va[y_col],
                          "callbacks": [lgb.early_stopping(stop_rounds, verbose=False)]}
            for q in self.quantiles:
                model = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                          random_state=self.seed, **params)
                model.fit(self._prepare(tr), tr[y_col], **fit_kw)
                self.models_[(h, q)] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("QuantileGBM.predict called before fit")
        Xp = self._prepare(X)
        frames = []
        for h in self.horizons:
            preds = np.column_stack([self.models_[(h, q)].predict(Xp) for q in self.quantiles])
            # Separately trained quantiles can cross; sorting each row restores the order
            preds.sort(axis=1)
            cols = [f"q{round(q * 100)}" for q in self.quantiles]
            frame = pd.DataFrame(preds, index=X.index, columns=cols)
            frame.insert(0, "horizon", h)
            frames.append(frame)
        return pd.concat(frames)
