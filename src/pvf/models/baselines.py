"""Baselines every ML model must beat (Analysis Guide section 4)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class NoChange:
    """Predict that value stays flat, i.e. a log ratio of zero."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NoChange":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class AgePositionCurve:
    """Predict the training median log ratio for the same age and position."""

    def __init__(self, age_col: str = "age", pos_col: str = "position", min_count: int = 20):
        self.age_col, self.pos_col, self.min_count = age_col, pos_col, min_count

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AgePositionCurve":
        # Integer ages keep cells populated enough for a stable median
        df = pd.DataFrame({"age": X[self.age_col].astype(int), "pos": X[self.pos_col], "y": y})
        g = df.groupby(["age", "pos"])["y"].agg(["median", "size"])
        self.table_ = g.loc[g["size"] >= self.min_count, "median"]
        # Age-only fallback for sparse age x position cells
        self.age_only_ = df.groupby("age")["y"].median()
        self.global_ = float(df["y"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = list(zip(X[self.age_col].astype(int), X[self.pos_col]))
        out = pd.Series(keys).map(self.table_.to_dict())
        out = out.fillna(X[self.age_col].astype(int).reset_index(drop=True).map(self.age_only_))
        return out.fillna(self.global_).to_numpy()


class LinearBaseline:
    """Ridge regression on age, age squared, log current value and position.

    The strongest baseline that stays explainable: it knows cheap players tend to rise and
    expensive ones to settle, which the age and position curve cannot see.
    """

    def __init__(self, age_col: str = "age", pos_col: str = "position",
                 value_col: str = "value_now", alpha: float = 1.0):
        self.age_col, self.pos_col, self.value_col, self.alpha = age_col, pos_col, value_col, alpha

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        age = X[self.age_col].to_numpy(float) - 26.0
        logv = np.log(X[self.value_col].to_numpy(float))
        # Unseen positions get all-zero dummies, i.e. the reference position
        pos = np.column_stack([(X[self.pos_col] == p).to_numpy(float) for p in self.positions_])
        return np.column_stack([age, age ** 2, logv, pos])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearBaseline":
        self.positions_ = sorted(X[self.pos_col].dropna().unique().tolist())[1:]
        self.model_ = Ridge(alpha=self.alpha).fit(self._design(X), np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._design(X))
