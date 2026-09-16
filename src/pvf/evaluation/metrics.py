"""Metrics from the Analysis Guide section 6."""
import numpy as np
import pandas as pd


def mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error on log ratio, roughly a percentage error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def median_ape_eur(value_now: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error after converting back to euros."""
    true_eur = value_now * np.exp(y_true)
    pred_eur = value_now * np.exp(y_pred)
    return float(np.median(np.abs(pred_eur - true_eur) / true_eur))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of truths inside [lo, hi]; target is 0.80 for a 10 to 90 band."""
    y = np.asarray(y_true)
    return float(np.mean((y >= lo) & (y <= hi)))


def breakdown(df: pd.DataFrame, by: str, y_col: str = "y", pred_col: str = "pred") -> pd.DataFrame:
    """MAE on log ratio per group, e.g. by age band, league or value tier."""
    return (df.groupby(by)
              .apply(lambda g: pd.Series({"n": len(g), "mae_log": mae_log(g[y_col], g[pred_col])}),
                     include_groups=False)
              .reset_index())
