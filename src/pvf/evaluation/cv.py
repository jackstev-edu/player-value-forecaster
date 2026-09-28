"""Out-of-fold predictions for the GBM variants and baselines under player-grouped CV.

Each fold trains on the other players' rows (manual and span), early-stops on a held-out
share of those training players, and scores only the fold's manual rows (splits.py).
"""
import numpy as np
import pandas as pd

from pvf.evaluation.metrics import interval_coverage, mae_log, median_ape_eur
from pvf.evaluation.splits import player_folds
from pvf.models.baselines import AgePositionCurve, NoChange
from pvf.models.gbm import QuantileGBM

TARGET = "y_h1"


def _split_val(train: pd.DataFrame, share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole players, so early stopping is judged on unseen players too."""
    players = train["player_id"].unique()
    n_val = max(1, int(round(share * len(players))))
    val_p = np.random.default_rng(seed).choice(players, n_val, replace=False)
    is_val = train["player_id"].isin(val_p)
    return train[~is_val], train[is_val]


def cross_validate(samples: pd.DataFrame, feature_sets: dict[str, list[str]], *,
                   quantiles: list[float], params: dict, n_splits: int = 5, seed: int = 42,
                   val_share: float = 0.15, min_count: int = 5) -> pd.DataFrame:
    """Long frame of out-of-fold predictions: fold, row, model, pred (median), q10, q90."""
    samples = samples[samples[TARGET].notna()]
    out = []
    for k, (train_idx, test_idx) in enumerate(player_folds(samples, n_splits, seed)):
        train, test = samples.loc[train_idx], samples.loc[test_idx]
        fit_part, val_part = _split_val(train, val_share, seed + k)

        def add(model: str, pred, lo=np.nan, hi=np.nan):
            out.append(pd.DataFrame({"fold": k, "row": test.index, "model": model,
                                     "pred": pred, "q10": lo, "q90": hi}))

        # Baselines see every training row; they have no early stopping to feed
        add("no_change", NoChange().fit(train, train[TARGET]).predict(test))
        add("age_position",
            AgePositionCurve(min_count=min_count).fit(train, train[TARGET]).predict(test))
        for name, features in feature_sets.items():
            p = QuantileGBM([1], quantiles, params, seed).fit(fit_part, val_part, features).predict(test)
            add(name, p["q50"].to_numpy(), p["q10"].to_numpy(), p["q90"].to_numpy())
    return pd.concat(out, ignore_index=True)


def summarize(oof: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Metrics per model on all scored rows and without the value-backfilled rows (#44)."""
    df = oof.join(samples[[TARGET, "value_now", "value_backfilled"]], on="row")
    subsets = {"all": df, "no_backfill": df[~df["value_backfilled"].astype(bool)]}
    rows = []
    for subset, d in subsets.items():
        for model, g in d.groupby("model", sort=False):
            has_band = g["q10"].notna().all()
            rows.append({
                "subset": subset, "model": model, "n": len(g),
                "mae_log": mae_log(g[TARGET], g["pred"]),
                "median_ape_eur": median_ape_eur(g["value_now"].to_numpy(), g[TARGET].to_numpy(),
                                                 g["pred"].to_numpy()),
                "coverage_80": interval_coverage(g[TARGET], g["q10"], g["q90"]) if has_band else np.nan,
            })
    return pd.DataFrame(rows)
