"""Out-of-fold predictions for the GBM variants and baselines under player-grouped CV.

Each fold trains on the other players' rows (manual and span), early-stops on a held-out
share of those training players, and scores only the fold's manual rows (splits.py).
The same held-out players then calibrate the GBM band to its nominal 80% (gbm.py).
"""
import numpy as np
import pandas as pd

from pvf.evaluation.metrics import interval_coverage, mae_log, median_ape_eur
from pvf.evaluation.splits import player_folds
from pvf.models.baselines import AgePositionCurve, LinearBaseline, NoChange
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
                   val_share: float = 0.15, min_count: int = 5,
                   calibrate: bool = True) -> pd.DataFrame:
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
        add("linear", LinearBaseline().fit(train, train[TARGET]).predict(test))
        for name, features in feature_sets.items():
            model = QuantileGBM([1], quantiles, params, seed).fit(fit_part, val_part, features)
            if calibrate:
                model.calibrate(val_part)
            p = model.predict(test)
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


def paired_diff(oof: pd.DataFrame, samples: pd.DataFrame, reference: str, others: list[str],
                n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Mean absolute log error of each model minus the reference's, row by row, with a
    95% bootstrap interval. Positive means worse than the reference. With 500 scored rows,
    gaps inside the interval are noise, so an ablation reads the interval, not the mean.
    """
    df = oof.join(samples[TARGET], on="row")
    err = (df[TARGET] - df["pred"]).abs().groupby([df["row"], df["model"]]).first().unstack()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    rows = []
    for m in others:
        d = (err[m] - err[reference]).to_numpy()
        boots = d[idx].mean(axis=1)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if d.any() else (0.0, 0.0))
        rows.append({"model": m, "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
                     "share_better": float((d < 0).mean())})
    return pd.DataFrame(rows)
