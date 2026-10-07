"""Score the off-the-shelf Chronos-Bolt model against the from-scratch GBM on the same rows.

Chronos is zero-shot, so it needs no fold to train on; it is still scored fold by fold on the
same 500 hand-collected rows as train_gbm.py, so the numbers line up. Its band is reported raw
and after a split-conformal shift fitted on each fold's training rows only. A third variant
feeds Chronos' forecast to the core GBM as two extra features, asking whether the pretrained
model knows something about the value path that age, position and value do not.

Usage:
    python scripts/eval_chronos.py --data-dir <Dataset root> [--model amazon/chronos-bolt-small]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.data.load import read_table  # noqa: E402
from pvf.evaluation.cv import TARGET, _split_val, cross_validate, paired_diff, summarize  # noqa: E402
from pvf.evaluation.splits import player_folds  # noqa: E402
from pvf.features.samples import CORE_FEATURES  # noqa: E402
from pvf.models.gbm import QuantileGBM  # noqa: E402
from pvf.models.foundation import (DEFAULT_MODEL, ChronosForecaster, conformal_shift,  # noqa: E402
                                   contexts_for)

STACK_FEATURES = ["chronos_q50", "chronos_width"]


def chronos_oof(samples: pd.DataFrame, preds: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Out-of-fold rows for Chronos raw and conformal, in the same layout cross_validate uses."""
    out = []
    for k, (train_idx, test_idx) in enumerate(player_folds(samples, n_splits, seed)):
        tr, te = preds.loc[train_idx], preds.loc[test_idx]
        shift = conformal_shift(tr["q10"].to_numpy(), tr["q90"].to_numpy(),
                                samples.loc[train_idx, TARGET].to_numpy())
        for name, adj in [("chronos_zero_shot", 0.0), ("chronos_conformal", shift)]:
            out.append(pd.DataFrame({"fold": k, "row": test_idx, "model": name,
                                     "pred": te["q50"].to_numpy(),
                                     "q10": te["q10"].to_numpy() - adj,
                                     "q90": te["q90"].to_numpy() + adj}))
    return pd.concat(out, ignore_index=True)


def coverage(oof: pd.DataFrame, samples: pd.DataFrame) -> pd.Series:
    """80% band coverage per model over the rows that have a band (Chronos has none on fallbacks)."""
    d = oof.join(samples[TARGET], on="row").dropna(subset=["q10", "q90"])
    inside = (d[TARGET] >= d["q10"]) & (d[TARGET] <= d["q90"])
    return inside.groupby(d["model"]).mean()


def league_check(samples: pd.DataFrame, panel: pd.DataFrame, valuations: pd.DataFrame,
                 forecaster: ChronosForecaster, cfg: dict, n_per_year: int = 1000) -> pd.DataFrame:
    """Both models on ordinary league players: the population the app serves, not future stars.

    Panel rows anchored 1 Jul 2023 and 2024 with an observed outcome, none of them a training
    player, sampled with the project seed. The GBM is the app's model: the core features,
    fitted on all 1,000 samples as export_bundle.py does.
    """
    m, seed = cfg["model"], cfg["project"]["seed"]
    pool = panel[panel["is_primary"] & panel["y_h1"].notna() & panel["value_eur"].notna()
                 & panel["anchor_date"].dt.year.isin([2023, 2024])
                 & ~panel["player_id"].isin(samples["player_id"])]
    rows = (pool.groupby(pool["anchor_date"].dt.year, group_keys=False)
            .apply(lambda g: g.sample(min(n_per_year, len(g)), random_state=seed))
            .rename(columns={"value_eur": "value_now"}).reset_index(drop=True))
    fit_part, val_part = _split_val(samples, m["val_share"], seed)
    gbm = (QuantileGBM([1], m["quantiles"], m["gbm"], seed)
           .fit(fit_part, val_part, CORE_FEATURES).calibrate(val_part))
    g = gbm.predict(rows)
    contexts = contexts_for(valuations, rows["player_id"], rows["anchor_date"])
    c = forecaster.predict(contexts, rows["value_now"].to_numpy())
    y = rows["y_h1"].to_numpy()
    out = []
    for name, q10, q50, q90 in [("no_change", None, np.zeros(len(y)), None),
                                ("gbm_core (app model)", g["q10"], g["q50"], g["q90"]),
                                ("chronos_zero_shot", c["q10"], c["q50"], c["q90"])]:
        q50 = np.asarray(q50)
        row = {"model": name, "n": len(y), "mae_log": float(np.mean(np.abs(y - q50))),
               "median_pred": float(np.median(q50)), "share_pred_rise": float((q50 > 0).mean())}
        if q10 is not None:
            lo, hi = np.asarray(q10), np.asarray(q90)
            ok = np.isfinite(lo)
            row["coverage_80"] = float(((y >= lo) & (y <= hi))[ok].mean())
        out.append(row)
    out.append({"model": "actual outcome", "n": len(y), "median_pred": float(np.median(y)),
                "share_pred_rise": float((y > 0).mean())})
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config()
    processed, m, seed = cfg["paths"]["processed"], cfg["model"], cfg["project"]["seed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR"),
                    help="Dataset root holding player-scores/ (defaults to FV_DATA_DIR)")
    ap.add_argument("--samples", type=Path, default=processed / "samples.parquet")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    samples = pd.read_parquet(args.samples)
    samples = samples[samples[TARGET].notna()]
    cache = processed / f"chronos_preds_{args.model.split('/')[-1]}.parquet"
    if cache.exists():
        preds = pd.read_parquet(cache)
        print(f"read cached Chronos forecasts from {cache.name}")
    else:
        kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else cfg["paths"]["raw_player_scores"]
        valuations = read_table(kaggle, "player_valuations")
        contexts = contexts_for(valuations, samples["player_id"], samples["anchor_date"])
        preds = ChronosForecaster(args.model).predict(contexts, samples["value_now"].to_numpy())
        preds.index = samples.index
        preds["context_months"] = [len(c) for c in contexts]
        preds.to_parquet(cache)
        print(f"Chronos forecast {len(preds)} rows ({int(preds['fallback'].sum())} fell back "
              f"to no change), median context {preds['context_months'].median():.0f} months")

    # Stacked variant: Chronos' view of the value path as two features for the core GBM
    stacked = samples.assign(chronos_q50=preds["q50"], chronos_width=preds["q90"] - preds["q10"])
    oof = cross_validate(stacked, {"gbm_core": CORE_FEATURES,
                                   "gbm_core_plus_chronos": CORE_FEATURES + STACK_FEATURES},
                         quantiles=m["quantiles"], params=m["gbm"], n_splits=cfg["split"]["n_splits"],
                         seed=seed, val_share=m["val_share"], min_count=m["baseline_min_count"])
    oof = pd.concat([oof, chronos_oof(samples, preds, cfg["split"]["n_splits"], seed)],
                    ignore_index=True)
    table = summarize(oof, samples)
    others = [x for x in oof["model"].unique() if x != "gbm_core"]
    diffs = paired_diff(oof, samples, reference="gbm_core", others=others)

    oof.to_parquet(processed / "chronos_oof_predictions.parquet", index=False)
    table.to_csv(processed / "chronos_cv_metrics.csv", index=False)
    diffs.to_csv(processed / "chronos_vs_gbm_core.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(table[table["subset"] == "all"].to_string(index=False))
        print("\nlog MAE minus gbm_core (positive = worse), 95% bootstrap interval:")
        print(diffs.to_string(index=False))
    print("\n80% band coverage over rows with a band:")
    print(coverage(oof, samples).round(3).to_string())
    manual = samples["origin"] == "manual"
    print(f"\nmanual rows that rose: {(samples.loc[manual, TARGET] > 0).mean():.0%}; "
          f"Chronos predicted a rise for {(preds.loc[manual, 'q50'] > 0).mean():.0%}")

    panel = pd.read_parquet(processed / "panel.parquet")
    kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else cfg["paths"]["raw_player_scores"]
    league = league_check(samples, panel, read_table(kaggle, "player_valuations"),
                          ChronosForecaster(args.model), cfg)
    league.to_csv(processed / "league_check.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print("\nOrdinary league players, 2023 and 2024 anchors, no training players:")
        print(league.to_string(index=False))


if __name__ == "__main__":
    main()
