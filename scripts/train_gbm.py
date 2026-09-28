"""Cross-validate the LightGBM quantile model against the baselines on the 1000 samples.

Five folds grouped by player; only the 500 hand-collected rows are scored. Three GBM
variants: all 49 features, without the contract features (the ablation), and the 7 sample
features alone. With --ablation, instead drops one feature group at a time from the full
model (samples.FEATURE_GROUPS) and adds a core-only model. Every model is also compared
row by row with the full model, with a 95% bootstrap interval. Writes out-of-fold
predictions and metrics to data/processed/.

Usage:
    python scripts/train_gbm.py [--samples data/processed/samples.parquet] [--ablation]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.evaluation.cv import cross_validate, paired_diff, summarize  # noqa: E402
from pvf.features.samples import CORE_FEATURES, FEATURE_GROUPS, FEATURES, SAMPLE_FEATURES  # noqa: E402

CONTRACT_FEATURES = ["years_left", "contract_years", "years_into_contract"]


def main() -> None:
    cfg = load_config()
    processed = cfg["paths"]["processed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=processed / "samples.parquet")
    ap.add_argument("--ablation", action="store_true", help="drop one feature group at a time")
    args = ap.parse_args()

    samples = pd.read_parquet(args.samples)
    if args.ablation:
        feature_sets = {"gbm_all": FEATURES, "gbm_core": CORE_FEATURES}
        for group, cols in FEATURE_GROUPS.items():
            feature_sets[f"drop_{group}"] = [f for f in FEATURES if f not in cols]
    else:
        feature_sets = {
            "gbm_all": FEATURES,
            "gbm_no_contract": [f for f in FEATURES if f not in CONTRACT_FEATURES],
            "gbm_sample_only": SAMPLE_FEATURES,
        }
    m = cfg["model"]
    oof = cross_validate(samples, feature_sets, quantiles=m["quantiles"], params=m["gbm"],
                         n_splits=cfg["split"]["n_splits"], seed=cfg["project"]["seed"],
                         val_share=m["val_share"], min_count=m["baseline_min_count"])
    table = summarize(oof, samples)
    others = [m for m in oof["model"].unique() if m != "gbm_all"]
    diffs = paired_diff(oof, samples, reference="gbm_all", others=others)

    # Ablation runs keep their own files so they never overwrite the main CV results
    prefix = "ablation_" if args.ablation else ""
    oof.to_parquet(processed / f"{prefix}oof_predictions.parquet", index=False)
    table.to_csv(processed / f"{prefix}cv_metrics.csv", index=False)
    diffs.to_csv(processed / f"{prefix}vs_gbm_all.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(table[table["subset"] == "all"].to_string(index=False))
        print("\nlog MAE minus gbm_all (positive = worse), 95% bootstrap interval:")
        print(diffs.to_string(index=False))
    print(f"\nwrote {prefix}oof_predictions.parquet, {prefix}cv_metrics.csv, "
          f"{prefix}vs_gbm_all.csv to {processed}")


if __name__ == "__main__":
    main()
