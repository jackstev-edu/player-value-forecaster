"""Cross-validate the LightGBM quantile model against the baselines on the 1000 samples.

Five folds grouped by player; only the 500 hand-collected rows are scored. Three GBM
variants: all 49 features, without the contract features (the ablation), and the 7 sample
features alone. Writes out-of-fold predictions and a metrics table to data/processed/.

Usage:
    python scripts/train_gbm.py [--samples data/processed/samples.parquet]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.evaluation.cv import cross_validate, summarize  # noqa: E402
from pvf.features.samples import FEATURES, SAMPLE_FEATURES  # noqa: E402

CONTRACT_FEATURES = ["years_left", "contract_years", "years_into_contract"]


def main() -> None:
    cfg = load_config()
    processed = cfg["paths"]["processed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=processed / "samples.parquet")
    args = ap.parse_args()

    samples = pd.read_parquet(args.samples)
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

    oof.to_parquet(processed / "oof_predictions.parquet", index=False)
    table.to_csv(processed / "cv_metrics.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print(f"\nwrote {processed / 'oof_predictions.parquet'} and {processed / 'cv_metrics.csv'}")


if __name__ == "__main__":
    main()
