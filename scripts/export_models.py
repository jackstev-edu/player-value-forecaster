"""Stage the two Hugging Face model repos: the trained LightGBM and the Chronos configuration.

lgbm-core: the app's model, fitted exactly as export_bundle.py fits it (same seed, same split),
saved as LightGBM text files plus meta.json. chronos: no weights of our own (the model is used
zero-shot), so the repo holds how we call it, its conformal band shift, our wrapper code and
the evaluation. Both get metrics.json from the CV and league-player runs, and a card from hf/.

Run scripts/train_gbm.py and scripts/eval_chronos.py first.

Usage:
    python scripts/export_models.py [--out data/processed/hf_models]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.evaluation.cv import TARGET, _split_val  # noqa: E402
from pvf.features.samples import CORE_FEATURES  # noqa: E402
from pvf.models.foundation import DEFAULT_MODEL, QUANTILES, conformal_shift  # noqa: E402
from pvf.models.gbm import QuantileGBM  # noqa: E402

MODEL_VERSION = "lgbm-core-v1"


def metric_rows(table: pd.DataFrame, models: list[str]) -> list[dict]:
    keep = table[(table["subset"] == "all") & table["model"].isin(models)]
    return keep.round(4).to_dict("records")


def main() -> None:
    cfg = load_config()
    processed, m, seed = cfg["paths"]["processed"], cfg["model"], cfg["project"]["seed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=processed / "hf_models")
    args = ap.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)

    samples = pd.read_parquet(processed / "samples.parquet")
    samples = samples[samples[TARGET].notna()]
    cv = pd.read_csv(processed / "chronos_cv_metrics.csv")
    league = pd.read_csv(processed / "league_check.csv").round(4).to_dict("records")
    diffs = pd.read_csv(processed / "chronos_vs_gbm_core.csv").round(4).to_dict("records")
    cv_rows = metric_rows(cv, ["no_change", "age_position", "linear", "gbm_core",
                               "gbm_core_plus_chronos", "chronos_zero_shot", "chronos_conformal"])

    # 1. The trained-from-scratch model, identical to the one behind the app's bundle
    lgbm_dir = args.out / "lgbm-core"
    fit_part, val_part = _split_val(samples, m["val_share"], seed)
    model = (QuantileGBM([1], m["quantiles"], m["gbm"], seed)
             .fit(fit_part, val_part, CORE_FEATURES).calibrate(val_part))
    model.save(lgbm_dir)
    meta = {"model_version": MODEL_VERSION, "features": CORE_FEATURES, "target": "y_h1 = "
            "log(value 1 season after anchor / value at anchor)", "train_rows": len(fit_part),
            "early_stop_and_calibration_rows": len(val_part), "params": m["gbm"],
            "band_shift": model.band_adjust_[1], "cv_5fold_player_grouped": cv_rows,
            "paired_vs_gbm_core": diffs, "league_players_check": league}
    (lgbm_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.copy(ROOT / "hf" / "lgbm_card.md", lgbm_dir / "README.md")

    # 2. The off-the-shelf model: configuration and evaluation, weights stay with Amazon
    chronos_dir = args.out / "chronos"
    chronos_dir.mkdir(parents=True)
    preds = pd.read_parquet(processed / f"chronos_preds_{DEFAULT_MODEL.split('/')[-1]}.parquet")
    shift = conformal_shift(preds["q10"].to_numpy(), preds["q90"].to_numpy(),
                            samples.loc[preds.index, TARGET].to_numpy())
    config = {"base_model": DEFAULT_MODEL, "prediction_length_months": 12,
              "quantile_levels": QUANTILES, "context": "monthly step series of the player's "
              "Transfermarkt value, first valuation to the anchor date, at most 240 months",
              "output": "log(forecast at month 12 / value at anchor)",
              "fallback": "no history or no current value -> no-change forecast, no band",
              "conformal_band_shift_on_1000_samples": shift}
    (chronos_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (chronos_dir / "metrics.json").write_text(json.dumps(
        {"cv_5fold_player_grouped": cv_rows, "paired_vs_gbm_core": diffs,
         "league_players_check": league}, indent=2), encoding="utf-8")
    shutil.copy(ROOT / "src" / "pvf" / "models" / "foundation.py", chronos_dir / "chronos_value.py")
    shutil.copy(ROOT / "hf" / "chronos_card.md", chronos_dir / "README.md")
    print(f"band shift lgbm {model.band_adjust_[1]:+.3f}, chronos conformal shift {shift:+.3f}")
    print(f"staged {lgbm_dir} and {chronos_dir}")


if __name__ == "__main__":
    main()
