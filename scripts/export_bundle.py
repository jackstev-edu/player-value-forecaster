"""Fit the final core model and write the real prediction bundle the app reads.

The core model (age, position, current value, 12-month change) matched the full 49-feature
model in the ablation (decision #47), and it needs no contract or panel row, so it can score
every league player. It trains on all 1000 samples, early-stops and calibrates its band on
held-out players, then forecasts one season ahead for each primary panel player at the anchor.

Usage:
    python scripts/export_bundle.py --data-dir <Dataset root> [--anchor 2026-07-01]
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
from pvf.evaluation.cv import _split_val  # noqa: E402
from pvf.export.predictions import write_bundle  # noqa: E402
from pvf.export.real import forecasts_table, history_table, players_table, universe  # noqa: E402
from pvf.features.samples import CORE_FEATURES, core_features_at  # noqa: E402
from pvf.models.gbm import QuantileGBM  # noqa: E402

MODEL_VERSION = "lgbm-core-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR"),
                    help="Dataset root holding player-scores/ (defaults to FV_DATA_DIR)")
    ap.add_argument("--anchor", default="2026-07-01", help="Date the forecasts start from")
    args = ap.parse_args()
    cfg = load_config()
    paths, m, seed = cfg["paths"], cfg["model"], cfg["project"]["seed"]
    anchor = pd.Timestamp(args.anchor)

    samples = pd.read_parquet(paths["processed"] / "samples.parquet")
    samples = samples[samples["y_h1"].notna()]
    fit_part, val_part = _split_val(samples, m["val_share"], seed)
    model = (QuantileGBM([1], m["quantiles"], m["gbm"], seed)
             .fit(fit_part, val_part, CORE_FEATURES).calibrate(val_part))
    print(f"trained on {len(fit_part)} rows, early-stopped and calibrated on {len(val_part)} "
          f"(band shift {model.band_adjust_[1]:+.3f})")

    kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else paths["raw_player_scores"]
    valuations = read_table(kaggle, "player_valuations")
    kaggle_players = read_table(kaggle, "players")
    clubs = read_table(kaggle, "clubs")
    panel = pd.read_parquet(paths["processed"] / "panel.parquet")

    univ = universe(panel, anchor)
    core = core_features_at(univ["player_id"], pd.Series(anchor, index=univ.index), valuations,
                            kaggle_players,
                            max_staleness_days=cfg["samples"]["max_value_staleness_days"])
    players = players_table(univ, core, kaggle_players, clubs, anchor)
    print(f"universe {len(univ)} players, {len(players)} with a current value")

    valued = core["value_now"].notna()
    q = model.predict(core[valued].set_index(univ.loc[valued, "player_id"]))
    current = players.set_index("player_id")["current_value_eur"]
    forecasts = forecasts_table(current, q, anchor, MODEL_VERSION)
    history = history_table(valuations, players["player_id"], anchor)

    rise = float((q["q50"] > 0).mean())
    notes = (f"One-season forecasts from {anchor.date()} by the 4-feature core LightGBM, trained on "
             f"the 1000 contract samples. Caveat: those samples are survivor-biased (median "
             f"training change +0.34 log, 71% rose; all panel players 2013-2025: -0.10, 32% "
             f"rose), so medians lean upward. {rise:.0%} of these forecasts predict a rise, "
             f"median {q['q50'].median():+.2f} log. Training covered ages "
             f"{samples['age'].min():.0f}-{samples['age'].max():.0f} and values from "
             f"EUR {samples['value_now'].quantile(0.1) / 1e6:.2f}M (10th percentile); forecasts "
             f"outside that range, mostly older and cheaper players, are extrapolations.")
    data_versions = {"player-scores": f"valuations through {valuations['date'].max().date()}",
                     "samples": f"{len(samples)} rows", "panel_anchor": str(anchor.date())}
    out = write_bundle(paths["bundle"], players, history, forecasts, is_mock=False,
                       model_version=MODEL_VERSION, data_versions=data_versions, notes=notes)

    width = np.log(forecasts["p90_eur"] / forecasts["p10_eur"])
    print(f"median q50 {q['q50'].median():+.3f} log, {rise:.0%} predict a rise, "
          f"median band width {width.median():.2f} log")
    print(f"wrote {len(players)} players, {len(history)} history rows, "
          f"{len(forecasts)} forecasts to {out}")


if __name__ == "__main__":
    main()
