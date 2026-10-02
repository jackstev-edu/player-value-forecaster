"""Build the modelling dataset: 500 hand-collected contracts, augmented to 1000 samples.

Steps: export the filled Drive sheet to data/manual/contracts.csv (optional), check it,
build one sample per usable contract, add contract-span samples, join the gathered panel
features (run scripts/build_panel.py first), cap at the configured total.

Usage:
    python scripts/build_dataset.py --data-dir <Dataset root> [--sheet "Contract Data Collection.xlsx"]
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.data.augment import (backfill_first_value, cap_augmented, shift_unanchored,  # noqa: E402
                              span_contracts)
from pvf.data.load import read_table  # noqa: E402
from pvf.data.manual import CONTRACT_COLUMNS, check_contracts, read_contracts, usable_contracts  # noqa: E402
from pvf.features.samples import add_panel_features, build_samples, drop_unmatched_augmented  # noqa: E402


def export_sheet(sheet: Path, out: Path) -> None:
    """Keep only what we collected plus its keys; drop the Kaggle-derived hint columns."""
    df = pd.read_excel(sheet, sheet_name="contracts")
    out.parent.mkdir(parents=True, exist_ok=True)
    df[CONTRACT_COLUMNS].to_csv(out, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    print(f"exported {len(df)} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR"),
                    help="Dataset root holding player-scores/ (defaults to FV_DATA_DIR)")
    ap.add_argument("--sheet", type=Path, help="Filled contract workbook to export first")
    ap.add_argument("--panel", type=Path, help="Panel parquet (defaults to data/processed/panel.parquet)")
    args = ap.parse_args()
    cfg = load_config()
    paths, seed = cfg["paths"], cfg["project"]["seed"]
    horizon = cfg["target"]["horizons"][0]
    until = cfg["split"]["values_available_until"]
    contracts_csv = paths["manual"] / "contracts.csv"

    if args.sheet:
        export_sheet(args.sheet, contracts_csv)
    contracts = read_contracts(contracts_csv)

    issues = check_contracts(contracts)
    issues.to_csv(paths["manual"] / "contract_issues.csv", index=False)
    print("contract issues:", issues.groupby(["severity", "rule"]).size().to_dict())
    manual = usable_contracts(contracts).assign(origin="manual")
    print(f"usable contracts: {len(manual)} of {len(contracts)}")

    kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else paths["raw_player_scores"]
    valuations = read_table(kaggle, "player_valuations")
    players = read_table(kaggle, "players")

    span = span_contracts(manual, valuations, horizon=horizon, values_until=until)
    samples = build_samples(pd.concat([manual, span], ignore_index=True), valuations, players,
                            horizon=horizon, values_until=until,
                            max_staleness_days=cfg["samples"]["max_value_staleness_days"])
    target = f"y_h{horizon}"
    # Manual rows with no value at the anchor (decision #44): move to a later season of the
    # same contract, else take the nearest Transfermarkt value; both are flagged
    is_manual = samples["origin"] == "manual"
    manual_s, span_s = shift_unanchored(samples[is_manual], samples[~is_manual])
    samples = backfill_first_value(pd.concat([manual_s, span_s], ignore_index=True), valuations,
                                   horizon=horizon, values_until=until)
    samples[["anchor_shifted", "value_backfilled"]] = (
        samples[["anchor_shifted", "value_backfilled"]].fillna(False).astype(bool))
    print("manual rows moved to a later season:", int(samples["anchor_shifted"].sum()),
          "| value backfilled:", int(samples["value_backfilled"].sum()))
    samples = samples.dropna(subset=["value_now", target])
    print("with a target:", samples["origin"].value_counts().to_dict())

    # Gathered features for each sample's own season (decision #43). Augmented rows
    # without a panel row are dropped here, before the cap refills to the target total.
    panel = pd.read_parquet(args.panel or paths["processed"] / "panel.parquet")
    samples = add_panel_features(samples, panel)
    print("with a panel row:", samples.groupby("origin")["in_panel"].sum().to_dict())
    if cfg["samples"]["require_panel_for_augmented"]:
        samples = drop_unmatched_augmented(samples)

    samples = cap_augmented(samples, cfg["augment"]["target_total"], seed)
    print("final:", samples["origin"].value_counts().to_dict(), "total", len(samples))

    paths["processed"].mkdir(parents=True, exist_ok=True)
    samples.to_parquet(paths["processed"] / "samples.parquet", index=False)
    # Augmented rows as contract keys only, so the synthetic set is auditable on its own
    keys = ["rank", "player_id", "anchor_date", "origin"]
    synthetic = samples.loc[samples["origin"] != "manual", keys].merge(
        manual[["rank", "contract_start", "contract_end"]], on="rank")
    synthetic.to_csv(paths["synthetic"] / "contract_span_samples.csv", index=False,
                     date_format="%Y-%m-%d")
    print(f"wrote {paths['processed'] / 'samples.parquet'} and {len(synthetic)} synthetic rows")


if __name__ == "__main__":
    main()
