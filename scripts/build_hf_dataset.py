"""Stage the Hugging Face dataset: contracts, modelling samples, value histories, EDA figures.

Manual and augmented rows go to separate files and splits, so the 500 hand-collected rows stay
countable on their own. The value histories (Transfermarkt via Kaggle player-scores, CC0) let a
notebook rebuild the Chronos inputs without the team Drive. The card is hf/dataset_card.md.

Usage:
    python scripts/build_hf_dataset.py --data-dir <Dataset root> [--out data/processed/hf_dataset]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.data.load import read_table  # noqa: E402

MANUAL, AUGMENTED = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
# Columns of the sheet the dataset publishes; all are hand-collected or prefilled lookups
CONTRACT_COLUMNS = ["rank", "collector", "player_id", "player_name", "season", "anchor_date",
                    "club_that_season", "contract_start", "contract_end", "source_url", "notes"]


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def figures(samples: pd.DataFrame, out: Path) -> dict:
    """Four EDA figures and the numbers quoted in the card."""
    out.mkdir(parents=True, exist_ok=True)
    man, aug = samples[samples["origin"] == "manual"], samples[samples["origin"] == "span"]
    kw = dict(dpi=150, facecolor=SURFACE)

    # 1. Rows per season, manual vs augmented
    seasons = sorted(samples["season"].unique())
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(8, 3.6), **kw)
    for off, d, c, lab in [(-0.21, man, MANUAL, "Manual (hand-collected)"),
                           (0.21, aug, AUGMENTED, "Augmented (contract span)")]:
        counts = d["season"].value_counts().reindex(seasons, fill_value=0)
        ax.bar(x + off, counts, width=0.4, color=c, label=lab, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, [f"{s}/{str(s + 1)[2:]}" for s in seasons], rotation=45, ha="right")
    style(ax, "Samples per season", "Season of the 1 July anchor", "Rows")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "rows_per_season.png")
    plt.close(fig)

    # 2. Target: one-season log change in value
    bins = np.linspace(-1.5, 4, 45)
    fig, ax = plt.subplots(figsize=(8, 3.6), **kw)
    for d, c, lab in [(man, MANUAL, "Manual"), (aug, AUGMENTED, "Augmented")]:
        ax.hist(d["y_h1"].clip(-1.5, 4), bins=bins, histtype="step", linewidth=2, color=c, label=lab)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    style(ax, "Target: log change in value one season later  (0 = no change, +0.69 = doubled)",
          "log(value in 1 year / value at anchor); the end bins hold everything beyond", "Rows")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png")
    plt.close(fig)

    # 3. Change by age band, manual rows (the age curve the model leans on)
    # The oldest manual player is under 29, so the last band closes there
    bands = pd.cut(man["age"], [15, 19, 21, 23, 25, 27, 29],
                   labels=["16–19", "19–21", "21–23", "23–25", "25–27", "27–29"])
    g = man.groupby(bands, observed=True)["y_h1"].agg(["median", "size"])
    fig, ax = plt.subplots(figsize=(8, 3.6), **kw)
    ax.bar(range(len(g)), g["median"], color=MANUAL, width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, (med, n) in enumerate(zip(g["median"], g["size"])):
        # Counts sit above the bar, or above the zero line for a falling band
        ax.annotate(f"n={n}", (i, max(med, 0)), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(g)), g.index.astype(str))
    style(ax, "Median one-season log change by age, manual rows", "Age at anchor", "Median log change")
    fig.tight_layout()
    fig.savefig(out / "change_by_age.png")
    plt.close(fig)

    # 4. Contract length left at the anchor, manual vs augmented
    # Contracts end in June and anchors fall on 1 July, so whole years are the honest bins
    years = np.arange(0, 8)
    fig, ax = plt.subplots(figsize=(8, 3.6), **kw)
    for off, d, c, lab in [(-0.21, man, MANUAL, "Manual"), (0.21, aug, AUGMENTED, "Augmented")]:
        counts = d["years_left"].round().clip(0, 7).value_counts().reindex(years, fill_value=0)
        ax.bar(years + off, counts, width=0.4, color=c, label=lab, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(years, [*map(str, years[:-1]), "7+"])
    style(ax, "Contract years left at the anchor", "Years until contract end (rounded)", "Rows")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "years_left.png")
    plt.close(fig)

    def q(s, p):
        return float(s.quantile(p))

    return {
        "rows": {"manual": len(man), "augmented": len(aug)},
        "players": int(samples["player_id"].nunique()),
        "seasons": [int(seasons[0]), int(seasons[-1])],
        "manual_share_rose": float((man["y_h1"] > 0).mean()),
        "manual_median_y": float(man["y_h1"].median()),
        "manual_age_median": float(man["age"].median()),
        "manual_age_range": [float(man["age"].min()), float(man["age"].max())],
        "manual_value_median_eur": float(man["value_now"].median()),
        "manual_value_p10_p90_eur": [q(man["value_now"], 0.1), q(man["value_now"], 0.9)],
        "positions": man["position"].value_counts().to_dict(),
        "anchor_shifted": int(man["anchor_shifted"].sum()),
        "value_backfilled": int(man["value_backfilled"].sum()),
        "missing_share_panel_features": float(samples["minutes"].isna().mean()),
    }


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR"))
    ap.add_argument("--out", type=Path, default=paths["processed"] / "hf_dataset")
    args = ap.parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "contracts").mkdir(parents=True)
    (out / "samples").mkdir()

    contracts = pd.read_csv(paths["manual"] / "contracts.csv")[CONTRACT_COLUMNS]
    contracts.to_csv(out / "contracts" / "manual.csv", index=False)
    pd.read_csv(paths["synthetic"] / "contract_span_samples.csv").to_csv(
        out / "contracts" / "augmented.csv", index=False)

    samples = pd.read_parquet(paths["processed"] / "samples.parquet")
    samples.insert(3, "player_name", samples["player_id"].map(
        contracts.drop_duplicates("player_id").set_index("player_id")["player_name"]))
    # "span" is the pipeline's code for contract-span augmentation
    for origin, name in [("manual", "manual"), ("span", "augmented")]:
        samples[samples["origin"] == origin].to_parquet(out / "samples" / f"{name}.parquet",
                                                        index=False)

    kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else paths["raw_player_scores"]
    vals = read_table(kaggle, "player_valuations")
    hist = (vals[vals["player_id"].isin(samples["player_id"])]
            [["player_id", "date", "market_value_in_eur"]].sort_values(["player_id", "date"]))
    hist.to_parquet(out / "value_history.parquet", index=False)

    stats = figures(samples, out / "figures")
    stats["value_history_rows"] = len(hist)
    stats["value_history_last_date"] = str(hist["date"].max().date())
    (out / "eda_summary.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    card = ROOT / "hf" / "dataset_card.md"
    if card.exists():
        shutil.copy(card, out / "README.md")
    print(json.dumps(stats, indent=2, default=str))
    print(f"staged {out}")


if __name__ == "__main__":
    main()
