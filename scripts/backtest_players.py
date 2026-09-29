"""Backtest the core model on five named players over the last two seasons.

Standing on 1 July 2024 and again on 1 July 2025, the model forecasts each player's value one
season ahead; the forecast is then set against the Transfermarkt value that followed. Each
date gets its own model, trained only on samples whose outcome was known by then
(pvf.evaluation.backtest). The five players are picked by role from their age and value on
1 July 2024 alone. Writes docs/figures/backtest_players.{png,csv}.

Usage:
    python scripts/backtest_players.py --data-dir <Dataset root>
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.data.load import read_table  # noqa: E402
from pvf.evaluation.backtest import ROLES, known_by, pick_players  # noqa: E402
from pvf.evaluation.cv import _split_val  # noqa: E402
from pvf.export.real import universe  # noqa: E402
from pvf.features.samples import CORE_FEATURES, asof, core_features_at  # noqa: E402
from pvf.models.gbm import QuantileGBM  # noqa: E402

ANCHORS = [pd.Timestamp("2024-07-01"), pd.Timestamp("2025-07-01")]
ROLE_LABELS = {"star": "Star", "prospect": "Prospect (<21)", "prime": "Prime age, ~€20M",
               "squad": "Squad player, ~€2M", "veteran": "Veteran (32+)"}
# Reference palette: ink for what happened, slot-1 blue for the model
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def eur(v: float, _pos=None) -> str:
    return f"€{v / 1e6:g}M"


def forecast_at(anchor, samples, ids, valuations, players, cfg, until):
    """Walk-forward forecast for `ids` from `anchor`, with the value that actually followed."""
    m, seed = cfg["model"], cfg["project"]["seed"]
    train = known_by(samples, anchor)
    fit_part, val_part = _split_val(train, m["val_share"], seed)
    model = (QuantileGBM([1], m["quantiles"], m["gbm"], seed)
             .fit(fit_part, val_part, CORE_FEATURES).calibrate(val_part))
    at = pd.Series(anchor, index=range(len(ids)))
    core = core_features_at(pd.Series(ids), at, valuations, players)
    q = model.predict(core)
    target = min(anchor + pd.DateOffset(years=1), until)
    actual = asof(valuations, pd.Series(ids), pd.Series(target, index=at.index),
                  ["market_value_in_eur"])
    now = core["value_now"].to_numpy()
    out = pd.DataFrame({
        "player_id": ids, "anchor_date": anchor, "trained_on": len(train),
        "value_at_anchor": now,
        "p10_eur": now * np.exp(q["q10"].to_numpy()), "p50_eur": now * np.exp(q["q50"].to_numpy()),
        "p90_eur": now * np.exp(q["q90"].to_numpy()),
        "actual_eur": actual["market_value_in_eur"].to_numpy(), "actual_date": actual["date"].to_numpy(),
    })
    # An outcome only counts if a valuation landed after the forecast date
    out.loc[out["actual_date"] <= anchor, ["actual_eur", "actual_date"]] = np.nan
    return out


def plot(results: pd.DataFrame, valuations: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, len(ROLES), figsize=(17, 4.2))
    for ax, (pid, r) in zip(axes, results.groupby("player_id", sort=False)):
        h = valuations[(valuations["player_id"] == pid) & (valuations["date"] >= "2022-01-01")]
        ax.plot(h["date"], h["market_value_in_eur"], color=MUTED, lw=2, drawstyle="steps-post",
                zorder=2)
        for f in r.itertuples():
            # Forecast shown at the date it was aimed at, joined to the value it started from
            t = f.anchor_date + pd.DateOffset(years=1)
            ax.plot([f.anchor_date, t], [f.value_at_anchor, f.p50_eur], color=BLUE, lw=1.2, ls="--",
                    zorder=3)
            ax.vlines(t, f.p10_eur, f.p90_eur, color=BLUE, lw=6, alpha=0.25, zorder=3)
            ax.scatter([t], [f.p50_eur], s=40, color=BLUE, zorder=4, edgecolor="white", linewidth=1.5)
            if pd.notna(f.actual_eur):
                ax.scatter([f.actual_date], [f.actual_eur], s=46, facecolor="white", edgecolor=INK,
                           linewidth=1.8, zorder=5)
            ax.axvline(f.anchor_date, color=GRID, lw=1, zorder=1)
        ax.set_yscale("log")
        # 1-2-5 steps, so a panel whose values never cross a power of ten still gets labels
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(eur))
        ax.yaxis.set_minor_formatter(NullFormatter())
        hits = r["in_band"].dropna()
        ax.set_title(r["name"].iloc[0], loc="left", fontsize=10, color=INK, pad=16)
        ax.text(0, 1.02, f"{ROLE_LABELS[r['role'].iloc[0]]} · inside range {int(hits.sum())} of {len(hits)}",
                transform=ax.transAxes, color=MUTED, fontsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8, which="major")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    handles = [
        plt.Line2D([], [], color=MUTED, lw=2, label="Transfermarkt value"),
        plt.Line2D([], [], color=BLUE, marker="o", ls="--", lw=1.2, label="Model forecast (median)"),
        plt.Line2D([], [], color=BLUE, lw=6, alpha=0.25, label="10-90% range"),
        plt.Line2D([], [], color=INK, marker="o", mfc="white", ls="", label="Actual value one season later"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.005, 1.0))
    fig.suptitle("One-season forecasts made on 1 Jul 2024 and 1 Jul 2025, against what happened",
                 x=0.005, y=1.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR"),
                    help="Dataset root holding player-scores/ (defaults to FV_DATA_DIR)")
    args = ap.parse_args()
    cfg = load_config()
    paths = cfg["paths"]
    until = pd.Timestamp(cfg["split"]["values_available_until"])

    kaggle = Path(args.data_dir) / "player-scores" if args.data_dir else paths["raw_player_scores"]
    valuations = read_table(kaggle, "player_valuations")
    players = read_table(kaggle, "players")
    samples = pd.read_parquet(paths["processed"] / "samples.parquet")
    panel = pd.read_parquet(paths["processed"] / "panel.parquet")

    # Candidates: league players on the first date, valued then and valued again after the
    # second date, so both forecasts can be checked
    univ = universe(panel, ANCHORS[0])
    core = core_features_at(univ["player_id"], pd.Series(ANCHORS[0], index=univ.index),
                            valuations, players)
    last_valued = valuations.groupby("player_id")["date"].max()
    ok = core["value_now"].notna() & (univ["player_id"].map(last_valued) > ANCHORS[1])
    picks = pick_players(pd.DataFrame({"player_id": univ.loc[ok, "player_id"],
                                       "age": core.loc[ok, "age"], "value_now": core.loc[ok, "value_now"]}))
    ids = list(picks.values())

    results = pd.concat([forecast_at(a, samples, ids, valuations, players, cfg, until)
                         for a in ANCHORS], ignore_index=True)
    names = players.drop_duplicates("player_id").set_index("player_id")["name"]
    results.insert(1, "name", results["player_id"].map(names))
    results.insert(2, "role", results["player_id"].map({v: k for k, v in picks.items()}))
    results["log_error"] = np.log(results["p50_eur"] / results["actual_eur"])
    results["in_band"] = ((results["actual_eur"] >= results["p10_eur"])
                          & (results["actual_eur"] <= results["p90_eur"])).where(results["actual_eur"].notna())
    order = {p: i for i, p in enumerate(ids)}
    results = (results.assign(_o=results["player_id"].map(order))
               .sort_values(["_o", "anchor_date"]).drop(columns="_o").reset_index(drop=True))

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "backtest_players.csv", index=False, float_format="%.4g")
    plot(results, valuations, out / "backtest_players.png")

    show = results.assign(**{c: results[c] / 1e6 for c in
                             ["value_at_anchor", "p10_eur", "p50_eur", "p90_eur", "actual_eur"]})
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(show[["name", "role", "anchor_date", "trained_on", "value_at_anchor", "p10_eur",
                    "p50_eur", "p90_eur", "actual_eur", "log_error", "in_band"]].to_string(index=False))
    scored = results.dropna(subset=["actual_eur"])
    print(f"\nlog MAE {scored['log_error'].abs().mean():.3f} over {len(scored)} forecasts, "
          f"{int(scored['in_band'].sum())} of {len(scored)} inside the 10-90% band")
    print(f"wrote {out / 'backtest_players.png'} and backtest_players.csv")


if __name__ == "__main__":
    main()
