"""Build the player-season panel that supplies the gathered features (PANEL_FEATURES).

Reads player-scores and football-datasets under the Dataset root and writes
data/processed/panel.parquet. Run this before scripts/build_dataset.py.

Usage:
    python scripts/build_panel.py --data-dir <Dataset root> [--out data/processed/panel.parquet]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.data.load import read_table  # noqa: E402
from pvf.features.build_panel import build_panel  # noqa: E402

TABLES = {
    "player-scores": ["games", "game_lineups", "appearances", "players", "player_valuations"],
    "football-datasets": ["player_profiles", "player_injuries", "transfer_history"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=os.environ.get("FV_DATA_DIR"),
                    help="Dataset root holding player-scores/ and football-datasets/")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.data_dir is None:
        ap.error("--data-dir or FV_DATA_DIR is required")
    cfg = load_config()

    tables = {t: read_table(Path(args.data_dir) / folder, t)
              for folder, names in TABLES.items() for t in names}
    panel = build_panel(tables, cfg)

    out = args.out or cfg["paths"]["processed"] / "panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel)} player-seasons x {panel.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
