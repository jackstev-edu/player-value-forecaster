"""Write the prediction bundle consumed by app/app.py."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pvf.export.schema import FORECASTS_COLUMNS, HISTORY_COLUMNS, PLAYERS_COLUMNS, validate


def write_bundle(out_dir: Path, players: pd.DataFrame, history: pd.DataFrame,
                 forecasts: pd.DataFrame, *, is_mock: bool, model_version: str,
                 data_versions: dict, notes: str = "") -> Path:
    """Validate, then write three parquet tables plus a manifest."""
    validate(players, history, forecasts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Column order fixed so diffs between bundles stay readable
    players[PLAYERS_COLUMNS].to_parquet(out_dir / "players.parquet", index=False)
    history[HISTORY_COLUMNS].to_parquet(out_dir / "history.parquet", index=False)
    forecasts[FORECASTS_COLUMNS].to_parquet(out_dir / "forecasts.parquet", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_mock": is_mock,
        "model_version": model_version,
        "data_versions": data_versions,
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir
