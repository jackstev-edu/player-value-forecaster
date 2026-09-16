"""Load configs/config.yaml and resolve paths against the repo root."""
from pathlib import Path

import yaml

# Repo root sits three levels above this file
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str = ROOT / "configs" / "config.yaml") -> dict:
    """Return the config dict with every paths entry made absolute."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Absolute paths so scripts work from any working directory
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg
