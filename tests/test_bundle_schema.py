"""The app must be able to read whatever bundle is committed in app/predictions."""
from pathlib import Path

import pandas as pd
import pytest

from pvf.export.schema import validate

BUNDLE = Path(__file__).resolve().parents[1] / "app" / "predictions"


@pytest.mark.skipif(not (BUNDLE / "players.parquet").exists(), reason="no bundle yet")
def test_committed_bundle_is_valid():
    validate(pd.read_parquet(BUNDLE / "players.parquet"),
             pd.read_parquet(BUNDLE / "history.parquet"),
             pd.read_parquet(BUNDLE / "forecasts.parquet"))
