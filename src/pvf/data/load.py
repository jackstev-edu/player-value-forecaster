"""Read raw Kaggle parquet tables with dates parsed and unknown sentinels cleaned."""
from pathlib import Path

import numpy as np
import pandas as pd

# Text date columns per table, parsed on load
DATE_COLUMNS = {
    "player_valuations": ["date"],
    "appearances": ["date"],
    "games": ["date"],
    "transfers": ["transfer_date"],
    # player-scores calls this table `transfers`; football-datasets calls its own
    # `transfer_history`. Both need parsing, and a missed one fails quietly: ISO
    # date strings still compare correctly, so only a .dt call gives the gap away.
    "transfer_history": ["transfer_date"],
    "players": ["date_of_birth"],
    "player_injuries": ["from_date", "end_date"],
    "player_market_value": ["date_unix"],
    "player_profiles": ["date_of_birth"],
}

# Zeros that really mean unknown, per the dataset overview
ZERO_MEANS_UNKNOWN = {
    "player_profiles": ["height"],
    "player_market_value": ["value"],
}


def read_table(folder: Path, table: str) -> pd.DataFrame:
    """Load one parquet table and apply the cleaning rules above."""
    df = pd.read_parquet(Path(folder) / f"{table}.parquet")
    for col in DATE_COLUMNS.get(table, []):
        # errors=coerce turns malformed strings into NaT instead of crashing
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ZERO_MEANS_UNKNOWN.get(table, []):
        df[col] = df[col].replace(0, np.nan)
    if table == "team_competitions_seasons":
        # About 73% exact duplicates in this release
        df = df.drop_duplicates()
    return df
