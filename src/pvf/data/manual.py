"""Read and check the hand-collected contract sheet (Task 19, data/manual/contracts.csv)."""
from pathlib import Path

import pandas as pd

CONTRACT_COLUMNS = ["rank", "collector", "player_id", "player_name", "season", "anchor_date",
                    "club_that_season", "contract_start", "contract_end", "source_url", "notes"]
DATE_COLUMNS = ["anchor_date", "contract_start", "contract_end"]


def read_contracts(path: Path | str) -> pd.DataFrame:
    """Load the sheet export with its three date columns parsed."""
    df = pd.read_csv(path)
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def check_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per problem: rank, rule, severity. Errors make a row unusable, warnings do not."""
    start, end, anchor = df["contract_start"], df["contract_end"], df["anchor_date"]
    dated = start.notna() & end.notna()
    rules = [
        ("missing_dates", "error", ~dated),
        ("end_not_after_start", "error", dated & (end <= start)),
        ("expired_before_anchor", "error", dated & (end > start) & (end <= anchor)),
        # A summer signing after 1 July is fine; a start a full year later is another season
        ("starts_after_season", "error", dated & (start >= anchor + pd.DateOffset(years=1))),
        ("duplicate_player", "error", df["player_id"].duplicated(keep=False)),
        ("source_not_url", "warning", ~df["source_url"].astype(str).str.startswith("http")),
    ]
    issues = [pd.DataFrame({"rank": df.loc[mask, "rank"], "rule": rule, "severity": severity})
              for rule, severity, mask in rules]
    return pd.concat(issues, ignore_index=True)


def usable_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no error-level issue."""
    issues = check_contracts(df)
    bad = issues.loc[issues["severity"] == "error", "rank"]
    return df[~df["rank"].isin(bad)].reset_index(drop=True)
