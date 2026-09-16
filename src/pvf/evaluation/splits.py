"""Time-based splits. Never random: a training target must resolve before the test anchor."""
import pandas as pd


def time_split(panel: pd.DataFrame, horizon: int, train_last_season: int, val_season: int,
               test_season: int, date_col: str = "anchor_date") -> dict[str, pd.DataFrame]:
    """Return train/val/test frames for one horizon with no target overlap."""
    season = panel[date_col].dt.year
    test_start = pd.Timestamp(year=test_season, month=panel[date_col].dt.month.iloc[0],
                              day=panel[date_col].dt.day.iloc[0])
    target_date = panel[date_col] + pd.DateOffset(years=horizon)
    # Purge rows whose target resolves after the test anchor
    resolved = target_date <= test_start
    has_y = panel[f"y_h{horizon}"].notna()

    train = panel[(season <= train_last_season) & resolved & has_y]
    val = panel[(season == val_season) & resolved & has_y]
    test = panel[(season == test_season) & has_y]
    return {"train": train, "val": val, "test": test}


def rolling_origins(test_seasons: list[int], horizon: int, values_until: str) -> list[int]:
    """Keep only test seasons whose horizon target is observable in the data."""
    last_year = pd.Timestamp(values_until).year
    # Anchor year plus horizon must not pass the last valuation year
    return [s for s in test_seasons if s + horizon <= last_year]
