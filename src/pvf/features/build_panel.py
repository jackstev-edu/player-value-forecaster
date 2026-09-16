"""Build the player-season panel: one row per player per anchor date.

Each row holds features known before the anchor and log-ratio targets
for every horizon in the config.
"""
import numpy as np
import pandas as pd


def anchor_dates(first_season: int, last_season: int, month_day: str) -> pd.DatetimeIndex:
    """One anchor per season, e.g. 2012-07-01 through 2025-07-01."""
    return pd.to_datetime([f"{y}-{month_day}" for y in range(first_season, last_season + 1)])


def value_asof(valuations: pd.DataFrame, anchors: pd.DataFrame,
               date_col: str = "anchor_date", max_staleness_days: int | None = None) -> pd.DataFrame:
    """Attach the latest market value on or before each anchor date."""
    v = valuations[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    a = anchors.sort_values(date_col)
    # Backward asof join means only past values are visible
    out = pd.merge_asof(a, v, left_on=date_col, right_on="date", by="player_id",
                        direction="backward")
    out = out.rename(columns={"date": "value_date", "market_value_in_eur": "value_eur"})
    out["value_age_days"] = (out[date_col] - out["value_date"]).dt.days
    if max_staleness_days is not None:
        # Stale values are not a real snapshot of the player at the anchor
        out.loc[out["value_age_days"] > max_staleness_days, "value_eur"] = np.nan
    return out


def add_targets(panel: pd.DataFrame, valuations: pd.DataFrame, horizons: list[int],
                max_staleness_days: int) -> pd.DataFrame:
    """Add log(value at anchor+h / value at anchor) for each horizon."""
    for h in horizons:
        future = panel[["player_id", "anchor_date"]].copy()
        future["target_date"] = future["anchor_date"] + pd.DateOffset(years=h)
        fut = value_asof(valuations, future, date_col="target_date",
                         max_staleness_days=max_staleness_days)
        fut = fut[["player_id", "anchor_date", "value_eur"]].rename(
            columns={"value_eur": f"value_h{h}"})
        panel = panel.merge(fut, on=["player_id", "anchor_date"], how="left")
        # NaN target marks a player who vanished; keep row for survivorship EDA
        panel[f"y_h{h}"] = np.log(panel[f"value_h{h}"] / panel["value_eur"])
    return panel


def build_panel(tables: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Assemble the full panel from cleaned tables."""
    p = cfg["panel"]
    last_value_year = pd.Timestamp(cfg["split"]["values_available_until"]).year
    anchors = anchor_dates(p["min_anchor_season"], last_value_year, p["anchor_month_day"])

    # Cross join every valued player with every anchor
    players = tables["player_valuations"][["player_id"]].drop_duplicates()
    grid = players.merge(pd.DataFrame({"anchor_date": anchors}), how="cross")

    panel = value_asof(tables["player_valuations"], grid,
                       max_staleness_days=p["max_value_staleness_days"])
    panel = panel.dropna(subset=["value_eur"])
    panel = add_targets(panel, tables["player_valuations"], cfg["target"]["horizons"],
                        p["max_value_staleness_days"])

    # >>> 1. PLAYER FEATURES HERE: age at anchor, position, foot, height, EU passport <<<

    # >>> 2. VALUE HISTORY FEATURES HERE: 12 month change, peak so far, days since update <<<

    # >>> 3. LAST SEASON PERFORMANCE HERE: minutes, start share, G+A per 90, cards <<<

    # >>> 4. CONTEXT FEATURES HERE: league strength, club strength, Euro comps, positional rank <<<

    # >>> 5. HEALTH AND MOVES HERE: days injured, transferred last window, fee <<<

    # >>> 6. LEAGUE FILTER HERE: apply cfg["panel"]["leagues"] once decided <<<

    return panel.reset_index(drop=True)
