"""Slot 2: what the player's own value has already done.

Where a player sits relative to his own past is not the same question as where he sits
relative to his peers (slot 4). A player at half his peak and a player at his peak can
carry the same price tag and play for the same club.

Depth of history is a feature here, not a filter: a newcomer gets nulls in the columns
that need a past, and `years_of_history` says how much past there was. Requiring several
years of history instead would be a proxy for "age >= 25" and would delete the young
players whose value actually moves. See decision #6.

Days since the last revaluation is already on the panel as `value_age_days`, set by
`value_asof`, so it is not recomputed here.
"""
import numpy as np
import pandas as pd


def add_value_history(panel: pd.DataFrame, valuations: pd.DataFrame,
                      max_staleness_days: int) -> pd.DataFrame:
    """Attach each player's own valuation history as it stood at the anchor.

    Every column is built from valuations dated on or before the anchor. The running
    peak and count are accumulated along each player's own timeline and then read off
    with a backward `merge_asof`, so the state picked up is the state at the anchor
    rather than the state today.
    """
    v = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    v["date"] = v["date"].astype("datetime64[ns]")
    v = v.sort_values(["player_id", "date"])

    by_player = v.groupby("player_id")
    # cummax and cumcount run forward along each player's own history, so the value at
    # any row is that player's state as of that date and nothing later
    v["peak_value_eur"] = by_player["market_value_in_eur"].cummax()
    v["n_valuations_so_far"] = by_player.cumcount() + 1
    v["first_value_date"] = by_player["date"].transform("min")

    state = v[["player_id", "date", "peak_value_eur", "n_valuations_so_far",
               "first_value_date"]].sort_values("date")

    out = panel.sort_values("anchor_date").copy()
    out["anchor_date"] = out["anchor_date"].astype("datetime64[ns]")
    out = pd.merge_asof(out, state, left_on="anchor_date", right_on="date",
                        by="player_id", direction="backward")
    out = out.drop(columns=["date"])

    out["value_vs_peak"] = out["value_eur"] / out["peak_value_eur"]
    out["years_of_history"] = (
        (out["anchor_date"] - out["first_value_date"]).dt.days / 365.25)

    # The 12-month lookback is a second asof against the same history, taken from a date
    # one year earlier. Capping staleness the same way the anchor value is capped stops
    # a five-year-old price standing in for "a year ago".
    lookback = out[["player_id", "anchor_date"]].copy()
    lookback["lookback_date"] = lookback["anchor_date"] - pd.DateOffset(years=1)
    lookback = lookback.sort_values("lookback_date")
    prices = v[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    lookback = pd.merge_asof(lookback, prices, left_on="lookback_date", right_on="date",
                             by="player_id", direction="backward")
    too_old = (lookback["lookback_date"] - lookback["date"]).dt.days > max_staleness_days
    lookback.loc[too_old, "market_value_in_eur"] = np.nan
    lookback = lookback[["player_id", "anchor_date", "market_value_in_eur"]].rename(
        columns={"market_value_in_eur": "value_12m_ago_eur"})

    out = out.merge(lookback, on=["player_id", "anchor_date"], how="left")
    out["value_change_12m"] = np.log(out["value_eur"] / out["value_12m_ago_eur"])

    return out.drop(columns=["first_value_date"]).reset_index(drop=True)
