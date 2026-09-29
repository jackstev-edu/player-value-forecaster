"""Walk-forward backtest on named players: forecast from a past date, compare with what happened.

The model for a forecast date trains only on samples whose one-season outcome was already
observed by that date, so the forecast never learns from the future it is judged on.
"""
import pandas as pd

TARGET = "y_h1"
ROLES = ["star", "prospect", "prime", "squad", "veteran"]


def known_by(samples: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """Samples whose outcome date (anchor + 1 season) is on or before `anchor`."""
    outcome_at = samples["anchor_date"] + pd.DateOffset(years=1)
    return samples[(outcome_at <= anchor) & samples[TARGET].notna()]


def pick_players(profiles: pd.DataFrame) -> dict[str, int]:
    """One player per role, chosen only from age and value at the start date, never the outcome.

    star: highest value. prospect: highest value under 21. prime: aged 25-28, value nearest
    EUR 20M. squad: aged 23-27, value nearest EUR 2M. veteran: highest value at 32 or over.
    Roles are filled in that order and a player fills at most one.
    """
    p = profiles.sort_values("player_id")
    age, value = p["age"], p["value_now"]
    rules = {
        "star": (p.index == p.index, -value),
        "prospect": (age < 21, -value),
        "prime": ((age >= 25) & (age < 29), (value - 20e6).abs()),
        "squad": ((age >= 23) & (age < 28), (value - 2e6).abs()),
        "veteran": (age >= 32, -value),
    }
    picks: dict[str, int] = {}
    for role in ROLES:
        eligible, order = rules[role]
        pool = p[eligible & ~p["player_id"].isin(picks.values())]
        # Stable sort keeps the lower player_id on ties, so the choice is reproducible
        best = order.loc[pool.index].sort_values(kind="stable").index[0]
        picks[role] = int(p.loc[best, "player_id"])
    return picks
