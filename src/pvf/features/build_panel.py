"""Build the player-season panel: one row per player per anchor date.

Each row holds features known before the anchor and log-ratio targets
for every horizon in the config.
"""
import numpy as np
import pandas as pd

from pvf.features.club_league import (
    club_domestic_league,
    player_club_season,
    players_at_anchor,
)
from pvf.features.context import add_context_features, positional_ranks
from pvf.features.history import add_value_history

# Allowlist, not a blocklist: anything not named here stays out of the panel, so a new
# snapshot column in players/ cannot leak in by being forgotten. See features/leakage.py.
PLAYER_FEATURE_COLUMNS = [
    "player_id", "date_of_birth", "position", "sub_position", "foot",
    "height_in_cm", "country_of_citizenship",
]


def anchor_dates(first_season: int, last_season: int, month_day: str) -> pd.DatetimeIndex:
    """One anchor per season, e.g. 2013-07-01 through 2025-07-01."""
    return pd.to_datetime([f"{y}-{month_day}" for y in range(first_season, last_season + 1)])


def anchor_population(tables: dict[str, pd.DataFrame], cfg: dict,
                      last_anchor_season: int) -> pd.DataFrame:
    """Who belongs in the panel at each anchor, and which league they came from.

    Membership is decided by the season that has already finished, never by the club a
    player is registered to on the anchor date, because transfers run through the summer
    and the incoming club would leak the future. Only the primary club is kept, so a
    mid-season move yields one row rather than two.
    """
    p = cfg["panel"]
    leagues = club_domestic_league(tables["games"], p["leagues"])
    involvement = player_club_season(tables["game_lineups"], tables["appearances"],
                                     tables["games"])

    frames = []
    for year in range(p["min_anchor_season"], last_anchor_season + 1):
        pop = players_at_anchor(involvement, leagues, anchor_year=year)
        pop = pop[pop["is_primary"]].copy()
        pop["anchor_date"] = pd.Timestamp(f"{year}-{p['anchor_month_day']}")
        frames.append(pop)

    out = pd.concat(frames, ignore_index=True)
    # A date-only Timestamp is datetime64[s] in pandas 2, but the parsed parquet is [ns]
    out["anchor_date"] = out["anchor_date"].astype("datetime64[ns]")
    return out


def value_asof(valuations: pd.DataFrame, anchors: pd.DataFrame,
               date_col: str = "anchor_date", max_staleness_days: int | None = None) -> pd.DataFrame:
    """Attach the latest market value on or before each anchor date."""
    v = valuations[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    a = anchors.sort_values(date_col)
    # merge_asof refuses to join datetime64[s] against datetime64[ns], and pandas 2 hands
    # out either depending on how the column was built, so pin both sides first
    v["date"] = v["date"].astype("datetime64[ns]")
    a[date_col] = a[date_col].astype("datetime64[ns]")
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


def add_player_features(panel: pd.DataFrame, players: pd.DataFrame,
                        profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the fixed facts about a player, plus age at the anchor.

    Only columns in PLAYER_FEATURE_COLUMNS cross over. Everything else in `players` is a
    scrape-time snapshot that would leak the future into historical anchors.
    """
    available = [c for c in PLAYER_FEATURE_COLUMNS if c in players.columns]
    out = panel.merge(players[available], on="player_id", how="left")

    out["age"] = (out["anchor_date"] - out["date_of_birth"]).dt.days / 365.25
    if "height_in_cm" in out.columns:
        # 4.7% of heights are 0, which means unknown rather than a 0 cm player
        out["height_in_cm"] = out["height_in_cm"].replace(0, np.nan)

    if profiles is not None and "is_eu" in profiles.columns:
        # Citizenship rarely changes, so the snapshot flag is safe as a fixed fact
        eu = profiles[["player_id", "is_eu"]].drop_duplicates("player_id")
        out = out.merge(eu, on="player_id", how="left")

    return out


def build_panel(tables: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Assemble the full panel from cleaned tables."""
    p = cfg["panel"]
    last_value_year = pd.Timestamp(cfg["split"]["values_available_until"]).year

    # Population comes from who actually played, not from a cross join of every player
    # against every anchor, so the league filter and the anti-leakage rule are one step
    panel = anchor_population(tables, cfg, last_anchor_season=last_value_year)

    panel = value_asof(tables["player_valuations"], panel,
                       max_staleness_days=p["max_value_staleness_days"])
    panel = panel.dropna(subset=["value_eur"])
    panel = add_targets(panel, tables["player_valuations"], cfg["target"]["horizons"],
                        p["max_value_staleness_days"])

    panel = add_player_features(panel, tables["players"], tables.get("player_profiles"))

    # Slot 2: the player's own past. Days since update is already here as value_age_days
    panel = add_value_history(panel, tables["player_valuations"],
                              p["max_value_staleness_days"])

    # >>> 3. LAST SEASON PERFORMANCE HERE: minutes, start share, G+A per 90 <<<

    # Slot 4: club strength, league strength, European participation, positional rank
    involvement = player_club_season(tables["game_lineups"], tables["appearances"],
                                     tables["games"])
    panel = add_context_features(panel, {**tables, "involvement": involvement}, cfg)
    # Ranks come last: they compare panel members against each other, so they need the
    # population and the anchor values already settled
    panel = positional_ranks(panel)

    # >>> 5. HEALTH AND MOVES HERE: days injured, transferred last window, fee <<<

    return panel.reset_index(drop=True)
