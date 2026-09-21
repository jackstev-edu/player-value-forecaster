"""Slot 3: what the player did on the field in the season that just ended.

Minutes, squad games and games played are already on the panel — `player_club_season`
computes them when it picks the population (decision #14). What is missing is the shape
of those minutes: whether they came as a starter, and what the player produced inside
them. Cards are deliberately absent (decision #12).

Everything is keyed on `(player_id, club_id, season)`, so a player who moved at the
winter break is measured at each club separately and the panel row for his primary club
carries only that club's half of the season.
"""
import numpy as np
import pandas as pd

_KEYS = ["player_id", "club_id", "season"]

# A named starter in `game_lineups`; the other value is "substitutes", which covers
# unused substitutes as well as those brought on.
_STARTER = "starting_lineup"


def player_season_stats(game_lineups: pd.DataFrame, appearances: pd.DataFrame,
                        games: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, club_id, season) with starts, goals and assists.

    Starts come from `game_lineups` and scoring from `appearances`, because only the
    first says who began the match and only the second says what happened in it.

    Season is taken from `games`, never from a date, so 2019-20 is season 2019 on both
    sides of the join.
    """
    season = games[["game_id", "season"]]

    lineups = game_lineups.merge(season, on="game_id")
    lineups["started"] = lineups["type"].eq(_STARTER)
    starts = lineups.groupby(_KEYS, as_index=False).agg(starts=("started", "sum"))

    scoring = (appearances.rename(columns={"player_club_id": "club_id"})
               .merge(season, on="game_id")
               .groupby(_KEYS, as_index=False)
               .agg(goals=("goals", "sum"), assists=("assists", "sum")))

    out = starts.merge(scoring, on=_KEYS, how="outer")
    # No appearance row means the player did not take the field, which is a measured
    # zero rather than a gap.
    out[["goals", "assists"]] = out[["goals", "assists"]].fillna(0)
    out["starts"] = out["starts"].fillna(0)

    # `game_lineups` begins on 2013-07-02, so season 2012 has no starting XI to read
    # (decision #15). Zero there would tell the model nobody started that season, so
    # the whole season goes null instead. Within a covered season a missing lineup row
    # is a real "was not named", and keeps its zero.
    covered = lineups["season"].unique()
    out.loc[~out["season"].isin(covered), "starts"] = np.nan

    return out.sort_values(_KEYS).reset_index(drop=True)


def add_performance_features(panel: pd.DataFrame,
                             tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach last season's playing record to each panel row.

    The panel's `season` is already the season before the anchor, so the join carries
    the anti-leakage rule through unchanged: nothing here can see a game played after
    the anchor.
    """
    stats = player_season_stats(tables["game_lineups"], tables["appearances"],
                               tables["games"])
    out = panel.merge(stats, on=_KEYS, how="left")
    out[["goals", "assists"]] = out[["goals", "assists"]].fillna(0)

    # Rates are undefined rather than infinite when the denominator is zero: a player
    # who was never named has no start share, and one who never played has no per-90.
    out["start_share"] = out["starts"] / out["squad_games"].replace(0, np.nan)
    out["goals_assists_per90"] = (
        (out["goals"] + out["assists"]) / (out["minutes"].replace(0, np.nan) / 90))

    return out
