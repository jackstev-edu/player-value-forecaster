"""Context features: what surrounded the player, rather than what the player did.

Club strength is rebuilt per club-season from `player_valuations` because
`clubs.total_market_value` is 100% empty and the whole `clubs` table is an undated
scrape-time snapshot (decision #10). European participation comes from `games`, because
`team_competitions_seasons` contains no UEFA competitions at all (decision #11). Both
traps fail silently — see `docs/data_coverage.md`.
"""
import pandas as pd

from pvf.features.club_league import (
    UEFA_COMPETITIONS,
    club_domestic_league,
    club_european_participation,
)


def _top5_mean(values: pd.Series) -> float:
    """Star power, not squad depth: one world-class player moves this, a deep bench does not."""
    return values.nlargest(5).mean()


def club_strength(involvement: pd.DataFrame, valuations: pd.DataFrame,
                  panel_cfg: dict) -> pd.DataFrame:
    """One row per (club_id, season): what that squad was worth when the season ended.

    Season S is priced as of the anchor that opens season S+1, so a feature attached to
    an anchor can only ever have read valuations dated on or before that anchor. A
    player revalued in the summer window carries his pre-window price here.
    """
    month_day = panel_cfg["anchor_month_day"]

    squads = involvement[["player_id", "club_id", "season"]].drop_duplicates()
    as_of = squads["season"].add(1).astype(str) + "-" + month_day
    # A date-only Timestamp is datetime64[s] in pandas 2 but the parsed parquet is [ns],
    # and merge_asof refuses to join across resolutions
    squads["as_of"] = pd.to_datetime(as_of).astype("datetime64[ns]")

    v = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    v["date"] = v["date"].astype("datetime64[ns]")

    priced = pd.merge_asof(squads.sort_values("as_of"), v.sort_values("date"),
                           left_on="as_of", right_on="date", by="player_id",
                           direction="backward")
    fresh = (priced["as_of"] - priced["date"]).dt.days <= panel_cfg["max_value_staleness_days"]
    priced = priced[fresh & priced["market_value_in_eur"].notna()]

    return (priced.groupby(["club_id", "season"])["market_value_in_eur"]
            .agg(club_squad_size="size",
                 club_squad_value_eur="sum",
                 club_median_value_eur="median",
                 club_top5_value_eur=_top5_mean)
            .reset_index())


def league_strength(strength: pd.DataFrame, leagues: pd.DataFrame) -> pd.DataFrame:
    """One row per (competition_id, season): how rich that league was that season.

    Built from the club-season strengths rather than from players directly, so a league
    is the sum of its clubs and `league_median_club_value_eur` is the median *club*, not
    the median player. An inner join means a club that fielded nobody that season is
    absent rather than counted as worth zero.
    """
    joined = leagues.merge(strength, on=["club_id", "season"], how="inner")
    return (joined.groupby(["competition_id", "season"])["club_squad_value_eur"]
            .agg(league_club_count="size",
                 league_total_value_eur="sum",
                 league_median_club_value_eur="median")
            .reset_index())


def add_context_features(panel: pd.DataFrame, tables: dict[str, pd.DataFrame],
                         cfg: dict) -> pd.DataFrame:
    """Attach the club the player came from: its squad value and its European season.

    Both join on `(club_id, season)`, and the panel's `season` is already the season
    before the anchor, so the anti-leakage rule that picked the population carries
    through to these features unchanged.
    """
    strength = club_strength(tables["involvement"], tables["player_valuations"], cfg["panel"])
    out = panel.merge(strength, on=["club_id", "season"], how="left")

    leagues = club_domestic_league(tables["games"], cfg["panel"]["leagues"])
    out = out.merge(league_strength(strength, leagues),
                    on=["competition_id", "season"], how="left")
    # How big a fish in what pond: the same squad value means something different in the
    # Premier League than in the Eredivisie, and neither number says that on its own
    out["club_value_share_of_league"] = (out["club_squad_value_eur"]
                                         / out["league_total_value_eur"])

    out = out.merge(club_european_participation(tables["games"]),
                    on=["club_id", "season"], how="left")
    for flag in UEFA_COMPETITIONS:
        # eq(True) rather than fillna(False): a left merge leaves True or NaN, and
        # fillna on the resulting object column is deprecated
        out[flag] = out[flag].eq(True)

    return out
