"""Context features: what surrounded the player, rather than what the player did.

Club strength is rebuilt per club-season from `player_valuations` because
`clubs.total_market_value` is 100% empty and the whole `clubs` table is an undated
scrape-time snapshot (decision #10). European participation comes from `games`, because
`team_competitions_seasons` contains no UEFA competitions at all (decision #11). Both
traps fail silently — see `docs/data_coverage.md`.
"""
import numpy as np
import pandas as pd

from pvf.features.club_league import (
    UEFA_COMPETITIONS,
    club_domestic_league,
    club_european_participation,
)

# `players.position` uses this string rather than a null for players whose position was
# never recorded. It is a sentinel, so it must not be ranked as if it were a position.
UNKNOWN_POSITIONS = {"Missing"}


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


def positional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank each player against the same-position players around him, at that anchor.

    Two scopes: his own club, and every club in his league. A centre-back worth €8M is
    a different proposition depending on whether he is the best defender at his club or
    the fourth, and neither his own value nor his club's strength says which.

    This is the one context feature that cannot be a `(club_id, season)` join, so it is
    computed here rather than in `add_context_features`. Grouping on `anchor_date` is
    what keeps it honest: the comparison is cross-sectional, between players at the same
    moment, and `value_eur` is already the as-of-anchor value, so nothing looks forward.

    The peers are the panel's own members, so a club-mate without a fresh enough
    valuation is not counted. `position_peers_*` carries the group size for that reason —
    rank 3 means something different out of 4 than out of 20.
    """
    out = panel.copy()
    scopes = {"at_club": "club_id", "in_league": "competition_id"}
    for suffix, scope_col in scopes.items():
        grouped = out.groupby(["anchor_date", scope_col, "position"])["value_eur"]
        # method="min" so tied players share the better rank and consume both places
        out[f"position_rank_{suffix}"] = grouped.rank(ascending=False, method="min")
        out[f"position_peers_{suffix}"] = grouped.transform("size").astype(float)
        # Ascending here so 1.0 is the most valuable, which reads the right way round
        out[f"position_pctile_{suffix}"] = grouped.rank(ascending=True, pct=True)

    # `players.position` marks 586 unknowns with the string "Missing" rather than a null,
    # so they group together and come out ranked 1 of 1 — which a model would read as
    # "best in his position" when it means the opposite of knowing anything
    unknown = out["position"].isin(UNKNOWN_POSITIONS) | out["position"].isna()
    rank_columns = [f"position_{stat}_{suffix}"
                    for suffix in scopes for stat in ("rank", "peers", "pctile")]
    out.loc[unknown, rank_columns] = np.nan
    return out


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
