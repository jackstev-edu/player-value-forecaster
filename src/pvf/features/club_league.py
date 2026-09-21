"""Work out which league a club played in each season, and who was in its squad.

Replaces `player_valuations.player_club_domestic_competition_id`, whose club-to-league
mapping decays from 2024 onwards: the players carrying a top-9 league id drop from
12,495 in 2023 to 6,573 in 2024 while valuation volume holds steady. Deriving league
membership from games actually played avoids that. See `docs/data_coverage.md`.

Season is taken from `games.season`, never from a date, so the 2019-20 season is 2019
throughout.
"""
import pandas as pd

# UEFA competition ids in player-scores, including their qualifying rounds
UEFA_COMPETITIONS = {
    "played_ucl": ["CL", "CLQ"],
    "played_uel": ["EL", "ELQ"],
    "played_uecl": ["UCOL", "ECLQ"],
}

_KEYS = ["player_id", "club_id", "season"]


def _club_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per club per game, folding home and away ids into a single column."""
    cols = ["game_id", "competition_id", "season"]
    home = games[cols + ["home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = games[cols + ["away_club_id"]].rename(columns={"away_club_id": "club_id"})
    return pd.concat([home, away], ignore_index=True)


def club_domestic_league(games: pd.DataFrame, league_ids: list[str]) -> pd.DataFrame:
    """One row per (club_id, season) giving the domestic league that club played in.

    Filtering to `league_ids` first is what keeps a club that played both its league
    and the Champions League from ending up with two rows for one season.
    """
    cg = _club_games(games)
    cg = cg[cg["competition_id"].isin(league_ids)]
    return (cg[["club_id", "season", "competition_id"]]
            .drop_duplicates()
            .sort_values(["club_id", "season"])
            .reset_index(drop=True))


def club_european_participation(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (club_id, season) with a flag per UEFA competition.

    Qualifying rounds count as participation, since playing them is itself a signal
    about the club.
    """
    cg = _club_games(games)
    out = cg[["club_id", "season"]].drop_duplicates()
    for flag, competitions in UEFA_COMPETITIONS.items():
        played = cg.loc[cg["competition_id"].isin(competitions), ["club_id", "season"]]
        played = played.drop_duplicates().assign(**{flag: True})
        out = out.merge(played, on=["club_id", "season"], how="left")
        # notna() rather than fillna(False): the merge leaves True or NaN, and
        # fillna on an object column is deprecated
        out[flag] = out[flag].notna()
    return out.sort_values(["club_id", "season"]).reset_index(drop=True)


def player_club_season(game_lineups: pd.DataFrame, appearances: pd.DataFrame,
                       games: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, club_id, season) describing that player's involvement.

    Squad membership comes from `game_lineups`, which names unused substitutes and so
    keeps players who were fit enough to be picked but never took the field. That table
    only starts in July 2013, so `appearances` is unioned in: for season 2012 it is the
    only source, and taking the union means the fallback needs no special case.

    A mid-season transfer produces one row per club. `is_primary` marks the club the
    player was named in most often, so callers that need a single club can filter on it
    without losing the other.
    """
    season = games[["game_id", "season"]]

    squads = (game_lineups.merge(season, on="game_id")
              .groupby(_KEYS, as_index=False)
              .size()
              .rename(columns={"size": "squad_games"}))

    played = (appearances.rename(columns={"player_club_id": "club_id"})
              .merge(season, on="game_id")
              .groupby(_KEYS, as_index=False)
              .agg(played_games=("game_id", "size"), minutes=("minutes_played", "sum")))

    out = squads.merge(played, on=_KEYS, how="outer")
    out["played_games"] = out["played_games"].fillna(0).astype(int)
    out["minutes"] = out["minutes"].fillna(0).astype(int)
    # Playing implies being in the squad, so appearances set the floor. One rule covers
    # both season 2012, where game_lineups does not exist at all, and the individual
    # games it simply misses: 1,010 player-club-seasons in the real data have more
    # appearances than lineup entries.
    out["squad_games"] = (out["squad_games"].fillna(0)
                          .clip(lower=out["played_games"])
                          .astype(int))

    # club_id breaks ties so the choice is deterministic across runs
    out = out.sort_values(["player_id", "season", "squad_games", "minutes", "club_id"],
                          ascending=[True, True, False, False, True])
    out["is_primary"] = ~out.duplicated(["player_id", "season"])
    return out.reset_index(drop=True)


def players_at_anchor(player_club_season: pd.DataFrame,
                      club_domestic_league: pd.DataFrame,
                      anchor_year: int) -> pd.DataFrame:
    """Players eligible at a 1 July anchor, with the league they came from.

    Eligibility is decided by the season that has already finished, never by the club a
    player is registered to on the day: transfers run through the summer, so the new
    club is not reliably known at the anchor and using it would leak the future.

    Players whose club has no row in `club_domestic_league` are dropped, which is what
    limits the panel to the tracked leagues.
    """
    previous_season = anchor_year - 1
    eligible = player_club_season[player_club_season["season"] == previous_season]
    return (eligible.merge(club_domestic_league, on=["club_id", "season"], how="inner")
            .reset_index(drop=True))
