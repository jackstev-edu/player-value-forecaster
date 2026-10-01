"""Build the app's bundle tables from real data: who is shown, their history, their forecast."""
import numpy as np
import pandas as pd

# Competition code -> (display name, country) for the nine leagues in scope
LEAGUES = {
    "GB1": ("Premier League", "England"), "ES1": ("LaLiga", "Spain"),
    "IT1": ("Serie A", "Italy"), "L1": ("Bundesliga", "Germany"),
    "FR1": ("Ligue 1", "France"), "NL1": ("Eredivisie", "Netherlands"),
    "PO1": ("Liga Portugal", "Portugal"), "BE1": ("Jupiler Pro League", "Belgium"),
    "TR1": ("Super Lig", "Turkey"),
}


def universe(panel: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """One row per player in a league squad at the anchor: their primary panel row."""
    at = panel[(panel["anchor_date"] == anchor) & panel["is_primary"]]
    return (at[["player_id", "club_id", "competition_id", "sub_position"]]
            .drop_duplicates("player_id").reset_index(drop=True))


def _whole_years(dob: pd.Series, at: pd.Timestamp) -> pd.Series:
    """Age in completed years on `at`, as a person would state it."""
    before_birthday = (dob.dt.month > at.month) | ((dob.dt.month == at.month) & (dob.dt.day > at.day))
    return (at.year - dob.dt.year - before_birthday.astype(int)).astype("Int64")


def players_table(univ: pd.DataFrame, core: pd.DataFrame, kaggle_players: pd.DataFrame,
                  clubs: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """Profile rows for the app; `core` is aligned to `univ`. Players with no current value are dropped."""
    profile = kaggle_players.drop_duplicates("player_id").set_index("player_id")
    club_names = clubs.drop_duplicates("club_id").set_index("club_id")["name"]
    ids = univ["player_id"]
    league = univ["competition_id"].map(LEAGUES)
    p = pd.DataFrame({
        "player_id": ids,
        "name": ids.map(profile["name"]),
        "age": _whole_years(ids.map(profile["date_of_birth"]), anchor),
        "position": core["position"],
        "sub_position": univ["sub_position"],
        "nationality": ids.map(profile["country_of_citizenship"]),
        "club_name": univ["club_id"].map(club_names),
        "league_id": univ["competition_id"],
        "league_name": league.str[0],
        "league_country": league.str[1],
        "current_value_eur": core["value_now"],
        "value_date": core["value_date"],
    })
    return p[p["current_value_eur"].notna()].reset_index(drop=True)


def history_table(valuations: pd.DataFrame, player_ids, anchor: pd.Timestamp) -> pd.DataFrame:
    """Every valuation of the given players up to the anchor; later ones would leak the answer."""
    v = valuations[valuations["player_id"].isin(set(player_ids)) & (valuations["date"] <= anchor)]
    return (v.rename(columns={"market_value_in_eur": "value_eur"})[["player_id", "date", "value_eur"]]
            .sort_values(["player_id", "date"]).reset_index(drop=True))


def forecasts_table(current_eur: pd.Series, q: pd.DataFrame, anchor: pd.Timestamp,
                    model: str) -> pd.DataFrame:
    """One-season-ahead bands in euros from log-ratio quantiles q10/q50/q90, both indexed by player."""
    # Sorting guards against crossed quantiles, which the schema rejects
    logq = np.sort(q[["q10", "q50", "q90"]].to_numpy(), axis=1)
    now = current_eur.reindex(q.index).to_numpy()[:, None]
    eur = now * np.exp(logq)
    return pd.DataFrame({
        "player_id": q.index.to_numpy(), "horizon": 1, "anchor_date": anchor,
        "target_date": anchor + pd.DateOffset(years=1),
        "p10_eur": eur[:, 0], "p50_eur": eur[:, 1], "p90_eur": eur[:, 2], "model": model,
    })
