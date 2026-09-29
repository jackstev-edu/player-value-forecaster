"""One model sample per contract row: features known at the anchor, value change after it.

The anchor is 1 July of the row's season, or the contract start when the contract was signed
later that summer, so the contract a row describes is always in force at its own anchor.
"""
import numpy as np
import pandas as pd

SAMPLE_FEATURES = ["age", "position", "value_now", "value_change_12m",
                   "years_left", "contract_years", "years_into_contract"]
# Gathered player-season features, joined from the panel by (player_id, season). An
# allowlist, so panel targets (y_h*, value_h*) and panel copies of sample columns stay out.
PANEL_FEATURES = [
    # playing time and output, last season
    "squad_games", "played_games", "minutes", "starts", "start_share",
    "goals", "assists", "goals_assists_per90",
    # club and league strength
    "competition_id", "club_squad_size", "club_squad_value_eur", "club_median_value_eur",
    "club_top5_value_eur", "league_club_count", "league_total_value_eur",
    "league_median_club_value_eur", "club_value_share_of_league",
    "played_ucl", "played_uel", "played_uecl",
    # rank against peers
    "position_rank_at_club", "position_peers_at_club", "position_pctile_at_club",
    "position_rank_in_league", "position_peers_in_league", "position_pctile_in_league",
    # value history, health, moves, fixed profile
    "peak_value_eur", "value_vs_peak", "n_valuations_so_far", "years_of_history",
    "injury_spells_12m", "injured_at_anchor", "days_injured_12m", "has_injury_record",
    "transferred_12m", "loaned_12m", "transfer_fee_12m", "has_transfer_record",
    "sub_position", "foot", "height_in_cm", "is_eu",
]
FEATURES = SAMPLE_FEATURES + PANEL_FEATURES
# Groups for the ablation: each is dropped in turn from the full model. Age, position and
# current value are never dropped, since every baseline already has them.
CORE_FEATURES = ["age", "position", "value_now", "value_change_12m"]
FEATURE_GROUPS = {
    "contract": ["years_left", "contract_years", "years_into_contract"],
    "performance": ["squad_games", "played_games", "minutes", "starts", "start_share",
                    "goals", "assists", "goals_assists_per90"],
    "club_league": ["competition_id", "club_squad_size", "club_squad_value_eur",
                    "club_median_value_eur", "club_top5_value_eur", "league_club_count",
                    "league_total_value_eur", "league_median_club_value_eur",
                    "club_value_share_of_league", "played_ucl", "played_uel", "played_uecl"],
    "peer_rank": ["position_rank_at_club", "position_peers_at_club", "position_pctile_at_club",
                  "position_rank_in_league", "position_peers_in_league",
                  "position_pctile_in_league"],
    "value_history": ["peak_value_eur", "value_vs_peak", "n_valuations_so_far", "years_of_history"],
    "injuries": ["injury_spells_12m", "injured_at_anchor", "days_injured_12m", "has_injury_record"],
    "transfers": ["transferred_12m", "loaned_12m", "transfer_fee_12m", "has_transfer_record"],
    "profile": ["sub_position", "foot", "height_in_cm", "is_eu"],
}
YEAR_DAYS = 365.25


def add_panel_features(samples: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join the panel row measured on the sample's own anchor summer; `in_panel` marks a match.

    A panel season is the season just played, so panel season S is measured on 1 July S+1,
    while a sample of season S is anchored on 1 July S or later that summer. The key is
    therefore the year of the panel's anchor, never the season label: equal labels would
    hand each sample the snapshot taken on its own target date. Other years are never borrowed.
    """
    cols = [c for c in PANEL_FEATURES if c in panel.columns]
    right = (panel[["player_id", "anchor_date", *cols]]
             .rename(columns={"anchor_date": "panel_anchor"})
             .assign(season=lambda d: d["panel_anchor"].dt.year, in_panel=True))
    joined = samples.merge(right, on=["player_id", "season"], how="left", validate="many_to_one")
    late = joined["panel_anchor"] > joined["anchor_date"]
    if late.any():
        raise ValueError(f"{int(late.sum())} panel rows measured after their sample's anchor")
    joined = joined.drop(columns="panel_anchor")
    joined["in_panel"] = joined["in_panel"].notna()
    for c in PANEL_FEATURES:
        if c not in joined:
            joined[c] = np.nan
    return joined


def drop_unmatched_augmented(samples: pd.DataFrame) -> pd.DataFrame:
    """Augmented rows need a panel row; manual rows are collected data and always stay."""
    keep = samples["in_panel"] | (samples["origin"] == "manual")
    return samples[keep].reset_index(drop=True)


def asof(valuations: pd.DataFrame, player_id: pd.Series, at: pd.Series,
         columns: list[str]) -> pd.DataFrame:
    """The player's latest valuation row on or before each date: `columns` plus its date."""
    # merge_asof needs both date keys at the same resolution; a Series built from one
    # Timestamp comes out in seconds
    left = pd.DataFrame({"player_id": player_id.to_numpy(),
                         "_at": at.to_numpy().astype("datetime64[ns]"),
                         "_row": np.arange(len(at))}).sort_values("_at")
    right = (valuations[["player_id", "date", *columns]]
             .astype({"date": "datetime64[ns]"}).sort_values("date"))
    m = pd.merge_asof(left, right, left_on="_at", right_on="date", by="player_id",
                      direction="backward").sort_values("_row")
    return m[[*columns, "date"]].set_axis(at.index)


def _value_asof(valuations: pd.DataFrame, player_id: pd.Series, at: pd.Series) -> pd.DataFrame:
    return asof(valuations, player_id, at, ["market_value_in_eur"]).rename(
        columns={"market_value_in_eur": "value"})


def core_features_at(player_id: pd.Series, anchor: pd.Series, valuations: pd.DataFrame,
                     players: pd.DataFrame, *, max_staleness_days: int = 365) -> pd.DataFrame:
    """Age, position, current value and its 12-month log change, as known at each anchor.

    Needs no contract or panel row, so any valued player can be scored. `value_date` is the
    date of the valuation behind `value_now`.
    """
    now = _value_asof(valuations, player_id, anchor)
    # A value older than the staleness limit says little about the player at the anchor
    fresh = (anchor - now["date"]).dt.days <= max_staleness_days
    out = pd.DataFrame(index=anchor.index)
    out["value_now"] = now["value"].where(fresh)
    out["value_date"] = now["date"].where(fresh)
    year_ago = _value_asof(valuations, player_id, anchor - pd.DateOffset(years=1))
    out["value_change_12m"] = np.log(out["value_now"] / year_ago["value"])

    profile = players.drop_duplicates("player_id").set_index("player_id")
    ids = pd.Series(player_id.to_numpy(), index=anchor.index)
    out["age"] = (anchor - ids.map(profile["date_of_birth"])).dt.days / YEAR_DAYS
    out["position"] = ids.map(profile["position"])
    return out


def build_samples(contracts: pd.DataFrame, valuations: pd.DataFrame, players: pd.DataFrame, *,
                  horizon: int = 1, values_until: str = "2026-06-12",
                  max_staleness_days: int = 365) -> pd.DataFrame:
    """Features and log-ratio target y_h{horizon} for each contract row."""
    c = contracts.reset_index(drop=True)
    anchor = np.maximum(c["anchor_date"], c["contract_start"])
    out = pd.DataFrame({
        "rank": c["rank"],
        "player_id": c["player_id"],
        "origin": c["origin"] if "origin" in c else "manual",
        "season": c["anchor_date"].dt.year,
        "anchor_date": anchor,
    })

    core = core_features_at(c["player_id"], anchor, valuations, players,
                            max_staleness_days=max_staleness_days)
    for col in ["value_now", "value_change_12m", "age", "position"]:
        out[col] = core[col]
    out["years_left"] = (c["contract_end"] - anchor).dt.days / YEAR_DAYS
    out["contract_years"] = (c["contract_end"] - c["contract_start"]).dt.days / YEAR_DAYS
    out["years_into_contract"] = (anchor - c["contract_start"]).dt.days / YEAR_DAYS

    target_at = anchor + pd.DateOffset(years=horizon)
    later = _value_asof(valuations, c["player_id"], target_at)
    # No target if the horizon runs past the data, or no valuation landed after the anchor
    observed = (target_at <= pd.Timestamp(values_until)) & (later["date"] > anchor)
    out[f"y_h{horizon}"] = np.log(later["value"] / out["value_now"]).where(observed)
    return out
