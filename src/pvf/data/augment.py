"""Contract-span augmentation: extra samples from contracts we already collected by hand.

A contract covers every 1 July between its start and end, not only the season we looked it
up for. Each extra 1 July becomes a sample with the same contract dates and its own anchor,
so its features and target are real, not generated. The one thing we cannot see is an
extension signed in between, which would make years_left too short; the club check below
removes the more common case, a move to another club.
"""
import numpy as np
import pandas as pd

from pvf.features.samples import asof


def _season_club(contracts: pd.DataFrame, valuations: pd.DataFrame) -> pd.Series:
    """Club the player was valued at during the hand-collected season, per contract row."""
    anchor = np.maximum(contracts["anchor_date"], contracts["contract_start"])
    rows = pd.DataFrame({"_row": contracts.index, "player_id": contracts["player_id"],
                         "_from": anchor, "_to": anchor + pd.DateOffset(years=1)})
    v = rows.merge(valuations[["player_id", "date", "current_club_id"]], on="player_id")
    v = v[(v["date"] >= v["_from"]) & (v["date"] < v["_to"])]
    during = v.groupby("_row")["current_club_id"].agg(lambda s: s.mode().iloc[0])
    # No valuation inside the season: fall back to the club on the anchor itself
    at_anchor = asof(valuations, contracts["player_id"], anchor, ["current_club_id"])
    return during.reindex(contracts.index).fillna(at_anchor["current_club_id"])


def span_contracts(contracts: pd.DataFrame, valuations: pd.DataFrame, *, horizon: int = 1,
                   values_until: str = "2026-06-12") -> pd.DataFrame:
    """One contract row per extra 1 July inside each contract, marked origin='span'."""
    c = contracts.reset_index(drop=True)
    club = _season_club(c, valuations)
    years = [range(s.year, e.year + 1) for s, e in zip(c["contract_start"], c["contract_end"])]
    out = c.assign(_year=years, _club=club).explode("_year")
    out["anchor_date"] = pd.to_datetime(out["_year"].astype(int).astype(str) + "-07-01")
    keep = ((out["anchor_date"] >= out["contract_start"])
            & (out["anchor_date"] < out["contract_end"])
            & (out["_year"] != c.loc[out.index, "anchor_date"].dt.year)
            & (out["anchor_date"] + pd.DateOffset(years=horizon) <= pd.Timestamp(values_until)))
    out = out[keep].reset_index(drop=True)
    # Drop seasons after the player moved: that contract no longer applies
    now = asof(valuations, out["player_id"], out["anchor_date"], ["current_club_id"])
    out = out[now["current_club_id"] == out["_club"]]
    return out.drop(columns=["_year", "_club"]).assign(origin="span").reset_index(drop=True)


def shift_unanchored(manual: pd.DataFrame, span: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move each manual row with no value or target to its earliest usable span row.

    The span row comes from the same hand-collected contract, so the row stays manual; it is
    only measured a season or more later (`anchor_shifted`). The span row it uses leaves the
    augmentation pool so it is not counted twice.
    """
    manual = manual.assign(anchor_shifted=False)
    missing = manual["value_now"].isna() | manual["y_h1"].isna()
    usable = span[span["rank"].isin(manual.loc[missing, "rank"])
                  & span["value_now"].notna() & span["y_h1"].notna()]
    first = usable.sort_values("anchor_date").drop_duplicates("rank")
    moved = first.assign(origin="manual", anchor_shifted=True).set_index("rank")

    keyed = manual.set_index("rank")
    replace = keyed.index.isin(moved.index) & missing.to_numpy()
    parts = [keyed[~replace], moved[moved.index.isin(keyed.index[replace])]]
    keyed = pd.concat([x for x in parts if len(x)]) if any(len(x) for x in parts) else keyed
    rest = span.drop(first.index)
    return keyed.reset_index()[manual.columns], rest.reset_index(drop=True)


def backfill_first_value(samples: pd.DataFrame, valuations: pd.DataFrame, *, horizon: int = 1,
                         values_until: str = "2026-06-12") -> pd.DataFrame:
    """Last resort for manual rows still without a value (`value_backfilled`, decision #44).

    Uses the latest value before the anchor however old, else the player's first value after
    it. The second case borrows a value from after the anchor, which a live forecast could not
    have; it is accepted for these few rows and flagged so evaluation can leave them out.
    The target is measured `horizon` years after whichever is later, the anchor or that value.
    """
    out = samples.assign(value_backfilled=False)
    rows = out[(out["origin"] == "manual") & out["value_now"].isna()]
    if rows.empty:
        return out
    v = valuations[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    left = pd.DataFrame({"player_id": rows["player_id"].to_numpy(),
                         "_at": rows["anchor_date"].to_numpy(), "_row": rows.index})
    base = {}
    for direction in ("backward", "forward"):
        m = pd.merge_asof(left.sort_values("_at"), v, left_on="_at", right_on="date",
                          by="player_id", direction=direction).set_index("_row")
        base[direction] = m.reindex(rows.index)
    before, after = base["backward"], base["forward"]
    value = before["market_value_in_eur"].fillna(after["market_value_in_eur"])
    value_date = before["date"].fillna(after["date"])

    target_at = np.maximum(rows["anchor_date"], value_date) + pd.DateOffset(years=horizon)
    later = asof(valuations, rows["player_id"], target_at, ["market_value_in_eur"])
    observed = (target_at <= pd.Timestamp(values_until)) & (later["date"] > value_date)
    out.loc[rows.index, "value_now"] = value
    out.loc[rows.index, f"y_h{horizon}"] = np.log(later["market_value_in_eur"] / value).where(observed)
    out.loc[rows.index, "value_backfilled"] = value.notna()
    return out


def cap_augmented(samples: pd.DataFrame, total: int | None, seed: int) -> pd.DataFrame:
    """Keep every manual row and a seeded sample of augmented rows, up to `total` in all."""
    manual = samples[samples["origin"] == "manual"]
    extra = samples[samples["origin"] != "manual"]
    room = None if total is None else max(total - len(manual), 0)
    if room is not None and len(extra) > room:
        extra = extra.sample(n=room, random_state=seed).sort_index()
    return pd.concat([manual, extra]).reset_index(drop=True)
