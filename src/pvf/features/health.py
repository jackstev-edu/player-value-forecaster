"""Slot 5: time lost to injury, and moves between clubs.

Both tables live in `football-datasets` rather than `player-scores`, and both carry the
same hazard: a player with no row is indistinguishable from a player nothing happened
to. Injury rows reach only 64.5% of panel players and transfer rows 88.0%, so a zero
would tell the model "never hurt" about thousands of players who are simply not tracked.
Every window column is therefore null for an untracked player, and a `has_*_record` flag
says which case a null is.

The injury spell is clipped to the window rather than taken whole. `days_missed` covers
the entire spell, so a player who goes down a fortnight before the anchor and is out for
a year would carry that year into a feature that is supposed to be knowable on the day —
the clearest leak in this slot. Injury reason is free text and is not read (decision #7).
"""
import numpy as np
import pandas as pd

# `transfer_history.transfer_type`. "Return from loan" is the other half of a loan and
# says the same thing about the player, so it is folded in with it.
_PERMANENT = {"Transfer"}
_LOAN = {"Loan", "Return from loan"}

_KEYS = ["player_id", "anchor_date"]


def _window(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel's keys plus the 12 months of history each anchor may look back on."""
    out = panel[_KEYS].drop_duplicates().copy()
    out["anchor_date"] = out["anchor_date"].astype("datetime64[ns]")
    out["window_start"] = out["anchor_date"] - pd.DateOffset(years=1)
    return out


def injury_window(injuries: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Days lost and spells started in the 12 months before each anchor.

    A spell is clipped to the window at both ends, so nothing after the anchor is
    counted and a long-running injury contributes only the part already served.
    """
    keys = _window(panel)

    spells = injuries[["player_id", "from_date", "end_date", "days_missed"]].drop_duplicates()
    spells = spells.dropna(subset=["from_date"])
    # 1,523 spells have no end date and 22 no duration. Falling back to the duration
    # keeps them; a spell with neither contributes no days but still counts as a spell,
    # which beats letting an open 2015 injury run to every later anchor.
    implied_end = spells["from_date"] + pd.to_timedelta(spells["days_missed"], unit="D")
    spells["spell_end"] = spells["end_date"].fillna(implied_end).fillna(spells["from_date"])

    m = keys.merge(spells, on="player_id", how="inner")
    overlaps = (m["spell_end"] > m["window_start"]) & (m["from_date"] <= m["anchor_date"])
    m = m[overlaps].copy()

    # Clip to the window: nothing before it opened, nothing after the anchor
    m["start"] = m[["from_date", "window_start"]].max(axis=1)
    m["end"] = m[["spell_end", "anchor_date"]].min(axis=1)
    m["covers_anchor"] = (m["from_date"] <= m["anchor_date"]) & (m["spell_end"] >= m["anchor_date"])

    counts = m.groupby(_KEYS, as_index=False).agg(
        injury_spells_12m=("start", "size"),
        injured_at_anchor=("covers_anchor", "max"))

    days = _union_days(m).rename(columns={"days": "days_injured_12m"})
    return counts.merge(days, on=_KEYS, how="left")


def _union_days(spells: pd.DataFrame) -> pd.DataFrame:
    """Days covered by at least one spell, per (player, anchor).

    Transfermarkt records the knock and the operation that follows it as separate rows,
    so a player's spells overlap. Adding their lengths counted the same fortnight twice
    and put 609 days inside a 365-day window on the real data; the union is the number
    of days of football actually missed.
    """
    m = spells.sort_values(_KEYS + ["start", "end"]).copy()
    by_row = m.groupby(_KEYS)
    # A spell opens a new block when it starts after every earlier spell has ended
    highest_end_so_far = by_row["end"].cummax()
    previous_end = highest_end_so_far.groupby([m[k] for k in _KEYS]).shift()
    opens_block = previous_end.isna() | (m["start"] > previous_end)
    m["block"] = opens_block.groupby([m[k] for k in _KEYS]).cumsum()

    blocks = m.groupby(_KEYS + ["block"], as_index=False).agg(
        start=("start", "min"), end=("end", "max"))
    blocks["days"] = (blocks["end"] - blocks["start"]).dt.days.clip(lower=0)
    return blocks.groupby(_KEYS, as_index=False)["days"].sum()


def transfer_window(transfers: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Moves made in the 12 months before each anchor.

    The anchor sits inside the summer window, so the cutoff is what keeps an incoming
    transfer agreed in July out of a feature dated 1 July.
    """
    keys = _window(panel)

    moves = transfers[["player_id", "transfer_date", "transfer_type",
                       "transfer_fee"]].drop_duplicates()
    moves = moves.dropna(subset=["transfer_date"])

    m = keys.merge(moves, on="player_id", how="inner")
    m = m[(m["transfer_date"] > m["window_start"]) &
          (m["transfer_date"] <= m["anchor_date"])].copy()

    m["is_permanent"] = m["transfer_type"].isin(_PERMANENT)
    m["is_loan"] = m["transfer_type"].isin(_LOAN)
    # 96.4% of rows carry fee 0, every loan among them. Zero is "not recorded", not
    # "moved for nothing", so it must not average in as a real price.
    m["fee"] = m["transfer_fee"].where(m["transfer_fee"] > 0, np.nan)

    return m.groupby(_KEYS, as_index=False).agg(
        transferred_12m=("is_permanent", "max"),
        loaned_12m=("is_loan", "max"),
        transfer_fee_12m=("fee", "max"))


def add_health_and_moves(panel: pd.DataFrame,
                         tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach injury and transfer history as it stood at each anchor."""
    injuries = tables["player_injuries"]
    transfers = tables["transfer_history"]

    out = panel.merge(injury_window(injuries, panel), on=_KEYS, how="left")
    out = out.merge(transfer_window(transfers, panel), on=_KEYS, how="left")

    # A tracked player with a quiet year scores zero; an untracked one stays null, and
    # the flag is what tells the two apart.
    out["has_injury_record"] = out["player_id"].isin(set(injuries["player_id"]))
    out["has_transfer_record"] = out["player_id"].isin(set(transfers["player_id"]))

    injury_cols = ["days_injured_12m", "injury_spells_12m", "injured_at_anchor"]
    move_cols = ["transferred_12m", "loaned_12m"]
    for cols, flag in ((injury_cols, "has_injury_record"), (move_cols, "has_transfer_record")):
        for c in cols:
            out[c] = out[c].astype(float).where(out[flag], np.nan)
            out.loc[out[flag] & out[c].isna(), c] = 0.0

    return out
