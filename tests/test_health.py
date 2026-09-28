import numpy as np
import pandas as pd

from pvf.features.health import add_health_and_moves

ANCHOR = pd.Timestamp("2020-07-01")


def _injuries():
    """Player 1 has a spell inside the window and one long before it.
    Player 2's spell straddles the window start. Player 3's spell is running on the
    anchor date and ends long after. Player 4 is tracked but was never hurt recently.
    Player 5 has no row at all."""
    rows = [
        (1, "2019-09-01", "2019-10-01", 30.0),    # 30 days, fully inside
        (1, "2015-01-01", "2015-06-01", 151.0),   # long before the window
        (2, "2019-06-01", "2019-08-01", 61.0),    # starts a month before the window
        (3, "2020-06-01", "2021-06-01", 365.0),   # running on the anchor, ends far after
        (4, "2016-01-01", "2016-02-01", 31.0),    # tracked, nothing recent
        # Transfermarkt logs a knock and the surgery for it as separate rows, so
        # spells overlap. 1 Sep to 15 Oct is 44 days of football missed, not 60.
        (6, "2019-09-01", "2019-10-01", 30.0),
        (6, "2019-09-15", "2019-10-15", 30.0),
        (7, "2019-09-01", "2019-10-01", 30.0),
        (7, "2019-11-01", "2019-12-01", 30.0),   # a clear gap between the two
        (8, "2020-08-15", "2020-12-01", 108.0),  # his first injury, a month AFTER
    ]
    return pd.DataFrame(rows, columns=["player_id", "from_date", "end_date",
                                       "days_missed"]).assign(
        from_date=lambda d: pd.to_datetime(d["from_date"]),
        end_date=lambda d: pd.to_datetime(d["end_date"]))


def _transfers():
    """Player 1 was bought inside the window. Player 2 went out on loan. Player 3 moved
    after the anchor. Player 4 moved for an unrecorded fee. Player 5 has no row."""
    rows = [
        (1, "2019-08-01", "Transfer", 5_000_000),
        (1, "2019-09-01", "Transfer", 1_000_000),   # smaller, same window
        (2, "2020-01-31", "Loan", 0),
        (3, "2018-01-01", "Transfer", 4_000_000),   # long before the window
        (3, "2020-08-01", "Transfer", 20_000_000),  # after the anchor
        (4, "2019-07-15", "Transfer", 0),           # fee not recorded
        (8, "2020-09-01", "Transfer", 9_000_000),   # his first move, AFTER
    ]
    return pd.DataFrame(rows, columns=["player_id", "transfer_date", "transfer_type",
                                       "transfer_fee"]).assign(
        transfer_date=lambda d: pd.to_datetime(d["transfer_date"]))


def _panel():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "anchor_date": [ANCHOR] * 8,
    })


def _tables():
    return {"player_injuries": _injuries(), "transfer_history": _transfers()}


def _out():
    return add_health_and_moves(_panel(), _tables()).set_index("player_id")


def test_days_injured_counts_a_spell_inside_the_window():
    assert _out().loc[1, "days_injured_12m"] == 30


def test_a_spell_long_before_the_window_is_ignored():
    # Player 1's 2015 spell must not be added to his 30 recent days
    assert _out().loc[1, "days_injured_12m"] == 30


def test_a_spell_straddling_the_window_start_counts_only_the_overlap():
    # Player 2 was out 1 June to 1 Aug 2019; the window opens 1 July 2019
    assert _out().loc[2, "days_injured_12m"] == 31


def test_an_open_spell_is_not_counted_past_the_anchor():
    # Player 3 goes down on 1 June 2020 and is out until June 2021. Only the 30 days
    # before the anchor were knowable then; the rest is the future.
    assert _out().loc[3, "days_injured_12m"] == 30


def test_days_injured_never_exceeds_the_window():
    out = _out()
    assert out["days_injured_12m"].max() <= 366


def test_a_player_out_on_the_anchor_date_is_flagged():
    out = _out()
    assert out.loc[3, "injured_at_anchor"] == 1
    assert out.loc[1, "injured_at_anchor"] == 0


def test_a_tracked_player_with_no_recent_injury_scores_zero_not_null():
    # Player 4 has an injury history, so "no days" is a measurement
    out = _out()
    assert out.loc[4, "has_injury_record"]
    assert out.loc[4, "days_injured_12m"] == 0
    assert out.loc[4, "injury_spells_12m"] == 0


def test_an_untracked_player_is_null_not_zero():
    # Player 5 has no injury row anywhere. Zero would claim he was never hurt.
    out = _out()
    assert not out.loc[5, "has_injury_record"]
    assert pd.isna(out.loc[5, "days_injured_12m"])
    assert pd.isna(out.loc[5, "injury_spells_12m"])


def test_spells_are_counted_only_inside_the_window():
    out = _out()
    assert out.loc[1, "injury_spells_12m"] == 1


def test_overlapping_spells_count_the_union_not_the_sum():
    # Two 30-day spells overlapping by a fortnight are 44 days out, not 60
    out = _out()
    assert out.loc[6, "days_injured_12m"] == 44
    assert out.loc[6, "injury_spells_12m"] == 2


def test_separate_spells_still_add_up():
    assert _out().loc[7, "days_injured_12m"] == 60


def test_a_permanent_move_inside_the_window_is_flagged():
    out = _out()
    assert out.loc[1, "transferred_12m"] == 1
    assert out.loc[1, "loaned_12m"] == 0


def test_a_loan_is_flagged_separately_from_a_permanent_move():
    out = _out()
    assert out.loc[2, "loaned_12m"] == 1
    assert out.loc[2, "transferred_12m"] == 0


def test_a_move_after_the_anchor_is_invisible():
    # Player 3 last moved in 2018, so he is on record at the anchor and scores a real
    # zero. His August transfer is the summer window the panel must never see, and its
    # 20M fee must not appear either.
    out = _out()
    assert out.loc[3, "has_transfer_record"]
    assert out.loc[3, "transferred_12m"] == 0
    assert pd.isna(out.loc[3, "transfer_fee_12m"])


def test_the_largest_fee_in_the_window_is_kept():
    assert _out().loc[1, "transfer_fee_12m"] == 5_000_000


def test_a_zero_fee_means_unrecorded_not_free():
    # 96.4% of transfer_history rows carry fee 0, including every loan
    out = _out()
    assert out.loc[4, "transferred_12m"] == 1
    assert pd.isna(out.loc[4, "transfer_fee_12m"])


def test_a_player_with_no_transfer_row_is_null_not_zero():
    out = _out()
    assert not out.loc[5, "has_transfer_record"]
    assert pd.isna(out.loc[5, "transferred_12m"])


def test_a_record_that_starts_after_the_anchor_does_not_exist_yet():
    """Coverage is itself a feature, so it has to be as-of the anchor. Player 8 is hurt
    in August and sold in September; on 1 July neither has happened, and knowing he is
    about to appear in both files is knowing the future."""
    out = _out()
    assert not out.loc[8, "has_injury_record"]
    assert not out.loc[8, "has_transfer_record"]
    assert pd.isna(out.loc[8, "days_injured_12m"])
    assert pd.isna(out.loc[8, "transferred_12m"])


def test_health_does_not_multiply_panel_rows():
    panel = _panel()
    assert len(add_health_and_moves(panel, _tables())) == len(panel)


def _full_tables():
    """One anchor at 2020-07-01, drawn from season 2019."""
    return {
        "games": pd.DataFrame({
            "game_id": [1], "competition_id": ["GB1"], "season": [2019],
            "home_club_id": [10], "away_club_id": [20],
        }),
        "game_lineups": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2], "club_id": [10, 20],
            "type": ["starting_lineup", "starting_lineup"],
        }),
        "appearances": pd.DataFrame({
            "game_id": [1, 1], "player_id": [1, 2],
            "player_club_id": [10, 20], "minutes_played": [90, 90],
            "goals": [0, 0], "assists": [0, 0],
        }),
        "players": pd.DataFrame({
            "player_id": [1, 2],
            "date_of_birth": pd.to_datetime(["1995-01-01", "1996-01-01"]),
            "position": ["Attack", "Defender"],
        }),
        "player_valuations": pd.DataFrame({
            "player_id": [1, 1, 2],
            "date": pd.to_datetime(["2019-06-01", "2020-06-01", "2020-06-01"]),
            "market_value_in_eur": [5e6, 10e6, 3e6],
        }),
        "player_injuries": _injuries(),
        "transfer_history": _transfers(),
    }


def _full_cfg():
    return {
        "panel": {"anchor_month_day": "07-01", "min_anchor_season": 2020,
                  "max_value_staleness_days": 365, "leagues": ["GB1"]},
        "target": {"horizons": [1]},
        "split": {"values_available_until": "2020-06-12"},
    }


def test_build_panel_attaches_health_and_moves():
    from pvf.features.build_panel import build_panel
    panel = build_panel(_full_tables(), _full_cfg()).set_index("player_id")
    assert panel.loc[1, "days_injured_12m"] == 30
    assert panel.loc[1, "transferred_12m"] == 1
    assert panel.loc[1, "transfer_fee_12m"] == 5_000_000
    assert panel.loc[2, "loaned_12m"] == 1
