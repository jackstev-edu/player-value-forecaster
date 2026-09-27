"""Contract-span augmentation: more seasons from the same hand-collected contract."""
import pandas as pd

from pvf.data.augment import cap_augmented, span_contracts

T = pd.Timestamp


def contract(start="2019-07-01", end="2023-06-30", anchor="2020-07-01"):
    return pd.DataFrame([{"rank": 1, "player_id": 1, "anchor_date": T(anchor),
                          "contract_start": T(start), "contract_end": T(end)}])


def clubs(*points):
    return pd.DataFrame([{"player_id": 1, "date": T(d), "market_value_in_eur": 1e6,
                          "current_club_id": club} for d, club in points])


SAME_CLUB = clubs(("2019-01-01", 5), ("2020-10-01", 5), ("2022-01-01", 5))


def anchors(df):
    return sorted(df["anchor_date"].dt.strftime("%Y-%m-%d"))


def test_every_other_first_of_july_inside_the_contract_becomes_a_row():
    out = span_contracts(contract(), SAME_CLUB, values_until="2026-06-12")
    assert anchors(out) == ["2019-07-01", "2021-07-01", "2022-07-01"]


def test_rows_keep_the_contract_and_are_marked_as_span():
    out = span_contracts(contract(), SAME_CLUB, values_until="2026-06-12")
    assert set(out["origin"]) == {"span"}
    assert (out["contract_end"] == T("2023-06-30")).all()
    assert (out["rank"] == 1).all()


def test_anchor_before_a_mid_summer_start_is_skipped():
    out = span_contracts(contract(start="2019-08-10"), SAME_CLUB, values_until="2026-06-12")
    assert anchors(out) == ["2021-07-01", "2022-07-01"]


def test_anchor_whose_horizon_passes_the_data_is_skipped():
    out = span_contracts(contract(), SAME_CLUB, values_until="2022-06-30")
    assert anchors(out) == ["2019-07-01"]


def test_anchor_after_a_move_to_another_club_is_skipped():
    v = clubs(("2019-01-01", 5), ("2020-10-01", 5), ("2022-01-01", 9))
    out = span_contracts(contract(), v, values_until="2026-06-12")
    assert anchors(out) == ["2019-07-01", "2021-07-01"]


def samples(n_manual, n_span):
    return pd.DataFrame({"origin": ["manual"] * n_manual + ["span"] * n_span,
                         "rank": range(n_manual + n_span)})


def test_cap_keeps_every_manual_row_and_fills_to_the_total():
    out = cap_augmented(samples(5, 10), total=8, seed=0)
    assert len(out) == 8
    assert (out["origin"] == "manual").sum() == 5


def test_cap_is_repeatable_for_a_seed():
    a = cap_augmented(samples(5, 10), total=8, seed=0)
    b = cap_augmented(samples(5, 10), total=8, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_cap_is_a_no_op_below_the_total_or_without_one():
    assert len(cap_augmented(samples(5, 2), total=8, seed=0)) == 7
    assert len(cap_augmented(samples(5, 10), total=None, seed=0)) == 15
