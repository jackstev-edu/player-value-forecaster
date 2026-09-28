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


# Manual rows with no value at their anchor (decision #44)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pvf.data.augment import backfill_first_value, shift_unanchored  # noqa: E402

NAN = np.nan


def srows(*rows):
    cols = ["rank", "player_id", "origin", "anchor_date", "value_now", "y_h1"]
    return pd.DataFrame([dict(zip(cols, (r[0], 1, r[1], T(r[2]), r[3], r[4]))) for r in rows],
                        columns=cols)


def test_unvalued_manual_row_moves_to_its_earliest_valued_span_row():
    manual = srows((1, "manual", "2019-07-01", NAN, NAN))
    span = srows((1, "span", "2021-07-01", 3e6, 0.2), (1, "span", "2020-07-01", 1e6, 0.1))
    m, s = shift_unanchored(manual, span)
    row = m.iloc[0]
    assert row["anchor_date"] == T("2020-07-01") and row["origin"] == "manual"
    assert row["anchor_shifted"]
    # The chosen span row is now the manual row, so it must leave the augmentation pool
    assert anchors(s) == ["2021-07-01"]


def test_span_rows_without_a_target_are_not_used_for_the_shift():
    manual = srows((1, "manual", "2019-07-01", NAN, NAN))
    span = srows((1, "span", "2020-07-01", 1e6, NAN), (1, "span", "2021-07-01", 3e6, 0.2))
    m, _ = shift_unanchored(manual, span)
    assert m.iloc[0]["anchor_date"] == T("2021-07-01")


def test_valued_manual_rows_are_left_alone():
    manual = srows((1, "manual", "2019-07-01", 5e6, 0.3))
    span = srows((1, "span", "2020-07-01", 1e6, 0.1))
    m, s = shift_unanchored(manual, span)
    assert m.iloc[0]["anchor_date"] == T("2019-07-01") and not m.iloc[0]["anchor_shifted"]
    assert len(s) == 1


def test_manual_row_without_any_valued_span_stays_unvalued():
    m, _ = shift_unanchored(srows((1, "manual", "2019-07-01", NAN, NAN)), srows())
    assert pd.isna(m.iloc[0]["value_now"]) and not m.iloc[0]["anchor_shifted"]


def vals(*points):
    return pd.DataFrame([{"player_id": 1, "date": T(d), "market_value_in_eur": v} for d, v in points])


def test_backfill_uses_the_first_value_after_the_anchor_when_none_before():
    s = srows((1, "manual", "2019-07-01", NAN, NAN))
    out = backfill_first_value(s, vals(("2019-12-01", 1e6), ("2020-06-01", 2e6)),
                               values_until="2026-06-12").iloc[0]
    assert out["value_now"] == 1e6 and out["value_backfilled"]
    assert out["y_h1"] == pytest.approx(np.log(2e6 / 1e6))  # value at 2020-07-01 over backfill


def test_backfill_prefers_a_stale_value_before_the_anchor():
    s = srows((1, "manual", "2019-07-01", NAN, NAN))
    out = backfill_first_value(s, vals(("2017-01-01", 4e6), ("2020-01-01", 6e6)),
                               values_until="2026-06-12").iloc[0]
    assert out["value_now"] == 4e6
    assert out["y_h1"] == pytest.approx(np.log(6e6 / 4e6))


def test_backfill_measures_a_year_after_the_first_value_if_it_lands_past_the_horizon():
    s = srows((1, "manual", "2019-07-01", NAN, NAN))
    out = backfill_first_value(s, vals(("2021-03-01", 1e6), ("2022-01-01", 3e6)),
                               values_until="2026-06-12").iloc[0]
    assert out["y_h1"] == pytest.approx(np.log(3e6 / 1e6))


def test_backfill_leaves_valued_rows_and_other_origins_alone():
    s = srows((1, "manual", "2019-07-01", 5e6, 0.3), (2, "span", "2019-07-01", NAN, NAN))
    out = backfill_first_value(s, vals(("2019-12-01", 1e6)), values_until="2026-06-12")
    assert out.loc[0, "value_now"] == 5e6 and not out["value_backfilled"].any()
