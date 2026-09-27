"""Checks on the hand-collected contract sheet (Task 19)."""
import pandas as pd

from pvf.data.manual import check_contracts, read_contracts, usable_contracts


def contract(rank, start, end, anchor="2021-07-01", player_id=None,
             source="https://example.com/a"):
    return {"rank": rank, "collector": "Yunus", "player_id": player_id or rank,
            "player_name": f"P{rank}", "season": "2021/22", "anchor_date": pd.Timestamp(anchor),
            "club_that_season": "Club", "contract_start": pd.Timestamp(start) if start else pd.NaT,
            "contract_end": pd.Timestamp(end) if end else pd.NaT, "source_url": source, "notes": None}


def rules_for(df, rank):
    issues = check_contracts(df)
    return set(issues.loc[issues["rank"] == rank, "rule"])


def test_clean_row_has_no_issues():
    df = pd.DataFrame([contract(1, "2020-07-01", "2024-06-30")])
    assert check_contracts(df).empty


def test_blank_dates_are_an_error():
    df = pd.DataFrame([contract(1, None, None)])
    assert rules_for(df, 1) == {"missing_dates"}


def test_end_before_start_is_an_error():
    df = pd.DataFrame([contract(1, "2022-07-01", "2021-06-30", anchor="2021-07-01")])
    assert "end_not_after_start" in rules_for(df, 1)


def test_contract_that_expired_before_the_anchor_is_an_error():
    df = pd.DataFrame([contract(1, "2018-07-01", "2021-06-30")])
    assert rules_for(df, 1) == {"expired_before_anchor"}


def test_contract_starting_after_the_season_is_an_error():
    # Signed a full year after the 1 July anchor, so it belongs to a later season
    df = pd.DataFrame([contract(1, "2022-08-01", "2026-06-30")])
    assert rules_for(df, 1) == {"starts_after_season"}


def test_summer_signing_after_the_anchor_is_allowed():
    df = pd.DataFrame([contract(1, "2021-08-15", "2026-06-30")])
    assert check_contracts(df).empty


def test_non_url_source_is_only_a_warning():
    df = pd.DataFrame([contract(1, "2020-07-01", "2024-06-30", source="Team Website")])
    issues = check_contracts(df)
    assert list(issues["rule"]) == ["source_not_url"]
    assert list(issues["severity"]) == ["warning"]
    assert len(usable_contracts(df)) == 1


def test_duplicate_player_is_an_error_on_both_rows():
    df = pd.DataFrame([contract(1, "2020-07-01", "2024-06-30", player_id=7),
                       contract(2, "2020-07-01", "2024-06-30", player_id=7)])
    assert rules_for(df, 1) == {"duplicate_player"} == rules_for(df, 2)


def test_usable_contracts_drops_only_error_rows():
    df = pd.DataFrame([contract(1, "2020-07-01", "2024-06-30"),
                       contract(2, None, None),
                       contract(3, "2020-07-01", "2024-06-30", source="Team Website")])
    assert list(usable_contracts(df)["rank"]) == [1, 3]


def test_read_contracts_parses_dates(tmp_path):
    path = tmp_path / "contracts.csv"
    pd.DataFrame([contract(1, "2020-07-01", "2024-06-30")]).to_csv(path, index=False)
    df = read_contracts(path)
    for col in ["anchor_date", "contract_start", "contract_end"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col])
