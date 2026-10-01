import pandas as pd

from pvf.evaluation.backtest import ROLES, known_by, pick_players

ANCHOR = pd.Timestamp("2024-07-01")


def test_training_keeps_only_outcomes_known_by_the_anchor():
    s = pd.DataFrame({
        "anchor_date": pd.to_datetime(["2022-07-01", "2023-07-01", "2023-08-15", "2024-07-01"]),
        "y_h1": [0.1, 0.2, 0.3, 0.4],
    })
    kept = known_by(s, ANCHOR)
    # 2023-08-15 + 1 year lands after the anchor, so its outcome was still unknown
    assert kept["y_h1"].tolist() == [0.1, 0.2]


def test_known_by_drops_rows_without_a_target():
    s = pd.DataFrame({"anchor_date": pd.to_datetime(["2021-07-01", "2022-07-01"]),
                      "y_h1": [0.1, None]})
    assert len(known_by(s, ANCHOR)) == 1


def profiles():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6, 7],
        "age": [24.0, 19.0, 26.5, 25.0, 33.0, 20.0, 34.0],
        "value_now": [150e6, 60e6, 21e6, 2.2e6, 8e6, 30e6, 12e6],
    })


def test_each_role_follows_its_stated_rule():
    picks = pick_players(profiles())
    assert list(picks) == ROLES
    assert picks["star"] == 1          # highest value
    assert picks["prospect"] == 2      # highest value under 21
    assert picks["prime"] == 3         # 25-28, nearest EUR 20M
    assert picks["squad"] == 4         # 23-27, nearest EUR 2M
    assert picks["veteran"] == 7       # highest value at 32+


def test_a_player_fills_at_most_one_role():
    one = pd.DataFrame({"player_id": [1, 2, 3, 4, 5, 6],
                        "age": [26.0, 19.0, 26.0, 25.0, 33.0, 27.0],
                        "value_now": [20e6, 5e6, 19e6, 2e6, 3e6, 1.9e6]})
    picks = pick_players(one)
    assert picks["star"] == 1 and picks["prime"] == 3  # 1 is taken, so the next nearest
    assert len(set(picks.values())) == len(picks)
