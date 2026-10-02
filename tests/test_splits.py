"""Player-grouped folds: no player on both sides, augmented rows never scored."""
import pandas as pd

from pvf.evaluation.splits import player_folds


def samples():
    rows = []
    for pid in range(20):
        rows.append({"player_id": pid, "origin": "manual"})
        rows += [{"player_id": pid, "origin": "span"}] * 2
    return pd.DataFrame(rows)


def test_no_player_is_on_both_sides_of_a_fold():
    df = samples()
    for train, test in player_folds(df, n_splits=5, seed=0):
        assert not set(df.loc[train, "player_id"]) & set(df.loc[test, "player_id"])


def test_only_manual_rows_are_scored():
    df = samples()
    for _, test in player_folds(df, n_splits=5, seed=0):
        assert set(df.loc[test, "origin"]) == {"manual"}


def test_augmented_rows_are_used_for_training():
    df = samples()
    train, _ = player_folds(df, n_splits=5, seed=0)[0]
    assert "span" in set(df.loc[train, "origin"])


def test_every_manual_row_is_scored_exactly_once():
    df = samples()
    scored = [i for _, test in player_folds(df, n_splits=5, seed=0) for i in test]
    assert sorted(scored) == sorted(df.index[df["origin"] == "manual"])


def test_folds_are_repeatable_for_a_seed():
    a = player_folds(samples(), n_splits=5, seed=3)
    b = player_folds(samples(), n_splits=5, seed=3)
    assert all(list(x[1]) == list(y[1]) for x, y in zip(a, b))
