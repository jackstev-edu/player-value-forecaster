"""Player-grouped cross-validation for the 500-player contract dataset.

A time split no longer fits: 194 of the 500 hand-collected rows are 2023/24. Instead every
player sits wholly on one side of each fold, so a model is never scored on a player it
trained on, and only hand-collected rows are scored. Augmented rows of the training
players are added to training only, which keeps them out of every reported number.
"""
import numpy as np
import pandas as pd


def player_folds(samples: pd.DataFrame, n_splits: int = 5,
                 seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_index, test_index) pairs of row labels; test holds manual rows only."""
    players = samples["player_id"].unique()
    # Shuffle players with the seed, then deal them into folds
    order = np.random.default_rng(seed).permutation(len(players))
    fold_of = pd.Series(order % n_splits, index=players)
    folds = samples["player_id"].map(fold_of).to_numpy()
    manual = (samples["origin"] == "manual").to_numpy()
    idx = samples.index.to_numpy()
    return [(idx[folds != k], idx[(folds == k) & manual]) for k in range(n_splits)]
