import numpy as np
import pandas as pd
import pytest

from pvf.models.gbm import QuantileGBM

FEATURES = ["age", "position", "value_change_12m"]
FAST = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5,
        "verbose": -1}


def _frame(n: int, seed: int = 0, noise: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 34, n)
    pos = rng.choice(["Attack", "Defender", "Goalkeeper"], n)
    # Young players rise, old ones fall, plus noise so the quantile band has width
    y = 0.08 * (26 - age) + rng.normal(0, noise, n)
    return pd.DataFrame({"age": age, "position": pos,
                         "value_change_12m": rng.normal(0, 0.5, n), "y_h1": y})


def _fit(train, val=None):
    return QuantileGBM(horizons=[1], quantiles=[0.1, 0.5, 0.9], params=FAST).fit(train, val, FEATURES)


def test_predict_shape_and_columns():
    train, test = _frame(400), _frame(50, seed=1)
    out = _fit(train, _frame(100, seed=2)).predict(test)
    assert list(out.columns) == ["horizon", "q10", "q50", "q90"]
    assert len(out) == 50
    assert (out.index == test.index).all()


def test_quantiles_do_not_cross():
    out = _fit(_frame(400)).predict(_frame(200, seed=3))
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()


def test_learns_age_signal():
    model = _fit(_frame(600))
    young = pd.DataFrame({"age": [19.0], "position": ["Attack"], "value_change_12m": [0.0]})
    old = young.assign(age=33.0)
    assert model.predict(young)["q50"].iloc[0] > model.predict(old)["q50"].iloc[0]


def test_band_covers_roughly_80_percent():
    model = _fit(_frame(1500))
    test = _frame(500, seed=4)
    out = model.predict(test)
    cover = ((test["y_h1"] >= out["q10"]) & (test["y_h1"] <= out["q90"])).mean()
    assert 0.6 < cover < 0.95


def test_rows_without_target_are_dropped_in_fit():
    train = _frame(300)
    train.loc[train.index[:50], "y_h1"] = np.nan
    _fit(train)  # must not raise on NaN targets


def test_unseen_category_at_predict_is_handled():
    model = _fit(_frame(300))
    out = model.predict(pd.DataFrame({"age": [25.0], "position": ["Unknown"],
                                      "value_change_12m": [0.0]}))
    assert np.isfinite(out[["q10", "q50", "q90"]].to_numpy()).all()


def test_object_columns_with_missing_values_become_categories():
    train = _frame(300)
    train["position"] = train["position"].astype(object)
    train.loc[train.index[:20], "position"] = None
    out = _fit(train).predict(train)
    assert np.isfinite(out["q50"]).all()


def test_early_stopping_uses_val():
    train, val = _frame(400), _frame(150, seed=5)
    model = QuantileGBM(horizons=[1], quantiles=[0.5],
                        params={**FAST, "n_estimators": 2000, "early_stopping_rounds": 20}
                        ).fit(train, val, FEATURES)
    assert model.models_[(1, 0.5)].best_iteration_ < 2000


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        QuantileGBM(horizons=[1], quantiles=[0.5]).predict(_frame(5))


def _coverage(model, test):
    out = model.predict(test)
    return ((test["y_h1"] >= out["q10"]) & (test["y_h1"] <= out["q90"])).mean()


def test_calibrate_widens_a_too_narrow_band_to_about_80_percent():
    # Trained on calmer data than it meets later, the band comes out too narrow
    model = _fit(_frame(800, noise=0.1))
    test = _frame(2000, seed=7)
    assert _coverage(model, test) < 0.7
    model.calibrate(_frame(800, seed=8))
    assert 0.74 < _coverage(model, test) < 0.86


def test_calibrate_keeps_quantiles_ordered():
    model = _fit(_frame(400)).calibrate(_frame(200, seed=9))
    out = model.predict(_frame(300, seed=10))
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()
