import numpy as np
import pandas as pd

from pvf.evaluation.cv import cross_validate, summarize

FAST = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5,
        "verbose": -1}


def _samples(n_players: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        for origin in ["manual", "span"]:
            age = rng.uniform(18, 34)
            rows.append({"player_id": p, "origin": origin, "age": age,
                         "position": rng.choice(["Attack", "Defender"]),
                         "value_now": rng.uniform(1e6, 5e7),
                         "value_backfilled": origin == "manual" and p % 10 == 0,
                         "y_h1": 0.08 * (26 - age) + rng.normal(0, 0.3)})
    return pd.DataFrame(rows, index=pd.RangeIndex(1000, 1000 + 2 * n_players))


def _run(samples):
    return cross_validate(samples, {"gbm": ["age", "position"]}, quantiles=[0.1, 0.5, 0.9],
                          params=FAST, n_splits=4, seed=1, val_share=0.2, min_count=5)


def test_every_manual_row_predicted_once_per_model():
    s = _samples()
    oof = _run(s)
    manual = s.index[s["origin"] == "manual"]
    for model, g in oof.groupby("model"):
        assert sorted(g["row"]) == sorted(manual), model


def test_models_include_baselines_and_gbm():
    oof = _run(_samples())
    assert set(oof["model"]) == {"no_change", "age_position", "linear", "gbm"}
    gbm = oof[oof["model"] == "gbm"]
    assert gbm[["q10", "q90"]].notna().all().all()
    assert (oof.loc[oof["model"] == "no_change", "pred"] == 0).all()


def test_test_players_never_seen_in_training(monkeypatch):
    s = _samples()
    seen = []
    import pvf.evaluation.cv as cv
    real = cv.QuantileGBM.fit

    def spy(self, train, val, features):
        seen.append((set(train["player_id"]), set(val["player_id"]) if val is not None else set()))
        return real(self, train, val, features)

    monkeypatch.setattr(cv.QuantileGBM, "fit", spy)
    oof = _run(s)
    for (train_p, val_p), (_, g) in zip(seen, oof[oof["model"] == "gbm"].groupby("fold")):
        test_p = set(s.loc[g["row"], "player_id"])
        assert not (test_p & train_p) and not (test_p & val_p) and not (train_p & val_p)


def test_gbm_beats_no_change_on_learnable_signal():
    s = _samples(300)
    table = summarize(_run(s), s)
    all_rows = table[table["subset"] == "all"].set_index("model")
    assert all_rows.loc["gbm", "mae_log"] < all_rows.loc["no_change", "mae_log"]


def test_summary_has_subset_without_backfilled_rows():
    s = _samples()
    table = summarize(_run(s), s)
    assert set(table["subset"]) == {"all", "no_backfill"}
    n = table.set_index(["subset", "model"])["n"]
    manual = s[s["origin"] == "manual"]
    assert n["all", "gbm"] == len(manual)
    assert n["no_backfill", "gbm"] == (~manual["value_backfilled"]).sum()
    assert {"mae_log", "median_ape_eur", "coverage_80"} <= set(table.columns)


def test_band_is_calibrated_on_validation_players_only(monkeypatch):
    s = _samples()
    import pvf.evaluation.cv as cv
    seen, real_fit, real_cal = {}, cv.QuantileGBM.fit, cv.QuantileGBM.calibrate

    def fit_spy(self, train, val, features):
        seen["train"], seen["val"] = set(train["player_id"]), set(val["player_id"])
        return real_fit(self, train, val, features)

    def cal_spy(self, cal):
        seen.setdefault("cal_ok", []).append(set(cal["player_id"]) == seen["val"])
        return real_cal(self, cal)

    monkeypatch.setattr(cv.QuantileGBM, "fit", fit_spy)
    monkeypatch.setattr(cv.QuantileGBM, "calibrate", cal_spy)
    _run(s)
    assert seen["cal_ok"] and all(seen["cal_ok"])
