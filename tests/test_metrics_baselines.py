import numpy as np
import pandas as pd

from pvf.evaluation.metrics import interval_coverage, mae_log, median_ape_eur
from pvf.models.baselines import AgePositionCurve, NoChange


def test_metrics_basic():
    assert mae_log([0, 1], [0, 0]) == 0.5
    assert median_ape_eur(np.array([1.0]), np.array([0.0]), np.array([0.0])) == 0.0
    assert interval_coverage([0, 5], np.array([-1, -1]), np.array([1, 1])) == 0.5


def test_baselines_shapes():
    X = pd.DataFrame({"age": [20] * 30 + [32] * 30, "position": ["Attack"] * 60})
    y = pd.Series([0.3] * 30 + [-0.2] * 30)
    assert (NoChange().fit(X, y).predict(X) == 0).all()
    pred = AgePositionCurve().fit(X, y).predict(pd.DataFrame({"age": [20, 32, 45], "position": ["Attack", "Attack", "Goalkeeper"]}))
    assert np.allclose(pred[:2], [0.3, -0.2])
    assert np.isfinite(pred[2])  # unseen cell falls back to global median
