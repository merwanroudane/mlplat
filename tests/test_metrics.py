"""Our formula implementations must match scikit-learn / SciPy."""

import numpy as np
import pytest
from scipy import stats
from sklearn import metrics as skm

from utils import metrics as M

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 500)
p = np.clip(0.3 * y + rng.random(500) * 0.7, 0.001, 0.999)
yr = rng.normal(size=300) + 5
pr = yr + rng.normal(scale=0.5, size=300)


def test_regression_metrics():
    assert M.mse(yr, pr) == pytest.approx(skm.mean_squared_error(yr, pr))
    assert M.rmse(yr, pr) == pytest.approx(skm.root_mean_squared_error(yr, pr))
    assert M.mae(yr, pr) == pytest.approx(skm.mean_absolute_error(yr, pr))
    assert M.r2(yr, pr) == pytest.approx(skm.r2_score(yr, pr))
    assert M.mape(yr, pr) == pytest.approx(skm.mean_absolute_percentage_error(yr, pr))
    assert M.pinball(yr, pr, 0.9) == pytest.approx(skm.mean_pinball_loss(yr, pr, alpha=0.9))
    cnt = rng.poisson(3, 200)
    mu = np.full(200, 3.2)
    assert M.mean_poisson_deviance(cnt, mu) == pytest.approx(skm.mean_poisson_deviance(cnt, mu))


def test_classification_metrics():
    yhat = (p >= 0.5).astype(int)
    rep = M.classification_report_dict(y, yhat)
    assert rep["precision"] == pytest.approx(skm.precision_score(y, yhat))
    assert rep["recall"] == pytest.approx(skm.recall_score(y, yhat))
    assert rep["f1"] == pytest.approx(skm.f1_score(y, yhat))
    assert rep["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, yhat))
    assert rep["mcc"] == pytest.approx(skm.matthews_corrcoef(y, yhat))
    assert M.log_loss(y, p) == pytest.approx(skm.log_loss(y, p))
    assert M.brier(y, p) == pytest.approx(skm.brier_score_loss(y, p))
    fpr, tpr, _ = M.roc_points(y, p)
    assert M.auc_trapezoid(fpr, tpr) == pytest.approx(skm.roc_auc_score(y, p))


def test_drift_statistics():
    a, b = rng.normal(size=1000), rng.normal(0.3, 1, 800)
    assert M.ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic)
    assert M.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert M.psi(a, b) > 0.02


def test_losses_shapes():
    r = np.linspace(-3, 3, 7)
    assert np.all(M.loss_huber(r, 1.0) <= M.loss_squared(r) / 2 + 1e-12)
    assert M.loss_hinge(np.array([2.0]))[0] == 0.0
    assert M.loss_logistic(np.array([0.0]))[0] == pytest.approx(1.0)
