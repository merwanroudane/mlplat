"""utils/breiman.py and utils/imbalance.py: formulas vs libraries and known properties."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from utils import breiman as B
from utils import imbalance as I


# ------------------------------------------------------------------ Breiman
def test_hosmer_lemeshow_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = rng.binomial(1, p)
    _, pv_good = B.hosmer_lemeshow(y, p)
    _, pv_bad = B.hosmer_lemeshow(y, np.clip(p * 0.6, 0, 1))
    assert pv_good > 0.01 and pv_bad < 1e-6


def test_two_cultures_data_linear_when_no_nonlinearity():
    X, y = B.two_cultures_data(n=500, nonlinearity=0.0, seed=1)
    assert X.shape == (500, 5) and set(np.unique(y)) <= {0, 1}


def test_rashomon_subsets_within_tolerance_and_sorted():
    X, y = B.rashomon_data(seed=3)
    r = B.rashomon_subsets(X, y, k=3, tol=0.02)
    assert len(r) >= 1
    assert (r["gap %"] <= 2.0 + 1e-9).all()
    assert r["RSS"].is_monotonic_increasing
    # the best subset reproduces a direct OLS fit
    cols = r.loc[0, "variables"].split(", ")
    A = np.c_[np.ones(len(y)), X[cols].to_numpy()]
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    assert np.isclose(((y.to_numpy() - A @ coef) ** 2).sum(), r.loc[0, "RSS"])


def test_trees_are_less_stable_than_logistic():
    X, y = B.two_cultures_data(n=600, nonlinearity=1.0, seed=2)
    Xv, yv = X.to_numpy(), y.to_numpy()
    tree = B.instability(DecisionTreeClassifier(random_state=0), Xv[:400], yv[:400], Xv[400:], n_models=15)
    logit = B.instability(LogisticRegression(max_iter=2000), Xv[:400], yv[:400], Xv[400:], n_models=15)
    assert tree > 2 * logit


def test_random_features_help_a_linear_model_on_circles():
    from utils.datasets import toy_2d
    X, y = toy_2d("circles", n=400, noise=0.1, seed=0)
    r = B.random_feature_curve(X[:300], y[:300], X[300:], y[300:], dims=(0, 200))
    assert r["test accuracy"].iloc[1] > r["test accuracy"].iloc[0] + 0.2


def test_tree_instability_frame():
    X, y = B.two_cultures_data(n=400, seed=5)
    r = B.tree_instability(X[:300], y[:300], X[300:], y[300:], n_boot=10)
    assert len(r) == 10 and set(r["root variable"]) <= set(X.columns)


# ------------------------------------------------------------------ imbalance: formulas
def test_cost_threshold_elkan():
    assert I.cost_optimal_threshold(1, 9) == pytest.approx(0.1)
    assert I.cost_optimal_threshold(1, 1) == pytest.approx(0.5)


def test_expected_cost_minimised_near_theory_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 5, 200_000)
    y = rng.binomial(1, p)
    grid = np.linspace(0.01, 0.99, 99)
    c = I.expected_cost(y, p, grid, c_fp=1, c_fn=9)
    assert abs(grid[c.argmin()] - 0.1) <= 0.02


def test_balanced_weight_matches_sklearn():
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    w = I.balanced_class_weight(y)
    ref = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    assert np.allclose([w[0], w[1]], ref)


def test_focal_reduces_to_log_loss():
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 500)
    y = rng.binomial(1, p)
    assert I.focal_loss(y, p, gamma=0) == pytest.approx(log_loss(y, p))
    assert I.focal_loss(y, p, gamma=2) < I.focal_loss(y, p, gamma=0)


def test_prior_shift_roundtrip_and_identity():
    p = np.linspace(0.01, 0.99, 50)
    assert np.allclose(I.prior_shift(p, 0.3, 0.3), p)
    assert np.allclose(I.prior_shift(I.prior_shift(p, 0.05, 0.5), 0.5, 0.05), p)


def test_saerens_em_recovers_prevalence():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.binomial(1, 0.1, n)
    x = rng.normal(np.where(y == 1, 1.5, 0.0), 1.0)
    # exact posterior under a training prevalence of 0.5
    lr = np.exp(1.5 * x - 1.125)
    p_train = lr / (1 + lr)
    pi_hat, adj, _ = I.saerens_em(p_train, pi_train=0.5)
    assert abs(pi_hat - 0.1) < 0.01
    assert abs(adj.mean() - y.mean()) < 0.01


def test_g_mean_and_metrics_panel():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 0])
    assert I.g_mean(y, pred) == pytest.approx(np.sqrt(0.5 * 0.75))
    m = I.imbalance_metrics(y, np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.4]))
    assert m["specificity"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 2000)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, 2000), 0, 1)
    lo, hi = I.bootstrap_ci(y, p, average_precision_score, n_boot=100)
    assert lo <= average_precision_score(y, p) <= hi


def test_imbalance_ratio_and_toy():
    X, y = I.make_imbalanced_2d(n=1000, minority_frac=0.05, n_clusters=3)
    assert X.shape == (1000, 2) and y.sum() == 50
    assert I.imbalance_ratio(y) == pytest.approx(950 / 50)


# ------------------------------------------------------------------ imbalance: imbalanced-learn
@pytest.mark.parametrize("name", list(I.SAMPLERS))
def test_every_sampler_runs_and_roles_are_consistent(name):
    pytest.importorskip("imblearn")
    X, y = I.make_imbalanced_2d(n=300, minority_frac=0.1, separation=2.0)
    s = I.make_sampler(name, seed=0, k=3)
    Xr, yr, roles, removed = I.resample_roles(s, X, y)
    assert len(Xr) == len(yr) == len(roles)
    fam = I.sampler_family(name)
    if fam == "over":
        assert not removed.any() and (roles != "original").sum() == len(yr) - len(y)
    if fam in ("under", "clean"):
        assert yr.sum() == y.sum()  # minority untouched
        assert (roles == "original").sum() == len(y) - removed.sum() or name == "ClusterCentroids"


@pytest.mark.parametrize("name", list(I.ENSEMBLES))
def test_balanced_ensembles_fit(name):
    pytest.importorskip("imblearn")
    X, y = I.make_imbalanced_2d(n=300, minority_frac=0.1, separation=2.0)
    m = I.make_balanced_ensemble(name, n_estimators=5).fit(X, y)
    assert m.predict_proba(X).shape == (300, 2)


def test_cv_evaluate_keeps_real_prevalence_in_validation():
    pytest.importorskip("imblearn")
    from imblearn.pipeline import make_pipeline
    X, y = I.make_imbalanced_2d(n=400, minority_frac=0.1, separation=2.0)
    res = I.cv_evaluate(make_pipeline(I.make_sampler("RandomUnderSampler"), KNeighborsClassifier()), X, y, n_splits=4)
    assert len(res) == 4 and res["ROC-AUC"].between(0, 1).all()
