"""From-scratch educational models agree with the library versions."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from utils.datasets import toy_2d
from utils.models import PLAYGROUND, KNNScratch, LinearRegressionGD, LogisticRegressionGD, PCAScratch, build_playground, kmeans_history


def test_linear_gd_matches_ols():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ [1.0, -2.0, 0.5] + 3 + rng.normal(scale=0.1, size=300)
    gd = LinearRegressionGD(lr=0.1, n_iter=3000).fit(X, y)
    ols = LinearRegression().fit(X, y)
    assert np.allclose(gd.coef_, ols.coef_, atol=1e-3) and abs(gd.intercept_ - ols.intercept_) < 1e-3


def test_logistic_gd_matches_sklearn_objective():
    X, y = toy_2d("linear", n=300, noise=0.3)
    X = (X - X.mean(0)) / X.std(0)
    gd = LogisticRegressionGD(C=1.0, lr=1.0, n_iter=20000).fit(X, y)
    sk = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(gd.coef_, sk.coef_[0], atol=5e-3)


def test_knn_scratch_agrees():
    X, y = toy_2d("moons", n=200)
    grid = np.random.default_rng(1).uniform(-1, 2, (300, 2))
    a = KNNScratch(7).fit(X, y).predict(grid)
    b = KNeighborsClassifier(7).fit(X, y).predict(grid)
    assert (a == b).mean() > 0.99


def test_kmeans_history_monotone_and_close_to_sklearn():
    X, _ = toy_2d("blobs", n=300)
    h = kmeans_history(X, 3, n_iter=50, seed=0, init="k-means++")
    inert = [s["inertia"] for s in h if s["inertia"] is not None]
    assert all(b <= a + 1e-9 for a, b in zip(inert, inert[1:]))
    assert h[-1]["inertia"] <= KMeans(3, n_init=10, random_state=0).fit(X).inertia_ * 1.05


def test_pca_scratch_matches():
    X = np.random.default_rng(0).normal(size=(200, 5)) @ np.random.default_rng(1).normal(size=(5, 5))
    a, b = PCAScratch(3).fit(X), PCA(3).fit(X)
    assert np.allclose(a.explained_variance_, b.explained_variance_)
    assert np.allclose(np.abs(np.sum(a.components_ * b.components_, axis=1)), 1.0)


def test_playground_models_fit_with_defaults():
    X, y = toy_2d("moons", n=120)
    for key, spec in PLAYGROUND.items():
        params = {p.name: p.default for p in spec.params}
        assert build_playground(key, params).fit(X, y).score(X, y) > 0.5, key
