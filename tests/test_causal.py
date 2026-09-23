"""DML utilities: correctness against DoubleML (same folds) and against known truth."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from utils import causal as C
from utils.datasets import load_dataset, make_plr

RF = RandomForestRegressor(80, min_samples_leaf=5, random_state=0, n_jobs=1)


def test_folds_partition_rows():
    folds = C.fold_indices(103, 5, seed=1)
    te = np.sort(np.concatenate([t for _, t in folds]))
    assert np.array_equal(te, np.arange(103))


def test_plr_matches_doubleml_exactly():
    dml = pytest.importorskip("doubleml")
    df = make_plr(n=600, seed=4)
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    folds = C.fold_indices(len(y), 5, 11)
    r = C.plr_from_nuisance(y, d, C.cross_fit_predict(RF, X, y, folds), C.cross_fit_predict(RF, X, d, folds))
    data = dml.DoubleMLData(df, "y", "d", [c for c in df if c.startswith("x")])
    obj = dml.DoubleMLPLR(data, RF, RF, n_folds=5, draw_sample_splitting=False)
    obj.set_sample_splitting([[(tr, te) for tr, te in folds]])
    obj.fit()
    assert r.theta == pytest.approx(float(obj.coef[0]), abs=1e-10)
    assert r.se == pytest.approx(float(obj.se[0]), rel=1e-8)


def test_plr_linear_nuisance_recovers_theta():
    df = make_plr(n=4000, nonlinearity=0.0, seed=2)
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    r = C.dml_plr(X, y, d, LinearRegression(), LinearRegression(), n_folds=5)
    assert abs(r.theta - 0.5) < 4 * r.se


def test_irm_ate_close_to_truth():
    df = load_dataset("causal")
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    r = C.dml_irm(X, y, d, RF, RandomForestClassifier(80, min_samples_leaf=5, random_state=0, n_jobs=1), n_folds=3)
    assert abs(r.theta - df["tau"].mean()) < 4 * r.se


def test_pliv_removes_endogeneity_bias():
    X, y, d, z = C.make_pliv(3000, strength=1.0, seed=3)
    iv = C.dml_pliv(X, y, d, z, RF, RF, RF, n_folds=3)
    ols = C.naive_ols(X, y, d)
    assert abs(iv.theta - 0.5) < abs(ols.theta - 0.5)
    assert abs(iv.theta - 0.5) < 0.15


def test_iivm_late():
    X, y, d, z, _ = C.make_iivm(4000, late=1.0, seed=5)
    clf = RandomForestClassifier(80, min_samples_leaf=10, random_state=0, n_jobs=1)
    r = C.dml_iivm(X, y, d, z, RF, clf, clf, n_folds=3)
    assert abs(r.theta - 1.0) < 0.25


def test_gate_and_summary():
    phi = np.r_[np.ones(50), 3 * np.ones(50)]
    g = C.gate(phi, np.r_[np.zeros(50), np.ones(50)])
    assert [round(x["gate"], 6) for x in g] == [1.0, 3.0]
    arr = np.array([[0.5, 0.1], [0.6, 0.1], [0.4, 0.1]])
    s = C.summarize_mc(arr, 0.5)
    assert s["bias"] == pytest.approx(0.0) and s["coverage"] == 1.0


def test_naive_ols_linear():
    rng = np.random.default_rng(0)
    d = rng.normal(size=1000)
    y = 2 * d + rng.normal(size=1000)
    assert C.naive_ols(None, y, d).theta == pytest.approx(2.0, abs=0.1)
