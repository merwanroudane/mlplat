import numpy as np
from sklearn.linear_model import Lasso

from utils.optimization import (SURFACES, gradient_descent, lasso_coordinate_descent, learning_rate_limit, make_linear_data, newton,
                                ols_closed_form, sgd_linear, soft_threshold)


def test_gd_converges_below_limit_and_diverges_above():
    s = SURFACES["quadratic"]()
    lim = learning_rate_limit(s.hess(np.zeros(2)))
    ok = gradient_descent(s.grad, np.array([3.0, 2.0]), 0.9 * lim, 300)
    bad = gradient_descent(s.grad, np.array([3.0, 2.0]), 1.1 * lim, 300)
    assert np.linalg.norm(ok[-1]) < 1e-3
    assert np.linalg.norm(bad[-1]) > 1e2


def test_newton_one_step_on_quadratic():
    s = SURFACES["quadratic"]()
    path = newton(s.grad, s.hess, np.array([3.0, -2.0]), 1)
    assert np.allclose(path[-1], 0.0)


def test_sgd_approaches_ols():
    X, y = make_linear_data(400, noise=0.3, seed=1)
    beta = ols_closed_form(X, y)
    path, _ = sgd_linear(X, y, np.zeros(2), 0.05, 40, batch_size=32, seed=0, decay=0.01)
    assert np.linalg.norm(path[-1] - beta) < 0.05


def test_soft_threshold():
    assert np.allclose(soft_threshold(np.array([-2.0, 0.3, 1.5]), 0.5), [-1.5, 0.0, 1.0])


def test_lasso_cd_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([3, -2, 0, 0, 1, 0, 0, 0.5]) + rng.normal(size=200)
    y = y - y.mean()
    w, _ = lasso_coordinate_descent(X, y, 0.2, n_sweeps=500, tol=1e-12)
    sk = Lasso(alpha=0.2, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
    assert np.allclose(w, sk, atol=1e-6)
