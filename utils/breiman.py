"""Computations behind the Leo Breiman modules: the two cultures, the Rashomon effect,
Occam's dilemma, Bellman's blessing and the instability that motivates bagging.

Pure NumPy / scikit-learn (no Streamlit), unit-tested in tests/test_breiman_imbalance.py.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_SEED


# ------------------------------------------------------------------ two cultures
def two_cultures_data(n: int = 1500, nonlinearity: float = 1.0, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Binary outcome whose log-odds contain an interaction and a quadratic term.

    ``nonlinearity = 0`` makes the additive logistic model (the "data model") exactly true.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    eta = 0.8 * X[:, 0] - 0.6 * X[:, 1] + nonlinearity * (1.5 * X[:, 2] * X[:, 3] + 1.2 * (X[:, 4] ** 2 - 1))
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(5)]), pd.Series(y, name="y")


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> tuple[float, float]:
    """Hosmer–Lemeshow goodness-of-fit statistic and p-value (chi-square with groups − 2 df).

    Observations are sorted by predicted probability and cut into ``groups`` bins of (almost) equal size.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    stat = 0.0
    for idx in np.array_split(order, groups):
        n_g, obs, exp_ = len(idx), y[idx].sum(), p[idx].sum()
        pbar = exp_ / n_g
        denom = n_g * pbar * (1 - pbar)
        if denom > 0:
            stat += (obs - exp_) ** 2 / denom
    return float(stat), float(stats.chi2.sf(stat, groups - 2))


# ------------------------------------------------------------------ Rashomon effect
def rashomon_data(n: int = 200, p: int = 12, rho: float = 0.85, n_true: int = 6, noise: float = 1.5,
                  seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Regression with equicorrelated predictors; the truth gives weight 0.5 to x1…x{n_true}
    (Breiman's 2001 subset-selection setting in miniature)."""
    rng = np.random.default_rng(seed)
    cov = np.full((p, p), rho) + (1 - rho) * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.zeros(p)
    beta[:n_true] = 0.5
    y = X @ beta + rng.normal(scale=noise, size=n)
    return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(p)]), pd.Series(y, name="y")


def rashomon_subsets(X: pd.DataFrame, y: pd.Series, k: int = 3, tol: float = 0.01) -> pd.DataFrame:
    """All k-variable OLS models whose residual sum of squares is within ``tol`` (relative) of the best one.

    Breiman (2001, section 8): many different subsets give almost the same fit — which one "explains" y?
    """
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    tss = float(((yv - yv.mean()) ** 2).sum())
    rows = []
    for cols in itertools.combinations(range(Xv.shape[1]), k):
        A = np.c_[np.ones(len(yv)), Xv[:, cols]]
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        rss = float(((yv - A @ coef) ** 2).sum())
        rows.append({"variables": ", ".join(X.columns[list(cols)]), "RSS": rss, "R2": 1 - rss / tss})
    out = pd.DataFrame(rows).sort_values("RSS", ignore_index=True)
    best = out["RSS"].iloc[0]
    out["gap %"] = 100 * (out["RSS"] / best - 1)
    return out[out["RSS"] <= best * (1 + tol)].reset_index(drop=True)


def tree_instability(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, n_boot: int = 40,
                     max_depth: int = 3, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Refit a shallow tree on bootstrap samples; record the root split variable and test accuracy of each tree."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        t = DecisionTreeClassifier(max_depth=max_depth, random_state=b).fit(X.iloc[idx], y.iloc[idx])
        root = X.columns[t.tree_.feature[0]] if t.tree_.feature[0] >= 0 else "(leaf)"
        rows.append({"bootstrap": b + 1, "root variable": root, "root threshold": float(t.tree_.threshold[0]),
                     "test accuracy": float(t.score(X_test, y_test))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Bellman: dimensionality as a blessing
def random_feature_curve(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                         dims: tuple[int, ...] = (0, 2, 5, 10, 25, 50, 100, 300), gamma: float = 1.0,
                         seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Test accuracy of a *linear* classifier as random Fourier features are added (0 = raw features only)."""
    rows = []
    for d in dims:
        if d == 0:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        else:
            model = make_pipeline(StandardScaler(), RBFSampler(gamma=gamma, n_components=d, random_state=seed),
                                  LogisticRegression(max_iter=5000, C=10.0))
        model.fit(X_tr, y_tr)
        rows.append({"features": d if d else X_tr.shape[1], "label": "raw" if d == 0 else f"{d} RFF",
                     "train accuracy": model.score(X_tr, y_tr), "test accuracy": model.score(X_te, y_te)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bagging: unstable vs stable procedures
def instability(estimator, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, n_models: int = 25,
                seed: int = RANDOM_SEED) -> float:
    """Mean disagreement rate between models refit on bootstrap samples: 0 = perfectly stable procedure."""
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_models):
        idx = rng.integers(0, len(y_tr), len(y_tr))
        preds.append(clone(estimator).fit(X_tr[idx], y_tr[idx]).predict(X_te))
    P = np.asarray(preds, float)
    q = P.mean(axis=0)
    return float(np.mean(2 * q * (1 - q)))  # probability that two random refits disagree at a test point
