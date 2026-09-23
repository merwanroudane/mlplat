"""Model-inspection helpers implemented from definitions (section 37)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import get_scorer


def permutation_importance_manual(model, X: np.ndarray, y: np.ndarray, scoring: str = "r2", n_repeats: int = 5,
                                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Drop in score when one column is shuffled (Breiman, 2001). Returns (mean, std) per feature."""
    rng = np.random.default_rng(seed)
    scorer = get_scorer(scoring)
    base = scorer(model, X, y)
    drops = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j, r] = base - scorer(model, Xp, y)
    return drops.mean(axis=1), drops.std(axis=1)


def ice_curves(model, X: np.ndarray, feature: int, grid: np.ndarray, n_lines: int = 60, seed: int = 0) -> np.ndarray:
    """Individual conditional expectation: prediction for each sampled row as feature j sweeps the grid."""
    rng = np.random.default_rng(seed)
    rows = X[rng.choice(len(X), min(n_lines, len(X)), replace=False)]
    out = np.empty((len(rows), len(grid)))
    for k, v in enumerate(grid):
        Xg = rows.copy()
        Xg[:, feature] = v
        out[:, k] = model.predict(Xg)
    return out


def shapley_exact(f, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Exact interventional Shapley values for a small number of features (≤ 8) by enumerating coalitions.

    v(S) = E_b[f(x_S, b_{S̄})] with b drawn from ``background``.
    """
    from itertools import combinations
    from math import factorial

    p = len(x)
    if p > 8:
        raise ValueError("exact enumeration is limited to 8 features")

    def value(S: tuple[int, ...]) -> float:
        Z = background.copy()
        if S:
            Z[:, list(S)] = x[list(S)]
        return float(np.mean(f(Z)))

    cache: dict[tuple[int, ...], float] = {}
    phi = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            w = factorial(size) * factorial(p - size - 1) / factorial(p)
            for S in combinations(others, size):
                S_with = tuple(sorted(S + (j,)))
                if S not in cache:
                    cache[S] = value(S)
                if S_with not in cache:
                    cache[S_with] = value(S_with)
                phi[j] += w * (cache[S_with] - cache[S])
    return phi
