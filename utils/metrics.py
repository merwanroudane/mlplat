"""Metrics and losses implemented from their formulas (sections 15 and 32).

Tests (tests/test_metrics.py) check every function against scikit-learn so
learners can trust that the formula on screen is the one being computed.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------- regression loss
def mse(y: Array, p: Array) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def rmse(y: Array, p: Array) -> float:
    return float(np.sqrt(mse(y, p)))


def mae(y: Array, p: Array) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y: Array, p: Array) -> float:
    y = np.asarray(y, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def adjusted_r2(y: Array, p: Array, n_features: int) -> float:
    n = len(y)
    return float(1 - (1 - r2(y, p)) * (n - 1) / (n - n_features - 1))


def mape(y: Array, p: Array) -> float:
    """Mean absolute percentage error as a fraction (like scikit-learn): undefined near y = 0."""
    y = np.asarray(y, float)
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), np.finfo(float).eps))))


def rmsle(y: Array, p: Array) -> float:
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


# pointwise losses of the residual r = y − ŷ (used by the Loss Function Lab)
def loss_squared(r: Array) -> Array:
    return np.asarray(r) ** 2


def loss_absolute(r: Array) -> Array:
    return np.abs(r)


def loss_huber(r: Array, delta: float = 1.0) -> Array:
    r = np.abs(r)
    return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))


def loss_quantile(r: Array, tau: float = 0.5) -> Array:
    """Pinball loss ρ_τ(r) = max(τ r, (τ − 1) r) with r = y − ŷ."""
    r = np.asarray(r)
    return np.maximum(tau * r, (tau - 1) * r)


def poisson_deviance_point(y: Array, mu: Array) -> Array:
    """Unit Poisson deviance 2(y log(y/μ) − y + μ), with y log y := 0 at y = 0."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - y + mu)


def mean_poisson_deviance(y: Array, mu: Array) -> float:
    return float(np.mean(poisson_deviance_point(y, mu)))


def pinball(y: Array, p: Array, tau: float) -> float:
    return float(np.mean(loss_quantile(np.asarray(y) - np.asarray(p), tau)))


# --------------------------------------------------- classification losses
# margin-based losses of m = y·f(x) with y ∈ {−1, +1}
def loss_zero_one(m: Array) -> Array:
    return (np.asarray(m) <= 0).astype(float)


def loss_hinge(m: Array) -> Array:
    return np.maximum(0, 1 - np.asarray(m))


def loss_logistic(m: Array) -> Array:
    """log(1 + e^{−m}) / log 2 — scaled so it passes through (0, 1) like the 0-1 loss."""
    return np.logaddexp(0, -np.asarray(m)) / np.log(2)


def loss_exponential(m: Array) -> Array:
    return np.exp(-np.asarray(m))


def log_loss(y: Array, p: Array, eps: float = 1e-15) -> float:
    """Binary cross-entropy, y ∈ {0, 1}, p = P(y = 1)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: Array, p: Array) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ------------------------------------------------------- confusion metrics
def confusion(y: Array, yhat: Array) -> tuple[int, int, int, int]:
    """(TN, FP, FN, TP) for binary labels 0/1."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    return tn, fp, fn, tp


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_report_dict(y: Array, yhat: Array, beta: float = 1.0) -> dict[str, float]:
    tn, fp, fn, tp = confusion(y, yhat)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    fbeta = _div((1 + beta ** 2) * precision * recall, beta ** 2 * precision + recall)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _div(tp * tn - fp * fn, denom)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": _div(2 * precision * recall, precision + recall), f"f{beta:g}": fbeta, "mcc": mcc,
    }


def expected_cost(y: Array, yhat: Array, cost_fp: float, cost_fn: float, gain_tp: float = 0.0) -> float:
    """Average business cost per case (negative = gain)."""
    tn, fp, fn, tp = confusion(y, yhat)
    return float((cost_fp * fp + cost_fn * fn - gain_tp * tp) / max(len(y), 1))


def roc_points(y: Array, score: Array) -> tuple[Array, Array, Array]:
    """ROC curve by sweeping every distinct threshold (from the definition)."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    s, ys = np.asarray(score)[order], y[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(ys)[distinct]
    fps = (distinct + 1) - tps
    P, N = ys.sum(), len(ys) - ys.sum()
    tpr = np.r_[0, tps / max(P, 1)]
    fpr = np.r_[0, fps / max(N, 1)]
    return fpr, tpr, np.r_[np.inf, s[distinct]]


def auc_trapezoid(x: Array, y: Array) -> float:
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def ks_statistic(a: Array, b: Array) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max ECDF gap)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.r_[a, b]
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


def psi(expected: Array, actual: Array, bins: int = 10) -> float:
    """Population Stability Index with quantile bins of the reference sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
