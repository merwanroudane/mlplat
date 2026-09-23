"""Imbalanced classification: metrics, costs, probability corrections, samplers and CV evaluation.

Formulas are implemented directly and checked against scikit-learn / imbalanced-learn in
tests/test_breiman_imbalance.py. imbalanced-learn is optional: factories return ``None`` without it.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, fbeta_score,
                             log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from config import RANDOM_SEED


# ------------------------------------------------------------------ data
def imbalance_ratio(y) -> float:
    """IR = size of the largest class / size of the smallest class."""
    counts = np.bincount(np.asarray(y, int))
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())


def make_imbalanced_2d(n: int = 800, minority_frac: float = 0.05, separation: float = 2.5, n_clusters: int = 1,
                       seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Majority ~ N(0, I); the minority is split into ``n_clusters`` small Gaussian clusters ("small disjuncts")
    placed at distance ``separation`` from the origin. Smaller separation = more class overlap."""
    rng = np.random.default_rng(seed)
    n_min = max(int(round(n * minority_frac)), n_clusters * 2)
    X0 = rng.normal(size=(n - n_min, 2))
    angles = np.linspace(np.pi / 4, np.pi / 4 + 2 * np.pi, n_clusters, endpoint=False)
    sizes = np.full(n_clusters, n_min // n_clusters)
    sizes[: n_min % n_clusters] += 1
    X1 = np.vstack([rng.normal([separation * np.cos(a), separation * np.sin(a)], 0.6, size=(s, 2))
                    for a, s in zip(angles, sizes)])
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(len(X0), int), np.ones(len(X1), int)]
    return X, y


# ------------------------------------------------------------------ costs and thresholds
def cost_optimal_threshold(c_fp: float, c_fn: float, c_tp: float = 0.0, c_tn: float = 0.0) -> float:
    """Bayes-optimal threshold on a calibrated P(y=1|x) (Elkan, 2001): predict 1 iff p ≥ t*."""
    return float((c_fp - c_tn) / ((c_fp - c_tn) + (c_fn - c_tp)))


def expected_cost(y, p, thresholds, c_fp: float, c_fn: float) -> np.ndarray:
    """Average misclassification cost per case at each threshold."""
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for t in np.atleast_1d(thresholds):
        pred = p >= t
        out.append((c_fp * np.sum(pred & (y == 0)) + c_fn * np.sum(~pred & (y == 1))) / len(y))
    return np.asarray(out, float)


def balanced_class_weight(y) -> dict[int, float]:
    """Same rule as class_weight='balanced': w_c = n / (K · n_c)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): float(len(y) / (len(classes) * n)) for c, n in zip(classes, counts)}


def focal_loss(y, p, gamma: float = 2.0, alpha: float | None = None) -> float:
    """Mean focal loss (Lin et al., 2017): −α_t (1 − p_t)^γ log p_t. γ = 0 and α = None gives log loss."""
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    pt = np.where(y == 1, p, 1 - p)
    at = 1.0 if alpha is None else np.where(y == 1, alpha, 1 - alpha)
    return float(np.mean(-at * (1 - pt) ** gamma * np.log(pt)))


# ------------------------------------------------------------------ probabilities
def prior_shift(p_s, pi: float, pi_s: float) -> np.ndarray:
    """Map probabilities learned under prevalence ``pi_s`` to the deployment prevalence ``pi`` (Saerens et al., 2002)."""
    p_s = np.asarray(p_s, float)
    a = p_s * pi / pi_s
    b = (1 - p_s) * (1 - pi) / (1 - pi_s)
    return a / (a + b)


def saerens_em(p, pi_train: float, max_iter: int = 200, tol: float = 1e-8) -> tuple[float, np.ndarray, int]:
    """Estimate an unknown new prevalence from unlabeled scores by EM (Saerens, Latinne & Decaestecker, 2002).

    Returns (estimated prevalence, adjusted probabilities, iterations)."""
    p = np.asarray(p, float)
    pi = pi_train
    for it in range(1, max_iter + 1):
        adj = prior_shift(p, pi, pi_train)
        new = float(adj.mean())
        if abs(new - pi) < tol:
            return new, adj, it
        pi = new
    return pi, prior_shift(p, pi, pi_train), max_iter


# ------------------------------------------------------------------ metrics
def g_mean(y, pred) -> float:
    """Geometric mean of sensitivity (recall) and specificity."""
    y, pred = np.asarray(y), np.asarray(pred)
    sens = np.mean(pred[y == 1] == 1) if np.any(y == 1) else 0.0
    spec = np.mean(pred[y == 0] == 0) if np.any(y == 0) else 0.0
    return float(np.sqrt(sens * spec))


def imbalance_metrics(y, p, pred=None, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free (ranking, probability) and threshold-dependent metrics for a binary problem."""
    y, p = np.asarray(y), np.asarray(p, float)
    pred = (p >= threshold).astype(int) if pred is None else np.asarray(pred)
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC (AP)": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": float(np.mean(pred[y == 0] == 0)),
        "F1": f1_score(y, pred, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "balanced acc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "G-mean": g_mean(y, pred),
        "Brier": brier_score_loss(y, p),
        "log loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]),
        "mean p": float(p.mean()),
    }


def bootstrap_ci(y, p, metric, n_boot: int = 300, level: float = 0.95, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Percentile interval from a stratified bootstrap (positives and negatives resampled separately)."""
    y, p = np.asarray(y), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(metric(y[idx], p[idx]))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# ------------------------------------------------------------------ imbalanced-learn factories
# name -> (module, class, family, fixed kwargs). Every class and default was checked against imbalanced-learn 0.14.2.
SAMPLERS: dict[str, tuple[str, str, str, dict]] = {
    "RandomOverSampler": ("imblearn.over_sampling", "RandomOverSampler", "over", {}),
    "ROS + shrinkage (smoothed bootstrap)": ("imblearn.over_sampling", "RandomOverSampler", "over", {"shrinkage": 1.0}),
    "SMOTE": ("imblearn.over_sampling", "SMOTE", "over", {}),
    "BorderlineSMOTE-1": ("imblearn.over_sampling", "BorderlineSMOTE", "over", {"kind": "borderline-1"}),
    "BorderlineSMOTE-2": ("imblearn.over_sampling", "BorderlineSMOTE", "over", {"kind": "borderline-2"}),
    "SVMSMOTE": ("imblearn.over_sampling", "SVMSMOTE", "over", {}),
    "ADASYN": ("imblearn.over_sampling", "ADASYN", "over", {}),
    "RandomUnderSampler": ("imblearn.under_sampling", "RandomUnderSampler", "under", {}),
    "NearMiss-1": ("imblearn.under_sampling", "NearMiss", "under", {"version": 1}),
    "NearMiss-2": ("imblearn.under_sampling", "NearMiss", "under", {"version": 2}),
    "NearMiss-3": ("imblearn.under_sampling", "NearMiss", "under", {"version": 3}),
    "ClusterCentroids": ("imblearn.under_sampling", "ClusterCentroids", "under", {}),
    "TomekLinks": ("imblearn.under_sampling", "TomekLinks", "clean", {}),
    "EditedNearestNeighbours": ("imblearn.under_sampling", "EditedNearestNeighbours", "clean", {}),
    "RepeatedEditedNearestNeighbours": ("imblearn.under_sampling", "RepeatedEditedNearestNeighbours", "clean", {}),
    "AllKNN": ("imblearn.under_sampling", "AllKNN", "clean", {}),
    "CondensedNearestNeighbour": ("imblearn.under_sampling", "CondensedNearestNeighbour", "clean", {}),
    "OneSidedSelection": ("imblearn.under_sampling", "OneSidedSelection", "clean", {}),
    "NeighbourhoodCleaningRule": ("imblearn.under_sampling", "NeighbourhoodCleaningRule", "clean", {}),
    "InstanceHardnessThreshold": ("imblearn.under_sampling", "InstanceHardnessThreshold", "clean", {}),
    "SMOTEENN": ("imblearn.combine", "SMOTEENN", "hybrid", {}),
    "SMOTETomek": ("imblearn.combine", "SMOTETomek", "hybrid", {}),
}

_RANDOM = {"RandomOverSampler", "SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "RandomUnderSampler",
           "ClusterCentroids", "CondensedNearestNeighbour", "OneSidedSelection", "InstanceHardnessThreshold",
           "SMOTEENN", "SMOTETomek"}
_K_PARAM = {"SMOTE": "k_neighbors", "BorderlineSMOTE": "k_neighbors", "SVMSMOTE": "k_neighbors", "ADASYN": "n_neighbors",
            "NearMiss": "n_neighbors", "EditedNearestNeighbours": "n_neighbors",
            "RepeatedEditedNearestNeighbours": "n_neighbors", "AllKNN": "n_neighbors"}

ENSEMBLES: dict[str, str] = {
    "BalancedRandomForestClassifier": "Balanced Random Forest (Chen, Liaw & Breiman, 2004)",
    "BalancedBaggingClassifier": "Balanced Bagging (under-sampled bags)",
    "EasyEnsembleClassifier": "EasyEnsemble (Liu, Wu & Zhou, 2009)",
    "RUSBoostClassifier": "RUSBoost (Seiffert et al., 2010)",
}


def sampler_family(name: str) -> str:
    return SAMPLERS[name][2]


def make_sampler(name: str, seed: int = RANDOM_SEED, k: int | None = None, sampling_strategy="auto"):
    """Instantiate a sampler by display name, or return None if imbalanced-learn is not installed."""
    mod, cls_name, _, fixed = SAMPLERS[name]
    try:
        cls = getattr(importlib.import_module(mod), cls_name)
    except ImportError:
        return None
    kw = dict(fixed, sampling_strategy=sampling_strategy)
    if cls_name in _RANDOM:
        kw["random_state"] = seed
    if k is not None and cls_name in _K_PARAM:
        kw[_K_PARAM[cls_name]] = k
    return cls(**kw)


def make_balanced_ensemble(name: str, n_estimators: int = 50, seed: int = RANDOM_SEED):
    """Instantiate an imbalanced-learn ensemble (n_jobs=1), or return None if the package is missing."""
    try:
        ens = importlib.import_module("imblearn.ensemble")
    except ImportError:
        return None
    kw = {"n_estimators": n_estimators, "random_state": seed}
    if name == "RUSBoostClassifier":
        # The default base learner is a depth-1 stump and learning_rate=1.0. On overlapping classes a weak learner
        # fitted to a random under-sample can be "worse than random", and the discrete boosting loop then raises
        # (observed on the "imbalanced" dataset with depth 1 and 2). Depth-3 trees with shrinkage 0.1 were stable.
        from sklearn.tree import DecisionTreeClassifier
        kw["estimator"] = DecisionTreeClassifier(max_depth=3, random_state=seed)
        kw["learning_rate"] = 0.1
    else:
        kw["n_jobs"] = 1
    return getattr(ens, name)(**kw)


def resample_roles(sampler, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit-resample and describe what happened to every row.

    Returns (X_res, y_res, roles, removed): ``roles`` labels each output row 'original', 'duplicate' (a repeated
    original) or 'synthetic' (not an input row: SMOTE-type interpolation, smoothed bootstrap, centroids);
    ``removed`` flags input rows that are absent from the output. Rows are matched exactly, which works for every
    sampler family, including hybrids that expose no ``sample_indices_``."""
    Xr, yr = sampler.fit_resample(X, y)
    lookup: dict[tuple, int] = {}
    for i, row in enumerate(np.asarray(X)):
        lookup.setdefault(tuple(row), i)
    seen = np.zeros(len(y), bool)
    roles = []
    for row in np.asarray(Xr):
        i = lookup.get(tuple(row))
        if i is None:
            roles.append("synthetic")
        elif seen[i]:
            roles.append("duplicate")
        else:
            seen[i] = True
            roles.append("original")
    return Xr, yr, np.asarray(roles), ~seen


# ------------------------------------------------------------------ evaluation protocol
def cv_evaluate(model, X, y, n_splits: int = 5, n_repeats: int = 1, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Repeated stratified CV; resampling (if any) lives inside ``model`` so it only touches training folds.

    Validation folds keep the real prevalence. Decisions use ``model.predict`` (so a tuned threshold is honoured)."""
    from sklearn.base import clone

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for f, (tr, te) in enumerate(cv.split(X, y)):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows.append({"fold": f + 1, **imbalance_metrics(y[te], p, pred=m.predict(X[te]))})
    return pd.DataFrame(rows)
