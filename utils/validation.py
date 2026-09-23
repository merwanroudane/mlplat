"""Environment introspection and input guards (sections 81–84)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from importlib import metadata

import numpy as np

from config import MAX_SAMPLES_LAB

CORE_PACKAGES = ("streamlit", "scikit-learn", "numpy", "scipy", "pandas", "statsmodels", "plotly")
OPTIONAL_PACKAGES = ("xgboost", "lightgbm", "catboost", "optuna", "shap", "imbalanced-learn", "DoubleML", "econml")


def package_versions(names: Iterable[str] = CORE_PACKAGES + OPTIONAL_PACKAGES) -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def clamp_n(n: int, limit: int = MAX_SAMPLES_LAB) -> int:
    """Bound the rows used by an interactive lab (public-deployment compute limit)."""
    return int(max(10, min(int(n), limit)))


def subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]


def version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
