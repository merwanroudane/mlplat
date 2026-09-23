"""Models: from-scratch educational implementations (section 71) and the
model factory used by the Parameter Playground (section 69).

From-scratch classes follow the scikit-learn estimator contract (fit /
predict / predict_proba, trailing-underscore fitted attributes) so they can
be compared side-by-side with the library versions. Tests check that they
agree with scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import MAX_TREES_LAB, RANDOM_SEED

Array = np.ndarray


# ======================================================== from scratch
class LinearRegressionGD:
    """Least squares by batch gradient descent on MSE (educational)."""

    def __init__(self, lr: float = 0.1, n_iter: int = 500):
        self.lr, self.n_iter = lr, n_iter

    def fit(self, X: Array, y: Array) -> "LinearRegressionGD":
        X = np.asarray(X, float)
        n, p = X.shape
        self.coef_, self.intercept_ = np.zeros(p), float(np.mean(y))
        self.loss_curve_ = []
        for _ in range(self.n_iter):
            r = y - (X @ self.coef_ + self.intercept_)
            self.coef_ += self.lr * 2 * X.T @ r / n
            self.intercept_ += self.lr * 2 * r.mean()
            self.loss_curve_.append(float(np.mean(r ** 2)))
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def sigmoid(z: Array) -> Array:
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(z)))  # numerically stable form of 1/(1+e^{-z})


class LogisticRegressionGD:
    """Binary logistic regression with L2 penalty, trained by gradient descent.

    Objective (same scaling as scikit-learn's lbfgs with C):
    C · Σ log-loss_i + ½‖w‖²  ⇒ gradient  C · Xᵀ(p − y) + w.
    Divided by n for a stable step size.
    """

    def __init__(self, C: float = 1.0, lr: float = 0.5, n_iter: int = 2000):
        self.C, self.lr, self.n_iter = C, lr, n_iter

    def fit(self, X: Array, y: Array) -> "LogisticRegressionGD":
        X = np.asarray(X, float)
        n, p = X.shape
        w, b = np.zeros(p), 0.0
        self.loss_curve_ = []
        for _ in range(self.n_iter):
            pr = sigmoid(X @ w + b)
            gw = (self.C * X.T @ (pr - y) + w) / n
            gb = self.C * np.sum(pr - y) / n
            w -= self.lr * gw
            b -= self.lr * gb
            eps = 1e-12
            self.loss_curve_.append(float(-np.mean(y * np.log(pr + eps) + (1 - y) * np.log(1 - pr + eps))))
        self.coef_, self.intercept_ = w, b
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: Array) -> Array:
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: Array) -> Array:
        p = sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X: Array) -> Array:
        return (self.decision_function(X) > 0).astype(int)


class KNNScratch:
    """k-nearest-neighbour classifier with Minkowski distance (brute force)."""

    def __init__(self, n_neighbors: int = 5, p: float = 2.0, weights: str = "uniform"):
        self.n_neighbors, self.p, self.weights = n_neighbors, p, weights

    def fit(self, X: Array, y: Array) -> "KNNScratch":
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def _dist(self, X: Array) -> Array:
        diff = np.abs(np.asarray(X, float)[:, None, :] - self.X_[None, :, :])
        if np.isinf(self.p):
            return diff.max(axis=2)
        return (diff ** self.p).sum(axis=2) ** (1 / self.p)

    def kneighbors(self, X: Array) -> tuple[Array, Array]:
        d = self._dist(X)
        idx = np.argsort(d, axis=1, kind="stable")[:, : self.n_neighbors]
        return np.take_along_axis(d, idx, axis=1), idx

    def predict_proba(self, X: Array) -> Array:
        d, idx = self.kneighbors(X)
        w = np.ones_like(d) if self.weights == "uniform" else 1 / np.maximum(d, 1e-12)
        labels = self.y_[idx]
        out = np.zeros((len(d), len(self.classes_)))
        for j, c in enumerate(self.classes_):
            out[:, j] = (w * (labels == c)).sum(axis=1)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: Array) -> Array:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def kmeans_history(X: Array, k: int, n_iter: int = 15, seed: int = RANDOM_SEED, init: str = "random") -> list[dict]:
    """Lloyd's algorithm, recording every assign/update half-step for animation.

    Returns a list of dicts: {"phase": "init"|"assign"|"update", "centers", "labels", "inertia"}.
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(centers)
    else:
        C = X[rng.choice(len(X), k, replace=False)].copy()
    hist = [{"phase": "init", "centers": C.copy(), "labels": None, "inertia": None}]
    labels = None
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), new_labels].sum())
        hist.append({"phase": "assign", "centers": C.copy(), "labels": new_labels.copy(), "inertia": inertia})
        C_new = np.array([X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else C[j] for j in range(k)])
        d2 = ((X[:, None, :] - C_new[None]) ** 2).sum(-1)
        hist.append({"phase": "update", "centers": C_new.copy(), "labels": new_labels.copy(),
                     "inertia": float(d2[np.arange(len(X)), new_labels].sum())})
        if labels is not None and np.array_equal(labels, new_labels) and np.allclose(C, C_new):
            break
        labels, C = new_labels, C_new
    return hist


class PCAScratch:
    """PCA via SVD of the centred data matrix (educational, simplified)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: Array) -> "PCAScratch":
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # sign convention: largest-|loading| of each component is positive (like sklearn's svd_flip on Vt)
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt = Vt * signs[:, None]
        self.components_ = Vt[: self.n_components]
        var = S ** 2 / (len(X) - 1)
        self.explained_variance_ = var[: self.n_components]
        self.explained_variance_ratio_ = (var / var.sum())[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        return self

    def transform(self, X: Array) -> Array:
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: Array) -> Array:
        return Z @ self.components_ + self.mean_


# =================================================== playground factory
@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "int" | "float" | "log" | "choice" | "bool"
    default: Any
    low: float | None = None
    high: float | None = None
    options: tuple = ()
    help: str = ""


@dataclass(frozen=True)
class PlaygroundModel:
    label: str
    family: str
    build: Callable[[dict], Any]
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)
    needs_scaling: bool = False


def _lr(p: dict):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=p["C"], l1_ratio=p["l1_ratio"], solver="saga" if p["l1_ratio"] > 0 else "lbfgs",
                              max_iter=5000)


def _knn(p: dict):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])


def _svc(p: dict):
    from sklearn.svm import SVC
    return SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], degree=p["degree"])


def _tree(p: dict):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(criterion=p["criterion"], max_depth=p["max_depth"] or None,
                                  min_samples_leaf=p["min_samples_leaf"], ccp_alpha=p["ccp_alpha"],
                                  random_state=RANDOM_SEED)


def _rf(p: dict):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"] or None,
                                  max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                  random_state=RANDOM_SEED, n_jobs=1)


def _hgb(p: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_iter=p["max_iter"],
                                          max_leaf_nodes=p["max_leaf_nodes"], l2_regularization=p["l2_regularization"],
                                          early_stopping=False, random_state=RANDOM_SEED)


def _nb(p: dict):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(var_smoothing=p["var_smoothing"])


def _qda(p: dict):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis(reg_param=p["reg_param"])


def _mlp(p: dict):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(p["units"],) * p["layers"], activation=p["activation"],
                         alpha=p["alpha"], learning_rate_init=p["learning_rate_init"], max_iter=800,
                         random_state=RANDOM_SEED)


PLAYGROUND: dict[str, PlaygroundModel] = {
    "logistic": PlaygroundModel("Logistic Regression", "linear", _lr, (
        ParamSpec("C", "log", 1.0, 1e-3, 1e3, help="مقلوب قوة التنظيم: أصغر ⇒ تنظيم أقوى."),
        ParamSpec("l1_ratio", "float", 0.0, 0.0, 1.0, help="0 = L2، 1 = L1، بينهما Elastic Net (بديل penalty منذ 1.8)."),
    ), needs_scaling=True),
    "knn": PlaygroundModel("k-Nearest Neighbors", "instance", _knn, (
        ParamSpec("n_neighbors", "int", 5, 1, 60, help="عدد الجيران: صغير ⇒ تباين عالٍ، كبير ⇒ تحيز عالٍ."),
        ParamSpec("weights", "choice", "uniform", options=("uniform", "distance")),
        ParamSpec("p", "choice", 2, options=(1, 2), help="1 = Manhattan، 2 = Euclidean (مع metric='minkowski')."),
    ), needs_scaling=True),
    "svc": PlaygroundModel("SVM (SVC)", "kernel", _svc, (
        ParamSpec("C", "log", 1.0, 1e-2, 1e3, help="كلفة أخطاء الهامش: كبير ⇒ هامش أضيق وتعقيد أعلى."),
        ParamSpec("kernel", "choice", "rbf", options=("linear", "poly", "rbf", "sigmoid")),
        ParamSpec("gamma", "choice", "scale", options=("scale", "auto", 0.1, 1.0, 10.0),
                  help="مدى تأثير النقطة الواحدة في نواة RBF/poly/sigmoid."),
        ParamSpec("degree", "int", 3, 1, 6, help="درجة نواة poly فقط."),
    ), needs_scaling=True),
    "tree": PlaygroundModel("Decision Tree", "tree", _tree, (
        ParamSpec("criterion", "choice", "gini", options=("gini", "entropy", "log_loss")),
        ParamSpec("max_depth", "int", 0, 0, 20, help="0 = بلا حد (None)."),
        ParamSpec("min_samples_leaf", "int", 1, 1, 50),
        ParamSpec("ccp_alpha", "float", 0.0, 0.0, 0.05, help="تقليم Cost-complexity."),
    )),
    "rf": PlaygroundModel("Random Forest", "ensemble", _rf, (
        ParamSpec("n_estimators", "int", 100, 1, MAX_TREES_LAB),
        ParamSpec("max_depth", "int", 0, 0, 20, help="0 = بلا حد (None)."),
        ParamSpec("max_features", "choice", "sqrt", options=("sqrt", "log2", 1.0)),
        ParamSpec("min_samples_leaf", "int", 1, 1, 50),
    )),
    "hgb": PlaygroundModel("HistGradientBoosting", "boosting", _hgb, (
        ParamSpec("learning_rate", "log", 0.1, 1e-3, 1.0),
        ParamSpec("max_iter", "int", 100, 1, MAX_TREES_LAB),
        ParamSpec("max_leaf_nodes", "int", 31, 2, 63),
        ParamSpec("l2_regularization", "float", 0.0, 0.0, 10.0),
    )),
    "nb": PlaygroundModel("Gaussian Naive Bayes", "probabilistic", _nb, (
        ParamSpec("var_smoothing", "log", 1e-9, 1e-12, 1.0),
    )),
    "qda": PlaygroundModel("QDA", "probabilistic", _qda, (
        ParamSpec("reg_param", "float", 0.0, 0.0, 1.0),
    )),
    "mlp": PlaygroundModel("MLP (neural network)", "neural", _mlp, (
        ParamSpec("layers", "int", 1, 1, 3),
        ParamSpec("units", "int", 16, 2, 64),
        ParamSpec("activation", "choice", "relu", options=("relu", "tanh", "logistic")),
        ParamSpec("alpha", "log", 1e-4, 1e-6, 1.0, help="تنظيم L2."),
        ParamSpec("learning_rate_init", "log", 1e-3, 1e-4, 0.1),
    ), needs_scaling=True),
}


def build_playground(name: str, params: dict):
    """Model wrapped in a Pipeline with a scaler when the algorithm needs it."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    spec = PLAYGROUND[name]
    model = spec.build(params)
    return make_pipeline(StandardScaler(), model) if spec.needs_scaling else model


def model_complexity(model) -> str:
    """A human-readable complexity summary of a fitted model."""
    est = model[-1] if hasattr(model, "steps") else model
    name = type(est).__name__
    if hasattr(est, "tree_"):
        return f"{est.get_n_leaves()} leaves, depth {est.get_depth()}"
    if hasattr(est, "estimators_") and name.startswith("RandomForest"):
        return f"{len(est.estimators_)} trees, mean leaves {np.mean([t.get_n_leaves() for t in est.estimators_]):.0f}"
    if name.startswith("HistGradientBoosting"):
        return f"{est.n_iter_} boosting iterations"
    if hasattr(est, "support_"):
        return f"{len(est.support_)} support vectors"
    if name.startswith("KNeighbors"):
        return f"stores all {est.n_samples_fit_} training points (k = {est.n_neighbors})"
    if hasattr(est, "coefs_"):
        return f"{sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_)} weights"
    if hasattr(est, "coef_"):
        return f"{np.count_nonzero(est.coef_)} non-zero coefficients"
    if hasattr(est, "theta_"):
        return f"{est.theta_.size} means + {est.var_.size} variances"
    return name
