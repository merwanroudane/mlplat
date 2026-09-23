"""Double/debiased machine learning implemented from the formulas.

Reference: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins
(2018), "Double/debiased machine learning for treatment and structural
parameters", The Econometrics Journal 21(1), C1–C68, doi:10.1111/ectj.12097.

Everything here is transparent NumPy + scikit-learn so learners see each
step; the DoubleML package (optional) is used on the package pages and the
tests check that both give matching estimates on the same folds.

Estimators return ``DMLResult`` with the point estimate, a standard error
from the (averaged) orthogonal score, and a normal 95% interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from config import RANDOM_SEED

Array = np.ndarray
Z95 = 1.959963984540054


@dataclass
class DMLResult:
    name: str
    theta: float
    se: float
    extra: dict = field(default_factory=dict)

    @property
    def ci(self) -> tuple[float, float]:
        return self.theta - Z95 * self.se, self.theta + Z95 * self.se

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return lo <= truth <= hi


# ------------------------------------------------------------ cross-fitting
def fold_indices(n: int, n_folds: int = 5, seed: int = RANDOM_SEED, stratify: Array | None = None) -> list[tuple[Array, Array]]:
    """(train_idx, test_idx) pairs. Every observation is in exactly one test fold."""
    if stratify is not None:
        return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n), stratify))
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n)))


def cross_fit_predict(learner, X: Array, target: Array, folds: list[tuple[Array, Array]],
                      proba: bool = False) -> Array:
    """Out-of-fold predictions: the model predicting unit i never saw unit i."""
    out = np.empty(len(target), dtype=float)
    for tr, te in folds:
        m = clone(learner).fit(X[tr], target[tr])
        out[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return out


def in_sample_predict(learner, X: Array, target: Array, proba: bool = False) -> Array:
    """Same learner fitted and evaluated on the full sample — what cross-fitting avoids."""
    m = clone(learner).fit(X, target)
    return m.predict_proba(X)[:, 1] if proba else m.predict(X)


# ---------------------------------------------------------------- PLR
def plr_from_nuisance(y: Array, d: Array, l_hat: Array, m_hat: Array, g_hat: Array | None = None,
                      score: str = "partialling out") -> DMLResult:
    """Solve the orthogonal moment for θ given nuisance predictions.

    partialling out:  ψ = (Y − ℓ(X) − θ(D − m(X))) (D − m(X))          (ℓ = E[Y|X])
    IV-type:          ψ = (Y − θD − g(X)) (D − m(X))                   (g = E[Y − θD|X])
    """
    v = d - m_hat
    if score == "partialling out":
        u = y - l_hat
        theta = float(np.sum(v * u) / np.sum(v * v))
        psi = (u - theta * v) * v
        J = -np.mean(v * v)
    else:
        if g_hat is None:
            raise ValueError("IV-type score needs g_hat")
        theta = float(np.sum(v * (y - g_hat)) / np.sum(v * d))
        psi = (y - theta * d - g_hat) * v
        J = -np.mean(v * d)
    se = float(np.sqrt(np.mean(psi ** 2) / J ** 2 / len(y)))
    return DMLResult(f"DML-PLR ({score})", theta, se, {"psi": psi, "v": v})


def dml_plr(X: Array, y: Array, d: Array, ml_l, ml_m, n_folds: int = 5, n_rep: int = 1,
            seed: int = RANDOM_SEED) -> DMLResult:
    """Cross-fitted PLR (partialling out, DML2). With n_rep > 1 the median over splits is used,
    with the median-adjusted standard error (Chernozhukov et al., 2018, Sec. 3.4)."""
    thetas, ses, r2 = [], [], {}
    for r in range(n_rep):
        folds = fold_indices(len(y), n_folds, seed + r)
        l_hat = cross_fit_predict(ml_l, X, y, folds)
        m_hat = cross_fit_predict(ml_m, X, d, folds)
        res = plr_from_nuisance(y, d, l_hat, m_hat)
        thetas.append(res.theta)
        ses.append(res.se)
        if r == 0:
            r2 = {"rmse_l": float(np.sqrt(np.mean((y - l_hat) ** 2))),
                  "rmse_m": float(np.sqrt(np.mean((d - m_hat) ** 2))), "l_hat": l_hat, "m_hat": m_hat}
    theta = float(np.median(thetas))
    se = float(np.sqrt(np.median(np.asarray(ses) ** 2 + (np.asarray(thetas) - theta) ** 2)))
    return DMLResult("DML (cross-fitted)", theta, se, {"thetas": thetas, **r2})


def plr_no_crossfit(X: Array, y: Array, d: Array, ml_l, ml_m) -> DMLResult:
    """Same orthogonal score but nuisances fitted and predicted in-sample (overfitting bias)."""
    l_hat = in_sample_predict(ml_l, X, y)
    m_hat = in_sample_predict(ml_m, X, d)
    res = plr_from_nuisance(y, d, l_hat, m_hat)
    res.name = "Orthogonal, no cross-fitting"
    return res


def naive_ols(X: Array | None, y: Array, d: Array) -> DMLResult:
    """OLS of Y on D (and X linearly if given) with HC0 robust SE for the D coefficient."""
    Z = np.c_[np.ones(len(y)), d] if X is None else np.c_[np.ones(len(y)), d, X]
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    cov = ZtZ_inv @ (Z.T * e ** 2) @ Z @ ZtZ_inv
    return DMLResult("OLS (linear controls)" if X is not None else "OLS (no controls)", float(beta[1]),
                     float(np.sqrt(cov[1, 1])))


def naive_plugin(X: Array, y: Array, d: Array, ml_g, n_iter: int = 5) -> DMLResult:
    """Non-orthogonal 'plug-in': alternate θ and g on Y − θD (regularisation bias).

    Mimics the naive estimator of Chernozhukov et al. (2018, Fig. 1): the ML
    estimate of g is plugged directly into θ = Σ D(Y − ĝ) / Σ D², no
    residualisation of D. SE is not valid, so it is reported as NaN.
    """
    theta = 0.0
    folds = fold_indices(len(y), 2, RANDOM_SEED)
    for _ in range(n_iter):
        g_hat = cross_fit_predict(ml_g, X, y - theta * d, folds)
        theta = float(np.sum(d * (y - g_hat)) / np.sum(d * d))
    return DMLResult("Naive ML plug-in (non-orthogonal)", theta, float("nan"))


# ----------------------------------------------------------------- PLIV
def dml_pliv(X: Array, y: Array, d: Array, z: Array, ml_l, ml_m, ml_r, n_folds: int = 5,
             seed: int = RANDOM_SEED) -> DMLResult:
    """PLIV, partialling out: ψ = (Ỹ − θ D̃) Z̃ with Ỹ = Y − ℓ(X), D̃ = D − r(X), Z̃ = Z − m(X)."""
    folds = fold_indices(len(y), n_folds, seed)
    yt = y - cross_fit_predict(ml_l, X, y, folds)
    zt = z - cross_fit_predict(ml_m, X, z, folds)
    dt = d - cross_fit_predict(ml_r, X, d, folds)
    theta = float(np.sum(zt * yt) / np.sum(zt * dt))
    psi = (yt - theta * dt) * zt
    J = -np.mean(zt * dt)
    se = float(np.sqrt(np.mean(psi ** 2) / J ** 2 / len(y)))
    first_stage = float(np.corrcoef(zt, dt)[0, 1])
    return DMLResult("DML-PLIV", theta, se, {"first_stage_corr": first_stage})


# ------------------------------------------------------------ IRM (AIPW)
def dml_irm(X: Array, y: Array, d: Array, ml_g, ml_m, n_folds: int = 5, score: str = "ATE",
            trim: float = 0.01, seed: int = RANDOM_SEED) -> DMLResult:
    """Doubly robust (AIPW) score with cross-fitting.

    ATE: ψ = g(1,X) − g(0,X) + D(Y − g(1,X))/m(X) − (1−D)(Y − g(0,X))/(1 − m(X)) − θ
    ATT: ψ = [D(Y − g(0,X)) − m(X)(1−D)(Y − g(0,X))/(1 − m(X))]/p − D θ/p
    Propensities are truncated to [trim, 1 − trim].
    """
    folds = fold_indices(len(y), n_folds, seed, stratify=d)
    g0, g1, m = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for tr, te in folds:
        t0, t1 = tr[d[tr] == 0], tr[d[tr] == 1]
        g0[te] = clone(ml_g).fit(X[t0], y[t0]).predict(X[te])
        g1[te] = clone(ml_g).fit(X[t1], y[t1]).predict(X[te])
        m[te] = clone(ml_m).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    m = np.clip(m, trim, 1 - trim)
    if score == "ATE":
        phi = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
        theta = float(np.mean(phi))
        psi = phi - theta
        se = float(np.std(psi, ddof=0) / np.sqrt(len(y)))
    else:
        p = d.mean()
        a = d * (y - g0) - m * (1 - d) * (y - g0) / (1 - m)
        theta = float(np.mean(a) / p)
        psi = (a - d * theta) / p
        se = float(np.sqrt(np.mean(psi ** 2) / len(y)))
    return DMLResult(f"DML-IRM ({score})", theta, se, {"m_hat": m, "g0": g0, "g1": g1})


def ipw_ate(y: Array, d: Array, m: Array) -> float:
    """Inverse propensity weighting (Horvitz–Thompson) — sensitive to extreme propensities."""
    return float(np.mean(d * y / m - (1 - d) * y / (1 - m)))


# -------------------------------------------------------- IIVM (LATE)
def dml_iivm(X: Array, y: Array, d: Array, z: Array, ml_g, ml_m, ml_r, n_folds: int = 5,
             trim: float = 0.01, seed: int = RANDOM_SEED) -> DMLResult:
    """LATE with a binary instrument: ratio of two doubly robust intention-to-treat effects."""
    folds = fold_indices(len(y), n_folds, seed, stratify=z)
    n = len(y)
    g0, g1, r0, r1, m = (np.empty(n) for _ in range(5))
    for tr, te in folds:
        a0, a1 = tr[z[tr] == 0], tr[z[tr] == 1]
        g0[te] = clone(ml_g).fit(X[a0], y[a0]).predict(X[te])
        g1[te] = clone(ml_g).fit(X[a1], y[a1]).predict(X[te])
        for arr, idx in ((r0, a0), (r1, a1)):
            dd = d[idx]
            if len(np.unique(dd)) < 2:  # one-sided non-compliance: D is constant in this arm
                arr[te] = float(dd.mean())
            else:
                arr[te] = clone(ml_r).fit(X[idx], dd).predict_proba(X[te])[:, 1]
        m[te] = clone(ml_m).fit(X[tr], z[tr]).predict_proba(X[te])[:, 1]
    m = np.clip(m, trim, 1 - trim)
    num = g1 - g0 + z * (y - g1) / m - (1 - z) * (y - g0) / (1 - m)
    den = r1 - r0 + z * (d - r1) / m - (1 - z) * (d - r0) / (1 - m)
    theta = float(np.mean(num) / np.mean(den))
    psi = num - theta * den
    se = float(np.sqrt(np.mean(psi ** 2) / np.mean(den) ** 2 / n))
    return DMLResult("DML-IIVM (LATE)", theta, se, {"first_stage": float(np.mean(den))})


# ------------------------------------------------------------ DR-learner
def dr_learner_cate(X: Array, y: Array, d: Array, ml_g, ml_m, ml_final, n_folds: int = 5,
                    trim: float = 0.01, seed: int = RANDOM_SEED) -> tuple[Array, Array]:
    """DR-learner (Kennedy, 2023): regress the cross-fitted AIPW pseudo-outcome on X.

    Returns (pseudo_outcome, cate_hat) where cate_hat is out-of-fold.
    """
    res = dml_irm(X, y, d, ml_g, ml_m, n_folds=n_folds, trim=trim, seed=seed)
    g0, g1, m = res.extra["g0"], res.extra["g1"], res.extra["m_hat"]
    phi = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
    cate = cross_fit_predict(ml_final, X, phi, fold_indices(len(y), n_folds, seed + 7))
    return phi, cate


def gate(phi: Array, groups: Array) -> list[dict]:
    """Group average treatment effects: mean of the orthogonal pseudo-outcome within each group."""
    out = []
    for g in np.unique(groups):
        v = phi[groups == g]
        se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        out.append({"group": g, "gate": float(v.mean()), "se": se, "lo": float(v.mean() - Z95 * se),
                    "hi": float(v.mean() + Z95 * se), "n": int(len(v))})
    return out


# --------------------------------------------------------- Monte Carlo
def monte_carlo_plr(n_reps: int, make_data, ml_l, ml_m, theta0: float, n_folds: int = 5,
                    include: tuple[str, ...] = ("ols", "no_cf", "dml"), seed: int = RANDOM_SEED) -> dict[str, np.ndarray]:
    """Repeat the experiment ``n_reps`` times; returns arrays of (theta, se) per method."""
    out = {k: [] for k in include}
    for r in range(int(n_reps)):
        df = make_data(seed + 1000 + r)
        X = df.filter(like="x").to_numpy()
        y, d = df["y"].to_numpy(), df["d"].to_numpy()
        if "ols" in out:
            res = naive_ols(X, y, d)
            out["ols"].append((res.theta, res.se))
        if "no_cf" in out:
            res = plr_no_crossfit(X, y, d, ml_l, ml_m)
            out["no_cf"].append((res.theta, res.se))
        if "dml" in out:
            res = dml_plr(X, y, d, ml_l, ml_m, n_folds=n_folds, seed=seed + r)
            out["dml"].append((res.theta, res.se))
    return {k: np.array(v) for k, v in out.items()}


def summarize_mc(arr: np.ndarray, theta0: float) -> dict[str, float]:
    th, se = arr[:, 0], arr[:, 1]
    covered = np.abs(th - theta0) <= Z95 * se
    return {"mean": float(th.mean()), "bias": float(th.mean() - theta0), "sd": float(th.std(ddof=1)),
            "mean_se": float(np.nanmean(se)), "rmse": float(np.sqrt(np.mean((th - theta0) ** 2))),
            "coverage": float(np.mean(covered))}


# -------------------------------------------------------- IV data (PLIV)
def make_pliv(n: int = 1000, strength: float = 1.0, theta: float = 0.5, seed: int = RANDOM_SEED):
    """Endogenous D (shares the unobservable U with Y) and an instrument Z whose relevance is ``strength``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    U = rng.normal(size=n)
    z = 0.5 * X[:, 0] + rng.normal(size=n)
    d = strength * z + np.sin(X[:, 1]) + U + rng.normal(scale=0.5, size=n)
    y = theta * d + np.cos(X[:, 0]) + 0.5 * X[:, 2] ** 2 + U + rng.normal(scale=0.5, size=n)
    return X, y, d, z


def make_iivm(n: int = 2000, late: float = 1.0, share_compliers: float = 0.6, seed: int = RANDOM_SEED):
    """Binary instrument (e.g., encouragement) with always-takers, never-takers and compliers."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    pz = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    z = (rng.random(n) < pz).astype(int)
    u = rng.random(n)
    rest = (1 - share_compliers) / 2
    typ = np.where(u < share_compliers, "complier", np.where(u < share_compliers + rest, "always", "never"))
    d = np.where(typ == "always", 1, np.where(typ == "never", 0, z))
    base = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.8 * (typ == "always") - 0.5 * (typ == "never")
    effect = np.where(typ == "complier", late, np.where(typ == "always", late + 1.0, late - 0.5))
    y = base + effect * d + rng.normal(size=n)
    return X, y, d, z, typ
