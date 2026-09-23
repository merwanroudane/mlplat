"""Hyperparameter-optimisation helpers (section 38).

Bounded by config.MAX_OPTUNA_TRIALS so the public app stays responsive.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from config import MAX_OPTUNA_TRIALS, RANDOM_SEED


def branin(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Branin–Hoo test function on [-5, 10] × [0, 15]; three global minima ≈ 0.397887."""
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def grid_vs_random(f, n_trials: int, seed: int = RANDOM_SEED, important_dim: int = 0) -> dict[str, np.ndarray]:
    """Bergstra & Bengio (2012) illustration: with one important dimension, a grid of n trials
    tests only √n distinct values of it, random search tests n."""
    k = int(np.floor(np.sqrt(n_trials)))
    g = np.linspace(0, 1, k)
    grid = np.array([(a, b) for a in g for b in g])
    rnd = np.random.default_rng(seed).random((k * k, 2))
    return {"grid": grid, "random": rnd}


def cv_objective(model, X, y, cv=3, scoring="roc_auc") -> float:
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring)))


def run_optuna(make_model, space: dict, X, y, n_trials: int, sampler: str = "tpe", pruner: str = "none",
               cv: int = 3, scoring: str = "roc_auc", seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Run an Optuna study over ``space`` = {name: ("float"|"log"|"int"|"cat", low, high | options)}.

    Pruning uses per-fold intermediate reports (each fold is one 'step').
    Returns one row per trial with parameters, value, state and elapsed time.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    samplers = {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed),
        "gp": lambda: optuna.samplers.GPSampler(seed=seed),
        "qmc": lambda: optuna.samplers.QMCSampler(seed=seed, qmc_type="sobol", scramble=True),
    }
    pruners = {"none": optuna.pruners.NopPruner, "median": lambda: optuna.pruners.MedianPruner(n_startup_trials=5),
               "hyperband": optuna.pruners.HyperbandPruner}
    splitter = StratifiedKFold(cv, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y))
    from sklearn.metrics import get_scorer
    scorer = get_scorer(scoring)

    def objective(trial):
        params = {}
        for name, spec in space.items():
            kind = spec[0]
            if kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == "log":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            else:
                params[name] = trial.suggest_categorical(name, spec[1])
        scores = []
        for step, (tr, te) in enumerate(folds):
            m = clone(make_model(params)).fit(X[tr], y[tr])
            scores.append(scorer(m, X[te], y[te]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=samplers[sampler](), pruner=pruners[pruner]())
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=int(min(n_trials, MAX_OPTUNA_TRIALS)))
    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, **t.params, "value": t.value, "state": t.state.name,
                     "seconds": (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else np.nan})
    df = pd.DataFrame(rows)
    df.attrs["total_seconds"] = time.perf_counter() - t0
    return df


def successive_halving_trace(n_configs: int = 27, eta: int = 3, min_resource: int = 1,
                             seed: int = RANDOM_SEED) -> list[dict]:
    """Synthetic successive-halving bracket: each config has a latent quality; noisy early scores
    become accurate as resource grows. Returns rungs with surviving configs (for animation)."""
    rng = np.random.default_rng(seed)
    quality = rng.beta(2, 5, n_configs)
    alive = np.arange(n_configs)
    r = min_resource
    rungs = []
    while len(alive) >= 1:
        noise = 0.25 / np.sqrt(r)
        scores = quality[alive] + rng.normal(scale=noise, size=len(alive))
        rungs.append({"resource": r, "configs": alive.copy(), "scores": scores.copy(), "quality": quality[alive].copy()})
        if len(alive) == 1:
            break
        keep = max(1, len(alive) // eta)
        alive = alive[np.argsort(-scores)[:keep]]
        r *= eta
    return rungs
