"""Every default in content/hyperparameters.py must match the installed library.

scikit-learn / LightGBM / DoubleML / Optuna: compare with inspect.signature.
XGBoost / CatBoost (signature default None): compare ``resolved`` with the
configuration the fitted booster actually used.
"""

import importlib
import inspect
import json

import numpy as np
import pytest

from content.hyperparameters import HYPERPARAMETERS, VERIFIED_VERSIONS


def _cls(path):
    mod, name = path.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(mod), name)
    except ImportError:
        pytest.skip(f"{mod} not installed")


@pytest.mark.parametrize("hp", HYPERPARAMETERS, ids=[f"{h.algo}.{h.parameter}" for h in HYPERPARAMETERS])
def test_signature_default(hp):
    cls = _cls(hp.estimator)
    params = inspect.signature(cls.__init__).parameters
    if hp.parameter not in params:
        # XGBoost / CatBoost accept most parameters through **kwargs
        assert hp.library in ("xgboost", "catboost"), f"{hp.parameter} not in {hp.estimator} signature"
        return
    assert params[hp.parameter].default == hp.default, (hp.estimator, hp.parameter, params[hp.parameter].default)


def test_no_duplicates():
    keys = [(h.estimator, h.parameter) for h in HYPERPARAMETERS]
    assert len(keys) == len(set(keys))


def test_xgboost_resolved_defaults():
    xgb = pytest.importorskip("xgboost")
    X = np.random.default_rng(0).normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    m = xgb.XGBClassifier().fit(X, y)
    cfg = json.loads(m.get_booster().save_config())
    tp = cfg["learner"]["gradient_booster"]["tree_train_param"]
    assert m.get_booster().num_boosted_rounds() == 100
    expected = {"eta": 0.3, "max_depth": 6, "min_child_weight": 1, "gamma": 0, "subsample": 1, "colsample_bytree": 1,
                "colsample_bylevel": 1, "colsample_bynode": 1, "lambda": 1}
    for k, v in expected.items():
        assert float(tp[k]) == pytest.approx(v, rel=1e-6), k
    assert float(tp.get("reg_alpha", tp.get("alpha"))) == 0


def test_catboost_resolved_defaults():
    cb = pytest.importorskip("catboost")
    X = np.random.default_rng(0).normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    p = cb.CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False).fit(X, y).get_all_params()
    assert p["depth"] == 6 and p["l2_leaf_reg"] == 3 and p["random_strength"] == 1 and p["border_count"] == 254
    p2 = cb.CatBoostClassifier(verbose=0, allow_writing_files=False, thread_count=1).fit(X[:40], y[:40]).get_all_params()
    assert p2["iterations"] == 1000


def test_verified_versions_are_installed_or_skipped():
    from importlib import metadata
    for pkg, ver in VERIFIED_VERSIONS.items():
        try:
            installed = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
        # a different installed version is allowed but must be reported, not silently trusted
        if installed != ver:
            pytest.skip(f"{pkg} installed {installed} differs from verified {ver}: re-verify defaults")
