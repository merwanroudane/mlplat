"""Optional-dependency registry (section 81).

Heavy libraries are optional. Pages call ``optional("xgboost")``; when the
package is missing the page still renders its theory and reference code and
shows ``missing_notice`` instead of crashing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import streamlit as st


@dataclass(frozen=True)
class OptionalDep:
    import_name: str
    pip_name: str
    verified_version: str
    purpose_ar: str
    docs_url: str


OPTIONAL_DEPS: dict[str, OptionalDep] = {
    "xgboost": OptionalDep("xgboost", "xgboost", "3.2.0 (3.4.1 على Python ≥ 3.12)", "XGBoost labs",
                           "https://xgboost.readthedocs.io/en/stable/parameter.html"),
    "lightgbm": OptionalDep("lightgbm", "lightgbm", "4.7.0", "LightGBM labs",
                            "https://lightgbm.readthedocs.io/en/stable/Parameters.html"),
    "catboost": OptionalDep("catboost", "catboost", "1.2.10", "CatBoost labs",
                            "https://catboost.ai/docs/en/references/training-parameters/"),
    "optuna": OptionalDep("optuna", "optuna", "5.0.0", "HPO / Optuna lab",
                          "https://optuna.readthedocs.io/en/stable/"),
    "shap": OptionalDep("shap", "shap", "0.51.0 (0.52.0 على Python ≥ 3.12)", "SHAP lab",
                        "https://shap.readthedocs.io/en/latest/"),
    "imblearn": OptionalDep("imblearn", "imbalanced-learn", "0.14.2", "SMOTE / resampling",
                            "https://imbalanced-learn.org/stable/"),
    "doubleml": OptionalDep("doubleml", "DoubleML", "0.11.4", "DoubleML package labs",
                            "https://docs.doubleml.org/stable/"),
    "econml": OptionalDep("econml", "econml", "0.17.0", "CausalForestDML / meta-learners",
                          "https://www.pywhy.org/EconML/"),
}


@lru_cache(maxsize=None)
def _try_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError, or a broken binary wheel on the host
        return None


def optional(name: str) -> ModuleType | None:
    """Return the imported module, or None if it is not installed."""
    return _try_import(OPTIONAL_DEPS[name].import_name if name in OPTIONAL_DEPS else name)


def available(name: str) -> bool:
    return optional(name) is not None


def missing_notice(name: str) -> None:
    dep = OPTIONAL_DEPS[name]
    st.info(
        f"المكتبة الاختيارية **`{dep.pip_name}`** غير مثبتة في هذه البيئة، لذلك يُعرض المحتوى النظري والكود المرجعي فقط. "
        f"للتشغيل محليًا: `pip install {dep.pip_name}` (الإصدار المتحقَّق منه: {dep.verified_version}).",
        icon=":material/extension_off:",
    )


def installed_version(name: str) -> str | None:
    mod = optional(name)
    return getattr(mod, "__version__", None) if mod is not None else None
