"""Pages that use optional libraries must render when those libraries are missing."""

import os

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
PAGES = ["trees_ensembles/xgboost.py", "trees_ensembles/lightgbm.py", "trees_ensembles/catboost.py", "trees_ensembles/boosting_comparison.py",
         "tuning/bayesian_optimization.py", "interpretability/shap.py", "evaluation/imbalanced.py", "causal_ml/plr.py", "causal_ml/irm.py",
         "causal_ml/hte.py", "causal_ml/doubleml_package.py", "causal_ml/panel_did.py", "causal_ml/dml_extensions.py",
         "unsupervised/manifold.py", "projects/projects_hub.py", "production/mlops.py", "resources/about.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_optional_libraries(page, monkeypatch):
    import core.registry as reg

    monkeypatch.setattr(reg, "_try_import", lambda name: None)
    at = AppTest.from_file(ENTRY, default_timeout=300)
    at.session_state["level"] = "research"
    at.run()
    at.switch_page(f"app_pages/{page}").run()
    assert not at.exception, [e.message for e in at.exception]
