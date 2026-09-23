"""Run every end-to-end project from the Projects hub and check its report renders."""

import os

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
PAGE = "app_pages/projects/projects_hub.py"


@pytest.mark.parametrize("pid", [f"p{i}" for i in range(1, 11)])
def test_project_runs(pid):
    at = AppTest.from_file(ENTRY, default_timeout=900)
    at.run()
    at.switch_page(PAGE).run()
    at.selectbox(key="proj_pick").set_value(pid).run()
    at.button(key="proj_run").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.dataframe) >= 1  # at least one results table
