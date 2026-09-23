"""Headless smoke test: every page renders without an exception (st.testing AppTest), at the research level."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from core.curriculum import MODULES

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")


@pytest.mark.parametrize("module", MODULES, ids=[m.id for m in MODULES])
def test_page_renders(module):
    at = AppTest.from_file(ENTRY, default_timeout=240)
    at.session_state["level"] = "research"  # render every optional section
    at.run()
    at.switch_page(module.file).run()
    assert not at.exception, [e.message for e in at.exception]
