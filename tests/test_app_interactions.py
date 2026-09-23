"""Click every button on every page (labs run their code paths) and assert no exception."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from core.curriculum import MODULES

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
SKIP_PREFIXES = ("bm_", "quiz_retry_", "FormSubmitter")


@pytest.mark.parametrize("module", MODULES, ids=[m.id for m in MODULES])
def test_click_all_buttons(module):
    at = AppTest.from_file(ENTRY, default_timeout=600)
    at.session_state["level"] = "research"
    at.session_state["reduce_motion"] = True  # animations step manually; no autoplay timers in tests
    at.run()
    at.switch_page(module.file).run()
    assert not at.exception, [e.message for e in at.exception]
    keys = [b.key for b in at.button if b.key and not b.key.startswith(SKIP_PREFIXES)]
    for key in keys:
        matches = [b for b in at.button if b.key == key]
        if not matches:
            continue  # the button disappeared after a previous click (e.g. play -> pause)
        matches[0].click().run()
        assert not at.exception, (key, [e.message for e in at.exception])
