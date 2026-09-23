import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.datasets import load_dataset
from utils.preprocessing import readiness_report
from utils.report import build_report


def test_report_contains_identity_sections_and_versions():
    rep = build_report("T", [("A", "text"), ("B", pd.DataFrame({"x": [1.5], "y": ["a|b"]})), ("C", {"k": 1}), ("D", ["i1", "i2"])])
    assert "Dr. Marwan Roudane" in rep
    assert "## A" in rep and "| x | y |" in rep and "a\\|b" in rep and "- **k:** 1" in rep and "- i2" in rep
    assert "scikit-learn" in rep


def test_readiness_has_thirteen_items_and_flags_leakage():
    df = load_dataset("mixed").copy()
    df["leak"] = df["approved"] * 10
    checks = readiness_report(df, "approved")
    assert len(checks) == 13
    assert [c for c in checks if c.item == "Leakage"][0].status == "fail"


def test_progress_export_import_roundtrip():
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from core.state import init_state, export_progress, import_progress, set_complete\n"
        "init_state()\n"
        "set_complete('what_is_ml')\n"
        "payload = export_progress()\n"
        "st.session_state['completed'] = set()\n"
        "ok, _ = import_progress(payload)\n"
        "bad, _ = import_progress('not json')\n"
        "st.session_state['result'] = (ok, sorted(st.session_state['completed']), bad)\n"
    )
    at.run()
    assert not at.exception
    assert at.session_state["result"] == (True, ["what_is_ml"], False)
