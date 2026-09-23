"""Safe interactive code labs.

The code shown to the learner is *documentation* of a predefined Python
function that actually runs. Nothing typed by a user is ever passed to
``eval``/``exec`` (section 56 and 81 of the brief).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.state import next_key


def _show(result: Any) -> None:
    if isinstance(result, go.Figure):
        st.plotly_chart(result, key=next_key("lab-fig"))
    elif isinstance(result, pd.DataFrame):
        st.dataframe(result, width="stretch")
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(name=result.name or "value"), width="stretch")
    elif isinstance(result, (list, tuple)):
        for item in result:
            _show(item)
    elif isinstance(result, str):
        st.markdown(result)
    else:
        st.code(repr(result), language="text")


def code_lab(
    key: str,
    title: str,
    code: str | Callable[[dict], str],
    runner: Callable[..., Any],
    params: Callable[[], dict] | None = None,
    explanation: str | None = None,
) -> None:
    """Code → Run → Result → Explanation → Reset.

    ``params`` renders parameter widgets and returns their values; the same
    dict is passed to ``runner`` and used to format ``code``.
    """
    state_key = f"lab_run_{key}"
    with st.container(key=f"ml-lab-{key}"):
        st.markdown(f"**:material/science: {title}**")
        values = params() if params else {}
        shown = code(values) if callable(code) else code
        st.code(shown, language="python")
        c1, c2 = st.columns([1, 1])
        if c1.button("تشغيل · Run", key=f"{key}_run", type="primary", icon=":material/play_arrow:"):
            st.session_state[state_key] = True
        if c2.button("إعادة ضبط · Reset", key=f"{key}_reset", icon=":material/replay:"):
            st.session_state[state_key] = False
        if st.session_state.get(state_key):
            st.markdown("**النتيجة · Output**")
            try:
                _show(runner(**values))
            except Exception as exc:  # show an educational message, never a raw trace
                if os.environ.get("ML_STRICT") == "1":  # tests surface the real error
                    raise
                st.error(f"تعذّر تنفيذ المثال: {type(exc).__name__}. جرّب قيمًا أخرى للمعاملات.")
            if explanation:
                st.markdown(f"**التفسير:** {explanation}")


def explain_code(code: str, parts: Sequence[tuple[str, str]], output: str | None = None,
                 interpretation: str | None = None) -> None:
    """Break a one-liner into its pieces (section 57 of the brief)."""
    st.code(code, language="python")
    rows = "\n".join(f"| `{token}` | {meaning} |" for token, meaning in parts)
    st.markdown("| الجزء | ماذا يفعل؟ |\n| --- | --- |\n" + rows)
    if output:
        st.markdown(f"**نوع الناتج:** {output}")
    if interpretation:
        st.markdown(f"**كيف نفسّره؟** {interpretation}")
