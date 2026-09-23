"""Mathematical layer: every equation is followed by its symbols, the
intuition and a worked numerical example (section 62 of the brief)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import streamlit as st

from core.state import next_key


def formula(
    latex: str,
    title: str | None = None,
    symbols: Mapping[str, str] | None = None,
    intuition: str | None = None,
    example: str | Callable[[], None] | None = None,
) -> None:
    with st.container(key=next_key("ml-formula")):
        if title:
            st.markdown(f"**{title}**")
        st.latex(latex)
        if symbols:
            st.markdown("**الرموز:**")
            st.markdown("\n".join(f"- ${sym}$ : {meaning}" for sym, meaning in symbols.items()))
        if intuition:
            st.markdown(f"**الحدس:** {intuition}")
        if example is not None:
            with st.expander("مثال عددي · Numerical example", icon=":material/calculate:"):
                if callable(example):
                    example()
                else:
                    st.markdown(example)
