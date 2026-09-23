"""Cards, badges and comparison tables shared by all pages."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

import pandas as pd
import streamlit as st

from core.state import next_key


def card(title: str, body: str, icon: str = "") -> None:
    with st.container(key=next_key("ml-card")):
        st.markdown(f"**{icon + ' ' if icon else ''}{title}**")
        st.markdown(body)


def card_grid(items: Sequence[tuple[str, str]], columns: int = 3, icon: str = "") -> None:
    """Responsive grid of cards. Streamlit stacks columns on narrow screens."""
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (title, body) in zip(cols, items[start:start + columns]):
            with col:
                card(title, body, icon)


def badges(labels: Iterable[str]) -> None:
    inner = "".join(f'<span class="ml-badge">{html.escape(x)}</span>' for x in labels)
    st.html(f'<div class="ml-badges">{inner}</div>')


def comparison_table(rows: Sequence[dict], caption: str | None = None) -> None:
    """Static comparison table (Arabic text, RTL friendly via markdown)."""
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("\n", " ").replace("|", "\\|") for v in row) + " |")
    st.markdown("\n".join(lines))
    if caption:
        st.caption(caption)


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def concept_steps(steps: Sequence[tuple[str, str]]) -> None:
    """The pedagogical sequence: What? Why? Intuition? ... rendered as tabs."""
    tabs = st.tabs([s[0] for s in steps])
    for tab, (_, body) in zip(tabs, steps):
        with tab:
            st.markdown(body)
