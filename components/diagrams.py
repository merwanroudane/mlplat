"""Diagrams rendered natively: Mermaid (st.mermaid_chart) and Graphviz DOT
(st.graphviz_chart, rendered client-side, so no system Graphviz is needed)."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

_THEME = (
    "%%{init: {'theme':'base','themeVariables':{"
    "'primaryColor':'#E7F5FF','primaryBorderColor':'#4DABF7','primaryTextColor':'#212529',"
    "'lineColor':'#7048E8','secondaryColor':'#F3F0FF','tertiaryColor':'#E6FCF5',"
    "'fontFamily':'IBM Plex Sans Arabic, sans-serif','fontSize':'15px'}}}%%\n"
)

# Light multicolour fills for consecutive steps (fill, border).
_STEP_COLORS = [("#E7F5FF", "#4DABF7"), ("#F3F0FF", "#9775FA"), ("#FFF9DB", "#FCC419"),
                ("#E6FCF5", "#38D9A9"), ("#FFF4E6", "#FFA94D"), ("#E3FAFC", "#3BC9DB")]


def mermaid(src: str) -> None:
    st.mermaid_chart(_THEME + src.strip(), width="stretch")


def flow(steps: Sequence[str], direction: str = "LR", highlight: int | None = None) -> None:
    """A linear process diagram with multicolour steps; optionally highlight one step."""
    lines = [f"flowchart {direction}"]
    for i, label in enumerate(steps):
        safe = label.replace('"', "'")
        lines.append(f'  S{i}["{safe}"]')
    for i in range(len(steps) - 1):
        lines.append(f"  S{i} --> S{i + 1}")
    for j, (fill, stroke) in enumerate(_STEP_COLORS):
        lines.append(f"  classDef c{j} fill:{fill},stroke:{stroke},color:#212529;")
    lines.append("  classDef hot fill:#7048E8,color:#fff,stroke:#5F3DC4;")
    for i in range(len(steps)):
        lines.append(f"  class S{i} {'hot' if i == highlight else f'c{i % len(_STEP_COLORS)}'};")
    mermaid("\n".join(lines))


def graphviz(dot: str) -> None:
    st.graphviz_chart(dot, width="stretch")
