"""Theme helpers: inject the design-system CSS and expose colour tokens.

Colours for Streamlit widgets and charts come from .streamlit/config.toml.
PALETTE mirrors those tokens for Plotly figures built in Python.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT / "assets" / "css" / "theme.css"

PALETTE = {
    "coral": "#F76707",
    "coral_deep": "#1971C2",
    "amber": "#F59F00",
    "peach": "#74C0FC",
    "purple": "#7048E8",
    "softred": "#F08C00",
    "sky": "#1C7ED6",
    "teal": "#12B886",
    "brown": "#15AABF",
    "indigo": "#4263EB",
    "muted": "#5C677D",
    "grid": "#E9EDF5",
    "bg": "#FCFCFF",
    "text": "#212529",
}
SEQUENCE = [PALETTE[k] for k in ("sky", "coral", "purple", "teal", "amber", "brown", "indigo")]
SEQUENTIAL_SCALE = ["#FFFBEA", "#C3FAE8", "#66D9E8", "#4DABF7", "#1C7ED6"]
DIVERGING_SCALE = [[0.0, "#7048E8"], [0.5, "#FCFCFF"], [1.0, "#F76707"]]


@lru_cache(maxsize=1)
def _css() -> str:
    # Comments are stripped: st.html sanitizes with DOMPurify, which drops a whole <style>
    # block if its text contains anything that looks like markup (e.g. "<name>").
    css = re.sub(r"/\*.*?\*/", "", CSS_PATH.read_text(encoding="utf-8"), flags=re.S)
    return css.replace("<", "").replace(">", " > ")


def inject_css() -> None:
    st.html(f"<style>{_css()}</style>")


def register_plotly_template() -> None:
    """The light DSplat-family Plotly template used by every figure in the app."""
    if "ml_academy" in pio.templates:
        return
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="IBM Plex Sans Arabic, Segoe UI, sans-serif", color=PALETTE["text"], size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        colorway=SEQUENCE,
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor="#D5DCE8", linecolor="#D5DCE8"),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor="#D5DCE8", linecolor="#D5DCE8"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=PALETTE["text"]),
        title=dict(x=0.02, font=dict(size=15)),
    )
    pio.templates["ml_academy"] = template
    pio.templates.default = "plotly_white+ml_academy"
