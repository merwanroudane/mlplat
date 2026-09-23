"""Plot helpers shared across pages (Plotly, DSplat palette, LTR-safe).

Colour is never the only signal: classes also get distinct marker symbols
(section 85 — accessibility).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MAX_PLOT_POINTS
from core.state import next_key
from core.theme import PALETTE, SEQUENCE

SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x", "star", "cross", "triangle-down"]
# Two-class light background for decision regions (blue / orange, colour-blind safe pair)
REGION_SCALE = [[0.0, "#D0EBFF"], [0.5, "#FCFCFF"], [1.0, "#FFE8CC"]]
CLASS_COLORS = [PALETTE["sky"], PALETTE["coral"], PALETTE["purple"], PALETTE["teal"], PALETTE["amber"]]


def plot(fig: go.Figure, height: int | None = None) -> None:
    """Render a Plotly figure with a stable unique key."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, key=next_key("fig"), config={"displaylogo": False})


def sample_for_plot(df: pd.DataFrame, n: int = MAX_PLOT_POINTS, seed: int = 0) -> tuple[pd.DataFrame, bool]:
    if len(df) <= n:
        return df, False
    return df.sample(n, random_state=seed), True


def scatter_classes(X: np.ndarray, y: np.ndarray, fig: go.Figure | None = None, names: Sequence[str] | None = None,
                    size: int = 8, opacity: float = 0.85, highlight: np.ndarray | None = None) -> go.Figure:
    """2-D scatter coloured *and* shaped by class."""
    fig = fig or go.Figure()
    classes = np.unique(y)
    for i, c in enumerate(classes):
        mask = y == c
        label = names[i] if names is not None else f"class {c}"
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=label,
                                 marker=dict(color=CLASS_COLORS[i % len(CLASS_COLORS)], symbol=SYMBOLS[i % len(SYMBOLS)],
                                             size=size, opacity=opacity, line=dict(width=0.6, color="white"))))
    if highlight is not None and highlight.any():
        fig.add_trace(go.Scatter(x=X[highlight, 0], y=X[highlight, 1], mode="markers", name="highlighted",
                                 marker=dict(size=size + 8, color="rgba(0,0,0,0)", line=dict(width=2, color="#212529"))))
    fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", legend=dict(orientation="h", y=1.08))
    return fig


def grid_for(X: np.ndarray, steps: int = 150, pad: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "", steps: int = 150,
                      show_proba: bool = True, highlight: np.ndarray | None = None) -> go.Figure:
    """Decision regions for a fitted 2-D classifier (probability shading for binary problems)."""
    xx, yy, grid = grid_for(X, steps)
    classes = np.unique(y)
    fig = go.Figure()
    if len(classes) == 2 and show_proba and hasattr(model, "predict_proba"):
        z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=z, colorscale=REGION_SCALE, zmin=0, zmax=1, opacity=0.9,
                                 contours=dict(start=0, end=1, size=0.1), line=dict(width=0), showscale=True,
                                 colorbar=dict(title="P(y=1)", thickness=12), hoverinfo="skip"))
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=z, showscale=False, hoverinfo="skip",
                                 contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                                 line=dict(width=2.5, color="#212529")))
    else:
        z = model.predict(grid).reshape(xx.shape)
        idx = np.searchsorted(classes, z)
        light = ["#D0EBFF", "#FFE8CC", "#E5DBFF", "#C3FAE8", "#FFF3BF"]
        scale = [[i / max(len(classes) - 1, 1), light[i % len(light)]] for i in range(len(classes))]
        if len(classes) == 1:
            scale = [[0, light[0]], [1, light[0]]]
        fig.add_trace(go.Heatmap(x=xx[0], y=yy[:, 0], z=idx, colorscale=scale, showscale=False, hoverinfo="skip",
                                 opacity=0.9))
    scatter_classes(X, y, fig, highlight=highlight)
    fig.update_layout(title=title, height=440, xaxis=dict(range=[xx.min(), xx.max()]),
                      yaxis=dict(range=[yy.min(), yy.max()]))
    return fig


def regression_fit(x: np.ndarray, y: np.ndarray, grid: np.ndarray, preds: dict[str, np.ndarray],
                   truth: np.ndarray | None = None, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="data",
                               marker=dict(color=PALETTE["muted"], size=7, opacity=0.7)))
    if truth is not None:
        fig.add_trace(go.Scatter(x=grid, y=truth, mode="lines", name="true f(x)",
                                 line=dict(color=PALETTE["teal"], dash="dash", width=2)))
    for i, (name, p) in enumerate(preds.items()):
        fig.add_trace(go.Scatter(x=grid, y=p, mode="lines", name=name,
                                 line=dict(color=SEQUENCE[i % len(SEQUENCE)], width=2.6)))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", legend=dict(orientation="h", y=1.1))
    return fig


def lines(x: Sequence, series: dict[str, Sequence], title: str = "", xaxis: str = "", yaxis: str = "",
          log_x: bool = False, markers: bool = False, dash: dict[str, str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=list(x), y=list(s), mode="lines+markers" if markers else "lines", name=name,
                                 line=dict(color=SEQUENCE[i % len(SEQUENCE)], width=2.4,
                                           dash=(dash or {}).get(name, "solid"))))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, legend=dict(orientation="h", y=1.12))
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def bars(labels: Sequence[str], values: Sequence[float], title: str = "", horizontal: bool = False,
         errors: Sequence[float] | None = None, color: str | Sequence[str] | None = None, text_fmt: str = ".3f") -> go.Figure:
    kw = dict(error_x=dict(type="data", array=list(errors)) if errors is not None and horizontal else None,
              error_y=dict(type="data", array=list(errors)) if errors is not None and not horizontal else None)
    c = color or PALETTE["sky"]
    if horizontal:
        fig = go.Figure(go.Bar(y=list(labels), x=list(values), orientation="h", marker_color=c,
                               text=[f"{v:{text_fmt}}" for v in values], textposition="auto", **{k: v for k, v in kw.items() if v}))
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=c,
                               text=[f"{v:{text_fmt}}" for v in values], textposition="auto", **{k: v for k, v in kw.items() if v}))
    fig.update_layout(title=title, showlegend=False)
    return fig


def heatmap(z: np.ndarray, x: Sequence, y: Sequence, title: str = "", colorscale=None, text_fmt: str | None = ".2f",
            zmid: float | None = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=list(x), y=list(y), colorscale=colorscale or "Blues", zmid=zmid,
                               text=None if text_fmt is None else [[f"{v:{text_fmt}}" for v in row] for row in z],
                               texttemplate="%{text}" if text_fmt else None, hoverongaps=False))
    fig.update_layout(title=title)
    return fig


def confusion_heatmap(cm: np.ndarray, labels: Sequence[str], title: str = "Confusion matrix") -> go.Figure:
    fig = heatmap(cm, [f"pred {l}" for l in labels], [f"true {l}" for l in labels], title=title, text_fmt="d",
                  colorscale=[[0, "#F8F9FF"], [1, "#1C7ED6"]])
    fig.update_layout(yaxis=dict(autorange="reversed"), height=340)
    return fig


def contour_surface(f, xlim: tuple[float, float], ylim: tuple[float, float], steps: int = 120,
                    log_levels: bool = True) -> go.Figure:
    """Contour map of a 2-parameter loss surface f(w1, w2) (vectorised)."""
    w1 = np.linspace(*xlim, steps)
    w2 = np.linspace(*ylim, steps)
    W1, W2 = np.meshgrid(w1, w2)
    Z = f(W1, W2)
    Zp = np.log1p(Z - Z.min()) if log_levels else Z
    fig = go.Figure(go.Contour(x=w1, y=w2, z=Zp, colorscale=[[0, "#E7F5FF"], [0.5, "#F3F0FF"], [1, "#FFF3BF"]],
                               contours=dict(showlines=True), line=dict(color="#ADB5BD", width=0.6),
                               showscale=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title="w₁", yaxis_title="w₂", height=460)
    return fig


def add_path(fig: go.Figure, path: np.ndarray, name: str = "path", color: str | None = None) -> go.Figure:
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", name=name,
                             line=dict(color=color or PALETTE["coral"], width=2),
                             marker=dict(size=5, color=color or PALETTE["coral"])))
    fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="markers", name="start", showlegend=False,
                             marker=dict(size=12, symbol="star", color=PALETTE["purple"])))
    return fig


def interval_plot(names: Sequence[str], est: Sequence[float], lo: Sequence[float], hi: Sequence[float],
                  truth: float | None = None, title: str = "", xaxis: str = "estimate") -> go.Figure:
    """Forest-style plot of estimates with confidence intervals (used by DML labs)."""
    fig = go.Figure()
    for i, (n, e, l, h) in enumerate(zip(names, est, lo, hi)):
        fig.add_trace(go.Scatter(x=[l, h], y=[n, n], mode="lines", showlegend=False,
                                 line=dict(color=SEQUENCE[i % len(SEQUENCE)], width=4)))
        fig.add_trace(go.Scatter(x=[e], y=[n], mode="markers", showlegend=False,
                                 marker=dict(color=SEQUENCE[i % len(SEQUENCE)], size=13, symbol=SYMBOLS[i % len(SYMBOLS)])))
    if truth is not None:
        fig.add_vline(x=truth, line=dict(color=PALETTE["teal"], dash="dash", width=2),
                      annotation_text=f"true = {truth:g}", annotation_position="top")
    fig.update_layout(title=title, xaxis_title=xaxis, height=110 + 60 * len(names), yaxis=dict(autorange="reversed"))
    return fig


def hist(x: Sequence[float], title: str = "", nbins: int = 40, color: str | None = None, name: str = "") -> go.Figure:
    fig = go.Figure(go.Histogram(x=list(x), nbinsx=nbins, marker_color=color or PALETTE["sky"], name=name, opacity=0.85))
    fig.update_layout(title=title, bargap=0.04, showlegend=False)
    return fig


def overlay_hist(series: dict[str, Sequence[float]], title: str = "", nbins: int = 40,
                 norm: str = "probability density") -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        fig.add_trace(go.Histogram(x=list(s), nbinsx=nbins, name=name, opacity=0.55, histnorm=norm,
                                   marker_color=SEQUENCE[i % len(SEQUENCE)]))
    fig.update_layout(barmode="overlay", title=title, legend=dict(orientation="h", y=1.1))
    return fig
