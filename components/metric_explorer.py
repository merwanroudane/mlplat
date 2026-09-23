"""Threshold-driven metric explorer: move the threshold, watch every metric and the cost change."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.theme import PALETTE
from utils.metrics import classification_report_dict, expected_cost
from utils.plotting import confusion_heatmap, plot


def threshold_explorer(y: np.ndarray, proba: np.ndarray, key: str = "thr", default_cost_fp: float = 1.0,
                       default_cost_fn: float = 5.0) -> float:
    """Interactive threshold slider with confusion matrix, metrics and a cost curve. Returns the chosen threshold."""
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("العتبة (threshold)", 0.01, 0.99, 0.50, 0.01, key=f"{key}_t")
    cfp = c2.number_input("تكلفة False Positive", 0.0, 1000.0, default_cost_fp, 0.5, key=f"{key}_cfp")
    cfn = c3.number_input("تكلفة False Negative", 0.0, 1000.0, default_cost_fn, 0.5, key=f"{key}_cfn")
    yhat = (proba >= thr).astype(int)
    rep = classification_report_dict(y, yhat)
    left, right = st.columns([1, 1.3])
    with left:
        cm = np.array([[rep["TN"], rep["FP"]], [rep["FN"], rep["TP"]]])
        plot(confusion_heatmap(cm, ["0", "1"], title=f"Confusion matrix @ threshold {thr:.2f}"))
    with right:
        with st.container(horizontal=True):
            st.metric("Precision", f"{rep['precision']:.3f}", border=True)
            st.metric("Recall", f"{rep['recall']:.3f}", border=True)
            st.metric("Specificity", f"{rep['specificity']:.3f}", border=True)
        with st.container(horizontal=True):
            st.metric("F1", f"{rep['f1']:.3f}", border=True)
            st.metric("Balanced acc.", f"{rep['balanced_accuracy']:.3f}", border=True)
            st.metric("Cost / case", f"{expected_cost(y, yhat, cfp, cfn):.3f}", border=True)
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y, (proba >= t).astype(int), cfp, cfn) for t in grid]
    prec = [classification_report_dict(y, (proba >= t).astype(int))["precision"] for t in grid]
    rec = [classification_report_dict(y, (proba >= t).astype(int))["recall"] for t in grid]
    best = float(grid[int(np.argmin(costs))])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=prec, name="precision", line=dict(color=PALETTE["sky"], width=2.4)))
    fig.add_trace(go.Scatter(x=grid, y=rec, name="recall", line=dict(color=PALETTE["coral"], width=2.4, dash="dot")))
    fig.add_trace(go.Scatter(x=grid, y=np.array(costs) / max(max(costs), 1e-9), name="cost (scaled)",
                             line=dict(color=PALETTE["purple"], width=2.4, dash="dash")))
    fig.add_vline(x=thr, line=dict(color="#212529", width=1.5), annotation_text="current")
    fig.add_vline(x=best, line=dict(color=PALETTE["teal"], width=1.5, dash="dash"), annotation_text="min cost",
                  annotation_position="bottom right")
    fig.update_layout(title="Precision, recall and cost vs threshold", xaxis_title="threshold", height=360,
                      legend=dict(orientation="h", y=1.15))
    plot(fig)
    theory = cfp / (cfp + cfn) if (cfp + cfn) > 0 else 0.5
    st.caption(f"العتبة ذات أقل تكلفة على هذه البيانات ≈ {best:.2f}. إن كانت الاحتمالات معايَرة، فالعتبة المثلى نظريًا = "
               f"C_FP / (C_FP + C_FN) = {theory:.2f} (Elkan, 2001). اختر العتبة على بيانات تحقق، لا على Test.")
    return thr


def metrics_table(y: np.ndarray, proba: np.ndarray, thresholds=(0.3, 0.5, 0.7)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        r = classification_report_dict(y, (proba >= t).astype(int))
        rows.append({"threshold": t, **{k: r[k] for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc")}})
    return pd.DataFrame(rows)
