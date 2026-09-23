"""Model comparison on identical folds, reported with uncertainty (never a single 'winner')."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

from core.theme import SEQUENCE
from utils.plotting import plot


def compare_models(models: dict, X, y, scoring: str = "roc_auc", cv: int = 5, classification: bool = True,
                   seed: int = 0) -> pd.DataFrame:
    """Cross-validate every model on the *same* folds; returns per-fold scores (long format)."""
    splitter = (StratifiedKFold if classification else KFold)(cv, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y))
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        res = cross_validate(model, X, y, cv=folds, scoring=scoring, return_train_score=True)
        elapsed = time.perf_counter() - t0
        for k, (tr, te) in enumerate(zip(res["train_score"], res["test_score"])):
            rows.append({"model": name, "fold": k + 1, "train": tr, "validation": te, "seconds": elapsed / cv})
    return pd.DataFrame(rows)


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby("model")
    out = pd.DataFrame({
        "mean validation": g["validation"].mean(),
        "± SD across folds": g["validation"].std(ddof=1),
        "mean train": g["train"].mean(),
        "train − val gap": g["train"].mean() - g["validation"].mean(),
        "seconds / fold": g["seconds"].mean(),
    }).sort_values("mean validation", ascending=False)
    return out


def fold_plot(scores: pd.DataFrame, metric_name: str) -> None:
    """Paired fold-wise plot: lines join the same fold across models, so paired differences are visible."""
    fig = go.Figure()
    models = list(dict.fromkeys(scores["model"]))
    for i, m in enumerate(models):
        s = scores[scores.model == m]
        fig.add_trace(go.Box(y=s["validation"], name=m, boxpoints="all", jitter=0.3, pointpos=0,
                             marker_color=SEQUENCE[i % len(SEQUENCE)], line_color=SEQUENCE[i % len(SEQUENCE)]))
    for f in sorted(scores["fold"].unique()):
        s = scores[scores.fold == f].set_index("model").reindex(models)
        fig.add_trace(go.Scatter(x=models, y=s["validation"], mode="lines", line=dict(color="rgba(92,103,125,0.25)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"Validation {metric_name} per fold (lines join the same fold)", showlegend=False,
                      yaxis_title=metric_name, height=420)
    plot(fig)


def paired_difference(scores: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """Mean and SD of fold-wise differences a − b (folds are shared, so differences are paired)."""
    sa = scores[scores.model == a].sort_values("fold")["validation"].to_numpy()
    sb = scores[scores.model == b].sort_values("fold")["validation"].to_numpy()
    d = sa - sb
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else float("nan")


def render_comparison(scores: pd.DataFrame, metric_name: str) -> None:
    st.dataframe(summary_table(scores).style.format("{:.4f}"), width="stretch")
    fold_plot(scores, metric_name)
    models = list(dict.fromkeys(scores["model"]))
    if len(models) >= 2:
        top = summary_table(scores).index[:2].tolist()
        mean, sd = paired_difference(scores, top[0], top[1])
        k = scores["fold"].nunique()
        st.caption(f"الفرق المزدوج بين أفضل نموذجين ({top[0]} − {top[1]}): {mean:+.4f} ± {sd:.4f} (SD عبر {k} طيات). "
                   "إن كان الفرق أصغر من تقلّبه عبر الطيات فلا تعلن أفضلية؛ الطيات غير مستقلة تمامًا لأن مجموعات التدريب متداخلة "
                   "(Nadeau & Bengio, 2003)، فالتقلب الحقيقي أكبر مما يبدو.")
    if np.isfinite(scores["validation"]).all():
        return
    st.warning("بعض الطيات أعادت قيمًا غير صالحة (NaN).")
