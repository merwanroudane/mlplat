"""Sampler Visualizer: what an imbalanced-learn sampler adds, duplicates and removes on a 2-D toy problem."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.registry import missing_notice, optional
from core.theme import PALETTE
from utils.imbalance import SAMPLERS, imbalance_ratio, make_imbalanced_2d, make_sampler, resample_roles
from utils.plotting import plot


@st.cache_data(show_spinner=False, max_entries=32)
def _resample(name: str, frac: float, sep: float, clusters: int, k: int, seed: int):
    X, y = make_imbalanced_2d(n=500, minority_frac=frac, separation=sep, n_clusters=clusters, seed=seed)
    k_eff = int(min(k, max(1, (y == 1).sum() - 1)))
    Xr, yr, roles, removed = resample_roles(make_sampler(name, seed=seed, k=k_eff), X, y)
    return X, y, Xr, yr, roles, removed, k_eff


def sampler_visualizer(key: str, families: Sequence[str], default: str) -> None:
    """Controls → resample → scatter (original / synthetic / duplicated / removed) → counts table."""
    if optional("imblearn") is None:
        missing_notice("imblearn")
        return
    names = [n for n, v in SAMPLERS.items() if v[2] in families]
    c1, c2 = st.columns([1.4, 1])
    name = c1.selectbox("Sampler", names, index=names.index(default), key=f"{key}_name")
    k = c2.slider("k (الجيران)", 1, 9, 5, key=f"{key}_k",
                  help="k_neighbors لعائلة SMOTE أو n_neighbors لـADASYN وNearMiss وENN. يُتجاهل للطرق التي لا تستخدمه.")
    c3, c4, c5 = st.columns(3)
    frac = c3.select_slider("نسبة الأقلية", [0.02, 0.05, 0.1, 0.2], value=0.05, key=f"{key}_frac")
    sep = c4.slider("المسافة بين الفئتين (أقل = تداخل أكبر)", 0.5, 4.0, 2.0, 0.5, key=f"{key}_sep")
    clusters = c5.segmented_control("عناقيد الأقلية", [1, 2, 3], default=1, key=f"{key}_cl") or 1

    try:
        X, y, Xr, yr, roles, removed, k_eff = _resample(name, frac, sep, clusters, k, 0)
    except (ValueError, RuntimeError) as err:
        st.warning(f"رفض {name} هذه الإعدادات: `{err}`. مثال: ADASYN لا يولّد شيئًا إذا لم يكن لأي مثال أقلية جار من "
                   "الأغلبية (فئتان منفصلتان تمامًا)، وBorderline/SVM-SMOTE قد لا تجد أمثلة حدودية. قرّب الفئتين أو غيّر k.",
                   icon=":material/info:")
        return
    fig = go.Figure()
    kept0 = (y == 0) & ~removed
    fig.add_trace(go.Scatter(x=X[kept0, 0], y=X[kept0, 1], mode="markers", name="majority (kept)",
                             marker=dict(color=PALETTE["sky"], symbol="circle", size=6, opacity=0.55)))
    if removed.any():
        fig.add_trace(go.Scatter(x=X[removed, 0], y=X[removed, 1], mode="markers", name="removed",
                                 marker=dict(color=PALETTE["muted"], symbol="x-thin", size=8, line=dict(width=1.5,
                                                                                                     color=PALETTE["muted"]))))
    orig1 = (y == 1) & ~removed
    fig.add_trace(go.Scatter(x=X[orig1, 0], y=X[orig1, 1], mode="markers", name="minority (original)",
                             marker=dict(color=PALETTE["coral"], symbol="diamond", size=10, line=dict(width=1, color="white"))))
    syn = roles == "synthetic"
    if syn.any():
        fig.add_trace(go.Scatter(x=Xr[syn, 0], y=Xr[syn, 1], mode="markers", name="synthetic",
                                 marker=dict(color=np.where(yr[syn] == 1, PALETTE["amber"], PALETTE["purple"]).tolist(),
                                             symbol="star", size=8, line=dict(width=0.5, color="#212529"))))
    dup = roles == "duplicate"
    if dup.any():
        fig.add_trace(go.Scatter(x=Xr[dup, 0], y=Xr[dup, 1], mode="markers", name="duplicate (ring)",
                                 marker=dict(color="rgba(0,0,0,0)", symbol="circle-open", size=15,
                                             line=dict(width=1.5, color=PALETTE["coral"]))))
    fig.update_layout(title=f"{name}: {len(y)} → {len(yr)} rows", xaxis_title="x₁", yaxis_title="x₂", height=460,
                      legend=dict(orientation="h", y=1.1))
    plot(fig)
    counts = pd.DataFrame({
        "": ["before", "after"],
        "majority (0)": [int((y == 0).sum()), int((yr == 0).sum())],
        "minority (1)": [int((y == 1).sum()), int((yr == 1).sum())],
        "IR": [imbalance_ratio(y), imbalance_ratio(yr)],
    })
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(counts.round(2), hide_index=True, width="stretch")
    c2.markdown(f"- جديدة اصطناعية: **{int(syn.sum())}**\n- مكررة: **{int(dup.sum())}**\n- محذوفة: **{int(removed.sum())}**"
                + (f"\n- k المستخدم: {k_eff}" if k_eff != k else ""))
