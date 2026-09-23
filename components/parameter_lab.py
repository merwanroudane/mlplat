"""Reusable Parameter Playground (section 69).

Shows controls → current values → fitted decision boundary → train and
validation scores → runtime → model complexity. Any page can call
``parameter_playground(model_key, key=...)`` to embed it.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from components.decision_boundary import boundary_chart
from core.state import log_experiment
from utils.datasets import toy_2d
from utils.models import PLAYGROUND, ParamSpec, build_playground, model_complexity


def _widget(spec: ParamSpec, key: str):
    k = f"{key}_{spec.name}"
    if spec.kind == "int":
        return st.slider(spec.name, int(spec.low), int(spec.high), int(spec.default), key=k, help=spec.help or None)
    if spec.kind == "float":
        return st.slider(spec.name, float(spec.low), float(spec.high), float(spec.default),
                         step=(spec.high - spec.low) / 100, key=k, help=spec.help or None)
    if spec.kind == "log":
        lo, hi = np.log10(spec.low), np.log10(spec.high)
        exps = np.round(np.linspace(lo, hi, int((hi - lo) * 4) + 1), 3)
        options = [float(f"{10 ** e:.3g}") for e in exps]
        default = min(options, key=lambda v: abs(np.log10(v) - np.log10(spec.default)))
        return st.select_slider(spec.name, options=options, value=default, key=k, help=spec.help or None,
                                format_func=lambda v: f"{v:g}")
    if spec.kind == "bool":
        return st.toggle(spec.name, value=bool(spec.default), key=k, help=spec.help or None)
    return st.selectbox(spec.name, list(spec.options), index=list(spec.options).index(spec.default), key=k,
                        help=spec.help or None, format_func=str)


@st.cache_data(show_spinner=False, max_entries=64)
def _fit(model_key: str, params_items: tuple, dataset: str, noise: float, n: int, seed: int):
    X, y = toy_2d(dataset, n=n, noise=noise, seed=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = build_playground(model_key, dict(params_items))
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    return model, X_tr, X_va, y_tr, y_va, fit_s


def parameter_playground(model_key: str | None = None, key: str = "pg", dataset: str = "moons",
                         allow_model_choice: bool = False, log: bool = True) -> None:
    with st.container(border=True):
        st.markdown("**:material/tune: ساحة المعاملات · Parameter Playground**")
        c_ctrl, c_plot = st.columns([1, 2.2])
        with c_ctrl:
            if allow_model_choice or model_key is None:
                model_key = st.selectbox("الخوارزمية", list(PLAYGROUND), format_func=lambda k: PLAYGROUND[k].label,
                                         key=f"{key}_model", index=list(PLAYGROUND).index(model_key or "svc"))
            spec = PLAYGROUND[model_key]
            ds = st.segmented_control("البيانات", ["moons", "circles", "linear", "xor"], default=dataset,
                                      key=f"{key}_ds", required=True)
            noise = st.slider("الضجيج noise", 0.0, 0.6, 0.25, 0.05, key=f"{key}_noise")
            n = st.slider("حجم العينة n", 60, 600, 300, 20, key=f"{key}_n")
            params = {p.name: _widget(p, f"{key}_{model_key}") for p in spec.params}
        model, X_tr, X_va, y_tr, y_va, fit_s = _fit(model_key, tuple(sorted(params.items(), key=lambda kv: kv[0])),
                                                    ds, noise, n, 0)
        tr, va = model.score(X_tr, y_tr), model.score(X_va, y_va)
        with c_plot:
            boundary_chart(model, np.r_[X_tr, X_va], np.r_[y_tr, y_va],
                           title=f"{spec.label}: decision boundary", highlight_val=len(X_tr))
            with st.container(horizontal=True):
                st.metric("Train accuracy", f"{tr:.3f}", border=True)
                st.metric("Validation accuracy", f"{va:.3f}", f"{va - tr:+.3f} gap", delta_color="off", border=True)
                st.metric("Fit time", f"{fit_s * 1000:.0f} ms", border=True)
            st.caption(f"Model complexity: {model_complexity(model)} · "
                       f"current parameters: `{', '.join(f'{k}={v}' for k, v in params.items())}`")
            gap = tr - va
            if gap > 0.08:
                st.warning(f"فجوة التدريب−التحقق = {gap:.2f}: علامة Overfitting. جرّب تنظيمًا أقوى أو نموذجًا أبسط.",
                           icon=":material/trending_down:")
            elif tr < 0.8 and va < 0.8:
                st.info("كلا الدرجتين منخفض: علامة Underfitting. جرّب نموذجًا أكثر مرونة أو تنظيمًا أضعف.",
                        icon=":material/trending_flat:")
        if log and st.button("سجّل هذه التجربة في سجل التجارب", key=f"{key}_log", icon=":material/history:",
                             type="tertiary"):
            log_experiment("Parameter Playground", spec.label, params, {"train_acc": tr, "val_acc": va},
                           seed=0, dataset=f"{ds} (n={n}, noise={noise})", split="holdout 70/30 stratified")
            st.toast("سُجّلت التجربة. راجع صفحة قابلية إعادة الإنتاج.", icon=":material/check:")
