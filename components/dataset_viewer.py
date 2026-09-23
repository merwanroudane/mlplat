"""Dataset picker and documentation card, shared by labs and lessons."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from utils.datasets import DATASET_INFO, load_dataset


def dataset_picker(key: str, default: str = "classification", allowed: Sequence[str] | None = None,
                   label: str = "اختر مجموعة البيانات") -> tuple[str, pd.DataFrame]:
    options = list(allowed or DATASET_INFO)
    index = options.index(default) if default in options else 0
    name = st.selectbox(label, options, index=index, format_func=lambda n: DATASET_INFO[n].title, key=key)
    return name, load_dataset(name)


def dataset_card(name: str, show_head: bool = True) -> None:
    info = DATASET_INFO[name]
    df = load_dataset(name)
    with st.expander(f"عن هذه البيانات: {info.title}", icon=":material/dataset:"):
        st.markdown(f"**الغرض:** {info.purpose}  \n**التصميم:** {info.design}  \n"
                    f"**المهمة:** `{info.task}` · **الهدف:** `{info.target}` · **الأبعاد:** {df.shape[0]:,} × {df.shape[1]}")
        if info.variables:
            st.markdown("**المتغيرات:**\n" + "\n".join(f"- `{k}`: {v}" for k, v in info.variables.items()))
        if info.truth:
            st.markdown(f"**الحقيقة المعروفة (Ground truth):** `{info.truth}`")
        if show_head:
            st.dataframe(df.head(8), width="stretch", hide_index=True)
