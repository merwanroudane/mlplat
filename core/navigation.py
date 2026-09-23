"""Build st.navigation from the curriculum registry (single source of truth)."""

from __future__ import annotations

import streamlit as st

from core.curriculum import GROUPS, MODULES


def build_pages() -> dict[str, list]:
    sections: dict[str, list] = {label: [] for label in GROUPS.values()}
    for m in MODULES:
        sections[GROUPS[m.group]].append(
            st.Page(m.file, title=m.title_ar, icon=m.icon, url_path=m.id, default=(m.id == "home"))
        )
    return {k: v for k, v in sections.items() if v}
