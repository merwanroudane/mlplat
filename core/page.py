"""Page template.

Every module page starts with ``page_header`` (title, description,
objectives, prerequisites, difficulty, study time) and ends with
``page_footer`` (takeaways, common mistakes, quiz, exercises, next module,
author identity).
"""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st

from config import APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN, DATA_SCIENCE_PLATFORM_URL
from core.curriculum import Module, get_module, next_module
from core.state import is_complete, mark_lab, mark_visited, set_complete, toggle_bookmark

_DIFF_COLOR = {"مبتدئ": "green", "متوسط": "blue", "متقدم": "violet", "بحثي": "orange"}


def page_link(mid: str, label: str | None = None, icon: str | None = None) -> None:
    m = get_module(mid)
    st.page_link(m.file, label=label or m.label, icon=icon or m.icon)


def page_header(module_id: str) -> Module:
    m = get_module(module_id)
    mark_visited(module_id)
    if m.kind == "lab" or m.lab:
        mark_lab(module_id)
    st.title(m.title_ar)
    st.markdown(f"##### :gray[{m.title_en}]")
    st.markdown(m.description)
    if m.kind in ("resource", "start") and not m.objectives:
        return m
    with st.container(key=f"ml-modinfo-{module_id}"):
        c1, c2 = st.columns([3, 2])
        with c1:
            if m.objectives:
                st.markdown("**:material/flag: أهداف التعلّم · Learning objectives**")
                st.markdown("\n".join(f"- {o}" for o in m.objectives))
            else:
                st.markdown("**:material/flag: ماذا ستفعل هنا؟**")
                st.markdown(m.description)
        with c2:
            color = _DIFF_COLOR.get(m.difficulty, "gray")
            st.markdown(f":{color}-badge[:material/signal_cellular_alt: {m.difficulty}] "
                        f":gray-badge[:material/schedule: {m.minutes} دقيقة] "
                        + (f":blue-badge[:material/science: {m.lab}]" if m.lab else ""))
            if m.prereqs:
                st.markdown("**:material/route: المتطلبات السابقة · Prerequisites**")
                for p in m.prereqs:
                    pm = get_module(p)
                    st.page_link(pm.file, label=pm.title_ar, icon=":material/done:" if is_complete(p) else pm.icon)
            else:
                st.caption("لا متطلبات سابقة.")
            bookmarked = module_id in st.session_state.get("bookmarks", set())
            st.button("إزالة من المفضلة" if bookmarked else "أضف إلى المفضلة",
                      key=f"bm_{module_id}", icon=":material/bookmark:",
                      on_click=toggle_bookmark, args=(module_id,), type="tertiary")
    return m


def key_takeaways(items: Iterable[str]) -> None:
    st.markdown("### :material/checklist: الخلاصة · Key takeaways")
    st.markdown("\n".join(f"- {t}" for t in items))


def page_footer(module_id: str, takeaways: Iterable[str] | None = None,
                mistakes: Iterable[str] | None = None, quiz: bool = True, exercises: bool = True) -> None:
    from components.callouts import mistakes as mistakes_box
    from components.quiz import render_exercises, render_quiz

    st.divider()
    if takeaways:
        key_takeaways(takeaways)
    if mistakes:
        mistakes_box(mistakes)
    if quiz:
        render_quiz(module_id)
    if exercises:
        render_exercises(module_id)

    m = get_module(module_id)
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if m.tracked:
            done = st.toggle("أنهيت هذه الوحدة", value=is_complete(module_id), key=f"done_{module_id}")
            set_complete(module_id, done)
    with c2:
        nxt = next_module(module_id)
        if nxt:
            st.page_link(nxt.file, label=f"الوحدة التالية: {nxt.title_ar}", icon=":material/arrow_back:")
    footer()


def dsplat_link(topic: str = "") -> None:
    """Pointer back to the companion Data Science platform for prerequisite skills."""
    extra = f" (خصوصًا: {topic})" if topic else ""
    st.caption(f":material/link: هل تحتاج مراجعة مهارات البيانات{extra}؟ راجع منصة علم البيانات المرافقة: "
               f"[DSplat]({DATA_SCIENCE_PLATFORM_URL})")


def footer() -> None:
    st.html(
        f'<div class="ml-footer">{html.escape(APP_NAME_AR)} · <span class="ml-en">{html.escape(APP_NAME_EN)}</span>'
        f"<br>تطوير: {html.escape(APP_AUTHOR_AR)} · <span class=\"ml-en\">Developed by {html.escape(APP_AUTHOR_EN)}</span></div>"
    )
