"""Callout boxes: Researcher Note, Real-world checks, Common mistakes, Why?.

Each callout is a native ``st.container`` whose key starts with ``ml-<kind>``;
assets/css/theme.css styles those keys. Content is ordinary Markdown, so
bold text, inline code and lists render natively.
"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from core.state import next_key

_TITLES = {
    "note": ":material/lightbulb: ملاحظة",
    "researcher": ":material/biotech: ملاحظة الباحث · Researcher Note",
    "realworld": ":material/work: في المشاريع الحقيقية: ماذا أفحص؟ · In real projects, what should I check?",
    "mistake": ":material/warning: أخطاء شائعة · Common Mistakes",
    "why": ":material/help_center: لماذا؟ · Why?",
    "tip": ":material/tips_and_updates: فكرة بديهية · Intuition",
    "warning": ":material/report: تنبيه",
    "domain": ":material/psychology: الإحصاء + معرفة المجال · Domain Knowledge",
    "definition": ":material/menu_book: تعريف · Definition",
    "causal": ":material/device_hub: تنبيه سببي · Causal caution",
}


def _box(kind: str, body: str | Iterable[str], title: str | None = None) -> None:
    with st.container(key=next_key(f"ml-{kind}")):
        st.markdown(f"**{title or _TITLES[kind]}**")
        if isinstance(body, str):
            st.markdown(body)
        else:
            st.markdown("\n".join(f"- {item}" for item in body))


def note(body: str, title: str | None = None) -> None:
    _box("note", body, title)


def researcher_note(body: str | Iterable[str]) -> None:
    _box("researcher", body)


def real_world(items: Iterable[str]) -> None:
    _box("realworld", list(items))


def mistakes(items: Iterable[str]) -> None:
    _box("mistake", list(items))


def intuition(body: str) -> None:
    _box("tip", body)


def warning(body: str, title: str | None = None) -> None:
    _box("warning", body, title)


def domain(body: str | Iterable[str]) -> None:
    _box("domain", body)


def why(recommendation: str, reason: str) -> None:
    """A recommendation always travels with its justification."""
    with st.container(key=next_key("ml-why")):
        st.markdown(f"**التوصية:** {recommendation}")
        st.markdown(f"**لماذا؟** {reason}")


def definition(term: str, body: str) -> None:
    """A formal definition box: the term in bold, then the definition."""
    _box("definition", body, f":material/menu_book: تعريف · {term}")


def causal_caution(body: str) -> None:
    """Reminder that predictive evidence is not causal evidence."""
    _box("causal", body)
