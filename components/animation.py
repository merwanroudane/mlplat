"""Step-based animation controller: Play / Pause / Previous / Next / Reset / Speed.

Autoplay uses ``st.fragment(run_every=...)`` so only the animated area
reruns. When "reduce motion" is on, autoplay is disabled and the learner
steps manually (section 60 and 83 of the brief).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import streamlit as st

from core.state import reduce_motion

_SPEEDS = {"0.5×": 2.0, "1×": 1.1, "2×": 0.6}


def stepper(
    key: str,
    n_steps: int,
    render: Callable[[int], None],
    labels: Sequence[str] | None = None,
) -> None:
    """Render ``render(step)`` for step in [0, n_steps) with media controls."""
    s_step, s_play, s_last = f"anim_{key}_step", f"anim_{key}_play", f"anim_{key}_t"
    st.session_state.setdefault(s_step, 0)
    st.session_state.setdefault(s_play, False)
    st.session_state[s_step] = min(st.session_state[s_step], n_steps - 1)
    motion_ok = not reduce_motion()

    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("السابق", key=f"{key}_prev", icon=":material/skip_previous:"):
            st.session_state[s_step] = max(0, st.session_state[s_step] - 1)
            st.session_state[s_play] = False
        if motion_ok:
            if st.session_state[s_play]:
                if st.button("إيقاف", key=f"{key}_pause", icon=":material/pause:"):
                    st.session_state[s_play] = False
            elif st.button("تشغيل", key=f"{key}_play", icon=":material/play_arrow:", type="primary"):
                if st.session_state[s_step] >= n_steps - 1:
                    st.session_state[s_step] = 0
                st.session_state[s_play] = True
                st.session_state[s_last] = time.monotonic()
        if st.button("التالي", key=f"{key}_next", icon=":material/skip_next:"):
            st.session_state[s_step] = min(n_steps - 1, st.session_state[s_step] + 1)
            st.session_state[s_play] = False
        if st.button("إعادة", key=f"{key}_reset", icon=":material/replay:"):
            st.session_state[s_step] = 0
            st.session_state[s_play] = False
        speed = "1×"
        if motion_ok:
            speed = st.segmented_control("السرعة", list(_SPEEDS), default="1×", key=f"{key}_speed",
                                         label_visibility="collapsed") or "1×"
    if not motion_ok:
        st.caption("وضع تقليل الحركة مفعّل: التشغيل التلقائي معطّل، استخدم «التالي» و«السابق».")

    interval = _SPEEDS[speed]
    playing = st.session_state[s_play] and motion_ok

    @st.fragment(run_every=interval if playing else None)
    def _frame() -> None:
        if st.session_state[s_play] and motion_ok:
            now = time.monotonic()
            if now - st.session_state.get(s_last, 0.0) >= interval * 0.9:
                st.session_state[s_last] = now
                if st.session_state[s_step] < n_steps - 1:
                    st.session_state[s_step] += 1
                else:
                    st.session_state[s_play] = False
                    st.rerun()
        step = st.session_state[s_step]
        st.progress((step + 1) / n_steps,
                     text=f"الخطوة {step + 1} من {n_steps}" + (f" — {labels[step]}" if labels else ""))
        render(step)

    _frame()
