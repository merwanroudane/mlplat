"""Quizzes and exercises.

Question bank lives in content/quizzes.py and content/exercises.py so
authors can add questions without touching UI code. After submission every
question shows *why* the answer is right, not only correct/incorrect.
"""

from __future__ import annotations

import streamlit as st

from core.state import record_exercise, record_quiz
from core.state import next_key

_TYPE_LABEL = {
    "mcq": "اختيار من متعدد",
    "tf": "صح / خطأ",
    "scenario": "سيناريو",
    "code": "تفسير كود",
    "chart": "تفسير رسم",
    "hyper": "استدلال على المعاملات الفائقة",
    "method": "اختيار الطريقة",
}


def render_quiz(module_id: str) -> None:
    from content.quizzes import QUIZZES

    questions = QUIZZES.get(module_id)
    if not questions:
        return
    result_key = f"quiz_{module_id}_result"
    st.markdown("### :material/quiz: اختبر فهمك · Quiz")

    with st.form(key=f"quiz_form_{module_id}"):
        for i, q in enumerate(questions):
            st.markdown(f"**س{i + 1}.** {q['q']}  \n:gray-badge[{_TYPE_LABEL.get(q.get('type', 'mcq'))}]")
            if q.get("code"):
                st.code(q["code"], language="python")
            if q.get("chart") is not None:
                st.plotly_chart(q["chart"](), key=next_key("quizfig"))
            st.radio(
                f"إجابة السؤال {i + 1}",
                options=list(range(len(q["options"]))),
                format_func=lambda j, opts=q["options"]: opts[j],
                index=None,
                key=f"quiz_{module_id}_{i}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("تحقّق من الإجابات", type="primary", icon=":material/check_circle:")

    if submitted:
        answers = [st.session_state.get(f"quiz_{module_id}_{i}") for i in range(len(questions))]
        st.session_state[result_key] = answers
        score = sum(1 for a, q in zip(answers, questions) if a == q["answer"])
        record_quiz(module_id, score, len(questions))

    answers = st.session_state.get(result_key)
    if answers is None:
        return
    score = sum(1 for a, q in zip(answers, questions) if a == q["answer"])
    st.metric("النتيجة", f"{score} / {len(questions)}")
    for i, (a, q) in enumerate(zip(answers, questions)):
        correct = q["options"][q["answer"]]
        if a is None:
            st.warning(f"س{i + 1}: لم تُجب. الإجابة الصحيحة: **{correct}**  \n{q['explain']}")
        elif a == q["answer"]:
            st.success(f"س{i + 1}: إجابة صحيحة.  \n{q['explain']}", icon=":material/check_circle:")
        else:
            st.error(f"س{i + 1}: اخترت «{q['options'][a]}». الصحيح: **{correct}**  \n{q['explain']}",
                     icon=":material/cancel:")
    if st.button("أعد المحاولة", key=f"quiz_retry_{module_id}", icon=":material/replay:"):
        st.session_state.pop(result_key, None)
        for i in range(len(questions)):
            st.session_state.pop(f"quiz_{module_id}_{i}", None)
        st.rerun()


def render_exercises(module_id: str) -> None:
    from content.exercises import EXERCISES

    items = EXERCISES.get(module_id)
    if not items:
        return
    st.markdown("### :material/edit_note: تمارين تطبيقية · Practice")
    labels = {"beginner": "تمرين مبتدئ", "intermediate": "تمرين متوسط", "research": "تحدٍّ بحثي"}
    for lvl, ex in items.items():
        with st.expander(f"{labels.get(lvl, lvl)}: {ex['title']}", icon=":material/assignment:"):
            st.markdown(ex["task"])
            if ex.get("hint"):
                st.caption(f"تلميح: {ex['hint']}")
            st.text_area("إجابتك (للمراجعة الذاتية، لا تُرسل إلى أي مكان)", key=f"ex_{module_id}_{lvl}",
                         height=110)
            show = st.toggle("اعرض إجابة نموذجية", key=f"ex_show_{module_id}_{lvl}")
            if show:
                st.markdown(ex["solution"])
                record_exercise(module_id, lvl)
