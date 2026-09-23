"""The algorithm template (section 11), rendered from content metadata.

Pages author the math, geometry, training algorithm, labs and code by hand;
this component renders the profile items that are shared with the Explorer
and the Selector, so they are written once and never drift apart.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from content.algorithm_metadata import get_algorithm
from content.hyperparameters import CHECKED, for_estimator
from core.state import at_least

_TEMPLATE_ITEMS = (
    "1 Problem solved", "2 Intuition", "3 Mathematical formulation", "4 Notation", "5 Geometry", "6 Loss / objective",
    "7 Training algorithm", "8 Learned parameters", "9 Hyperparameters", "10 Assumptions",
    "11 Preprocessing requirements", "12 Scaling requirement", "13 Categorical handling", "14 Missing-value behavior",
    "15 Computational complexity", "16 Strengths", "17 Weaknesses", "18 Failure modes", "19 Diagnostics",
    "20 Interpretation", "21 From-scratch implementation", "22 Official-library implementation", "23 Interactive lab",
    "24 Parameter playground", "25 Exercises", "26 Quiz", "27 References",
)

_SCALING_AR = {"required": "إلزامي", "recommended": "مستحسن", "not needed": "غير مطلوب"}


def algorithm_profile(algo_id: str, expanded: bool = False) -> None:
    """Profile card: problem, intuition, learned parameters, assumptions, preprocessing, scaling,
    categorical & missing handling, complexity, strengths/weaknesses, failure modes, diagnostics, interpretation."""
    a = get_algorithm(algo_id)
    with st.container(key=f"ml-algoprofile-{algo_id}"):
        st.markdown(f"#### :material/badge: بطاقة الخوارزمية · {a.name}")
        st.markdown(f":blue-badge[{' / '.join(a.tasks)}] "
                    f":violet-badge[{'خطي' if a.linear else 'غير خطي'}] "
                    f":gray-badge[{'معلمي' if a.parametric else 'غير معلمي'}] "
                    f":orange-badge[القياس: {_SCALING_AR[a.scaling]}] "
                    + (":green-badge[فئات أصلية] " if a.categorical_native else "")
                    + (":green-badge[NaN أصلي] " if a.missing_native else "")
                    + (":blue-badge[تجميعي] " if a.ensemble else "")
                    + f"`{a.estimator}`")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**المسألة التي يحلها:** {a.problem}")
            st.markdown(f"**الحدس:** {a.intuition}")
            st.markdown(f"**ما الذي يُتعلَّم؟** {a.learned}")
        with c2:
            st.markdown(f"**الافتراضات:** {a.assumptions}")
            st.markdown(f"**المعالجة المسبقة:** {a.preprocessing}")
            st.markdown(f"**الفئات:** {a.categorical}  \n**القيم المفقودة:** {a.missing}  \n**التعقيد:** {a.complexity}")
        with st.expander("المزايا، العيوب، أنماط الفشل، التشخيص، والتفسير", expanded=expanded,
                         icon=":material/fact_check:"):
            c3, c4 = st.columns(2)
            c3.markdown("**المزايا · Strengths**\n" + "\n".join(f"- {s}" for s in a.strengths))
            c3.markdown("**العيوب · Weaknesses**\n" + "\n".join(f"- {s}" for s in a.weaknesses))
            c4.markdown("**أنماط الفشل · Failure modes**\n" + "\n".join(f"- {s}" for s in a.failure_modes))
            c4.markdown("**التشخيص · Diagnostics**\n" + "\n".join(f"- {s}" for s in a.diagnostics))
            st.markdown(f"**التفسير · Interpretation:** {a.interpretation}")
            if a.references:
                st.markdown("**مراجع أساسية:** " + "؛ ".join(a.references))


def hyperparameter_table(algo: str, title: str | None = None) -> None:
    """Hyperparameters of one estimator from the central metadata, with verified defaults."""
    rows = for_estimator(algo)
    if not rows:
        return
    st.markdown(title or f"#### :material/tune: المعاملات الفائقة لـ `{algo}`")
    detailed = at_least("advanced")
    data = []
    for h in rows:
        default = repr(h.default) + (f" → {h.resolved}" if h.resolved else "")
        row = {"parameter": h.parameter, "default (verified)": default, "المعنى": h.meaning,
               "عند الزيادة ↑": h.increase, "عند النقصان ↓": h.decrease}
        if detailed:
            row.update({"المدى / الخيارات": h.allowed, "التفاعلات": h.interactions, "الزمن": h.runtime,
                        "خطر Over/Underfitting": h.risk})
        data.append(row)
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")
    notes = [f"- **{h.parameter}**: {h.note}" for h in rows if h.note]
    if notes:
        st.markdown("\n".join(notes))
    lib = rows[0].library
    st.caption(f"القيم الافتراضية مقروءة من {lib} {rows[0].verified_version} المثبتة بتاريخ {CHECKED} ومُختبرة آليًا "
               f"(tests/test_hyperparameters.py). المصدر: [{lib} docs]({rows[0].url})"
               + ("" if detailed else " · اختر المستوى «متقدم» لعرض المدى والتفاعلات والزمن."))


def template_checklist(covered: dict[int, str]) -> None:
    """Show where each of the 27 template items lives on the page (teacher's map)."""
    with st.expander("خريطة القالب: أين يوجد كل عنصر من العناصر الـ27 في هذه الصفحة؟", icon=":material/checklist:"):
        lines = []
        for i, item in enumerate(_TEMPLATE_ITEMS, start=1):
            where = covered.get(i, "بطاقة الخوارزمية أعلاه")
            lines.append(f"| {item} | {where} |")
        st.markdown("| العنصر | الموضع |\n| --- | --- |\n" + "\n".join(lines))
