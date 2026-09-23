import pandas as pd
import streamlit as st

from content.algorithm_metadata import ALGORITHMS, ALGO_BY_ID
from core.curriculum import get_module
from core.page import footer, page_header

page_header("algorithm_explorer")

q = st.text_input("ابحث بالاسم أو الوصف", key="ae_q", placeholder="مثال: boosting، kernel، density")
c1, c2, c3, c4 = st.columns(4)
sup = c1.segmented_control("الإشراف", ["الكل", "Supervised", "Unsupervised"], default="الكل", key="ae_sup", required=True)
task = c2.pills("المهمة", ["regression", "classification", "clustering", "dimred", "anomaly"], selection_mode="multi", key="ae_task")
lin = c3.segmented_control("الخطية", ["الكل", "Linear", "Nonlinear"], default="الكل", key="ae_lin", required=True)
par = c4.segmented_control("المعلمية", ["الكل", "Parametric", "Nonparametric"], default="الكل", key="ae_par", required=True)
flags = st.pills("خصائص مطلوبة", ["no scaling needed", "categorical native", "missing native", "interpretable", "probabilistic", "online",
                                  "ensemble"], selection_mode="multi", key="ae_flags")

rows = []
for a in ALGORITHMS:
    if q and q.lower() not in f"{a.name} {a.problem} {a.intuition} {a.estimator}".lower():
        continue
    if sup != "الكل" and a.supervised != (sup == "Supervised"):
        continue
    if task and not set(task) & set(a.tasks):
        continue
    if lin != "الكل" and a.linear != (lin == "Linear"):
        continue
    if par != "الكل" and a.parametric != (par == "Parametric"):
        continue
    need = set(flags or [])
    if "no scaling needed" in need and a.scaling != "not needed":
        continue
    if "categorical native" in need and not a.categorical_native:
        continue
    if "missing native" in need and not a.missing_native:
        continue
    if "interpretable" in need and a.interpretability != "high":
        continue
    if "probabilistic" in need and not a.probabilistic:
        continue
    if "online" in need and not a.online:
        continue
    if "ensemble" in need and not a.ensemble:
        continue
    rows.append({"id": a.id, "algorithm": a.name, "tasks": ", ".join(a.tasks), "linear": a.linear, "parametric": a.parametric,
                 "scaling": a.scaling, "categorical native": a.categorical_native, "missing native": a.missing_native,
                 "interpretability": a.interpretability, "probabilistic": a.probabilistic, "online (partial_fit)": a.online,
                 "ensemble": a.ensemble, "complexity": a.complexity, "estimator": a.estimator})
df = pd.DataFrame(rows)
st.metric("خوارزميات مطابقة", len(df))
if df.empty:
    st.info("لا نتائج؛ خفّف المرشحات.")
else:
    st.dataframe(df.drop(columns="id"), hide_index=True, width="stretch", height=440)
    pick = st.selectbox("اعرض بطاقة خوارزمية", df["id"], format_func=lambda i: ALGO_BY_ID[i].name, key="ae_pick")
    a = ALGO_BY_ID[pick]
    with st.container(border=True):
        st.markdown(f"### {a.name}")
        st.markdown(f"**الحدس:** {a.intuition}  \n**الافتراضات:** {a.assumptions}  \n**المعالجة:** {a.preprocessing}")
        c1, c2 = st.columns(2)
        c1.markdown("**مزايا:**\n" + "\n".join(f"- {s}" for s in a.strengths))
        c2.markdown("**عيوب:**\n" + "\n".join(f"- {s}" for s in a.weaknesses))
        st.page_link(get_module(a.module).file, label=f"افتح وحدة {get_module(a.module).title_ar}", icon=":material/arrow_back:")
st.download_button("تنزيل مصفوفة الخوارزميات CSV", df.to_csv(index=False).encode("utf-8-sig") if not df.empty else b"",
                   "algorithm_matrix.csv", "text/csv", icon=":material/download:", disabled=df.empty)
footer()
