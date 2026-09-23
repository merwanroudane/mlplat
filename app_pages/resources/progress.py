import pandas as pd
import streamlit as st

from core.curriculum import GROUPS, get_module, modules_in_group
from core.page import footer, page_header
from core.state import export_progress, import_progress, is_complete, progress_pct

page_header("progress")
with st.container(horizontal=True):
    st.metric("التقدّم الكلي", f"{progress_pct():.0f}%", border=True)
    st.metric("وحدات مكتملة", len(st.session_state.get("completed", set())), border=True)
    st.metric("اختبارات أُجريت", len(st.session_state.get("quiz_scores", {})), border=True)
    st.metric("مختبرات زُرت", len(st.session_state.get("labs_visited", set())), border=True)
    st.metric("سجلات تجارب", len(st.session_state.get("experiment_log", [])), border=True)

rows = []
for g, label in GROUPS.items():
    mods = [m for m in modules_in_group(g) if m.tracked]
    if mods:
        rows.append({"المجموعة": label, "مكتملة": sum(is_complete(m.id) for m in mods), "الكل": len(mods)})
df = pd.DataFrame(rows)
df["النسبة"] = (df["مكتملة"] / df["الكل"]).round(2)
st.dataframe(df, hide_index=True, width="stretch", column_config={"النسبة": st.column_config.ProgressColumn("النسبة", min_value=0, max_value=1)})

scores = st.session_state.get("quiz_scores", {})
if scores:
    st.markdown("### نتائج الاختبارات")
    st.dataframe(pd.DataFrame([{"الوحدة": get_module(k).title_ar, "النتيجة": f"{v['score']}/{v['total']}", "الوقت": v["at"]}
                               for k, v in scores.items()]), hide_index=True, width="stretch")
marks = st.session_state.get("bookmarks", set())
if marks:
    st.markdown("### المفضلة")
    for mid in sorted(marks):
        m = get_module(mid)
        st.page_link(m.file, label=m.title_ar, icon=":material/bookmark:")

st.markdown("### حفظ التقدم واستعادته")
st.caption("لا حسابات ولا قاعدة بيانات: التقدم في جلستك فقط. صدّره JSON واستورده لاحقًا (JSON فقط — لا يُقبل أي تنسيق قابل للتنفيذ).")
st.download_button("تصدير التقدم (JSON)", export_progress(), "ml_academy_progress.json", "application/json", icon=":material/download:")
up = st.file_uploader("استيراد ملف تقدم", type=["json"], key="prog_up")
if up is not None:
    ok, msg = import_progress(up.getvalue().decode("utf-8", errors="replace"))
    (st.success if ok else st.error)(msg)
footer()
