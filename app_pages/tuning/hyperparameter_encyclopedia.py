import pandas as pd
import streamlit as st

from content.hyperparameters import CHECKED, HYPERPARAMETERS, VERIFIED_VERSIONS
from core.page import footer, page_header
from core.registry import installed_version

page_header("hyperparameter_encyclopedia")

st.caption(f"كل القيم الافتراضية قُرئت من المكتبات المثبتة بتاريخ {CHECKED} وتُعاد مقارنتها آليًا مع كل تشغيل للاختبارات "
           "(tests/test_hyperparameters.py). إذا ثبّتت إصدارًا مختلفًا، يظهر التحذير في العمود الأخير.")
libs = sorted({h.library for h in HYPERPARAMETERS})
c1, c2 = st.columns([2, 1])
q = c1.text_input("ابحث (اسم المعامل، الخوارزمية، أو كلمة عربية)", key="hpe_q", placeholder="مثال: learning_rate، تنظيم، gamma")
lib = c2.multiselect("المكتبة", libs, default=libs, key="hpe_lib")
algos = sorted({h.algo for h in HYPERPARAMETERS if h.library in lib})
algo = st.pills("الخوارزمية", algos, selection_mode="multi", key="hpe_algo")

rows = []
for h in HYPERPARAMETERS:
    if h.library not in lib or (algo and h.algo not in algo):
        continue
    text = " ".join([h.parameter, h.algo, h.meaning, h.increase, h.decrease, h.interactions, h.note]).lower()
    if q and q.lower() not in text:
        continue
    import_name = {"scikit-learn": "sklearn", "DoubleML": "doubleml"}.get(h.library, h.library)
    inst = installed_version(import_name) or "not installed"
    rows.append({"library": h.library, "estimator": h.algo, "parameter": h.parameter,
                 "default": repr(h.default) + (f" → {h.resolved}" if h.resolved else ""), "allowed": h.allowed, "meaning": h.meaning,
                 "↑ increase": h.increase, "↓ decrease": h.decrease, "interactions": h.interactions, "runtime": h.runtime,
                 "over/underfitting": h.risk, "verified version": h.verified_version,
                 "installed": inst + ("" if inst == h.verified_version or inst == "not installed" else " ⚠ re-verify"),
                 "source": h.url})
df = pd.DataFrame(rows)
st.metric("معاملات مطابقة", len(df))
if df.empty:
    st.info("لا نتائج. جرّب كلمة أخرى.")
else:
    st.dataframe(df, hide_index=True, width="stretch", height=520,
                 column_config={"source": st.column_config.LinkColumn("source", display_text="docs")})
    st.download_button("تنزيل كـCSV", df.to_csv(index=False).encode("utf-8-sig"), "hyperparameters.csv", "text/csv",
                       icon=":material/download:")
with st.expander("الإصدارات المتحقَّق منها"):
    st.dataframe(pd.DataFrame([{"library": k, "verified": v,
                                "installed here": installed_version({"scikit-learn": "sklearn", "DoubleML": "doubleml"}.get(k, k)) or "—"}
                               for k, v in VERIFIED_VERSIONS.items()]), hide_index=True)
st.markdown("**كيف تُقرأ «→»؟** حين يمرر الـwrapper القيمة None (XGBoost/CatBoost)، يُعرض بعدها القيمة الفعلية التي استخدمها "
            "الـbooster، مقروءة من إعداداته المحلولة.")
footer()
