import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from components.callouts import causal_caution, definition, real_world, researcher_note
from components.cards import comparison_table
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import bars, plot

page_header("prediction_vs_causality")

st.markdown("## ثلاثة أسئلة مختلفة")
comparison_table([
    {"السؤال": "Prediction · التنبؤ", "الصيغة": "ما قيمة Y المتوقعة لهذه الحالة؟", "الهدف الإحصائي": "E[Y | X = x]",
     "التقييم": "خطأ خارج العينة", "مثال": "من سيتعثر في سداد القرض؟"},
    {"السؤال": "Explanation · التفسير", "الصيغة": "ما البنية التي تولّد البيانات؟", "الهدف الإحصائي": "معاملات نموذج نظري",
     "التقييم": "صحة الافتراضات والاستدلال", "مثال": "هل يرتبط الدخل بالتعثر بعد ضبط العمر؟"},
    {"السؤال": "Causality · السببية", "الصيغة": "ماذا يحدث لو تدخّلنا؟", "الهدف الإحصائي": "E[Y(1) − Y(0)]",
     "التقييم": "افتراضات التعريف + تحليل الحساسية", "مثال": "هل يقلل خفض الفائدة التعثر؟"},
])
causal_caution("نموذج تنبؤي ممتاز قد يعتمد على متغيرات **نتيجة** للهدف لا **سبب** له. «عدد زيارات الطبيب» يتنبأ "
               "بالمرض جيدًا، لكن منع الزيارات لن يمنع المرض.")

st.markdown("## مختبر: أهم متغير تنبؤيًا ليس بالضرورة سببًا")
st.caption("DGP معروف: `ice_cream` لا يؤثر في `drownings` إطلاقًا؛ كلاهما يتأثر بـ`temperature`. و`rescue_calls` نتيجة للغرق.")
n = st.slider("حجم العينة", 200, 3000, 1000, 100, key="pvc_n")
include_temp = st.toggle("أدخل temperature (المُربك) في النموذج", value=False, key="pvc_temp")


@st.cache_data(show_spinner=False)
def _sim(n: int, include_temp: bool):
    rng = np.random.default_rng(1)
    temp = rng.normal(25, 6, n)
    ice = 3 * temp + rng.normal(0, 8, n)
    drown = 0.4 * temp + rng.normal(0, 2, n)         # true causal parent: temperature only
    rescue = 1.5 * drown + rng.normal(0, 1, n)        # a consequence of the outcome
    X = pd.DataFrame({"ice_cream": ice, "rescue_calls": rescue, "noise": rng.normal(size=n)})
    if include_temp:
        X["temperature"] = temp
    X_tr, X_te, y_tr, y_te = train_test_split(X, drown, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X_tr, y_tr)
    pi = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0)
    # naive regression of drownings on ice cream alone vs. controlling for temperature
    b_naive = np.polyfit(ice, drown, 1)[0]
    Z = np.c_[np.ones(n), ice, temp]
    b_adj = np.linalg.lstsq(Z, drown, rcond=None)[0][1]
    return X.columns.tolist(), pi.importances_mean, rf.score(X_te, y_te), b_naive, b_adj


cols, imp, r2, b_naive, b_adj = _sim(n, include_temp)
c1, c2 = st.columns([1.3, 1])
with c1:
    order = np.argsort(-imp)
    plot(bars([cols[i] for i in order], [imp[i] for i in order], title=f"Permutation importance (test R² = {r2:.2f})",
              horizontal=True, color=PALETTE["purple"]), height=320)
with c2:
    st.metric("ميل drownings على ice_cream (ساذج)", f"{b_naive:.3f}")
    st.metric("الميل بعد ضبط temperature", f"{b_adj:.3f}", f"{b_adj - b_naive:+.3f}", delta_color="off")
    st.markdown("**الحقيقة:** الأثر السببي لـ`ice_cream` = **0**، و`rescue_calls` نتيجة لا سبب.")
st.markdown("لاحظ: `rescue_calls` هو الأهم تنبؤيًا (لأنه **نتيجة** للغرق)، و`ice_cream` مهم تنبؤيًا دون أي أثر سببي. "
            "ضبط المُربك يقرّب ميل `ice_cream` من الصفر. التنبؤ يستغل كل الارتباطات؛ السببية تحتاج بنية.")

mermaid("""
flowchart LR
  T((temperature)) --> I[ice_cream]
  T --> D[drownings]
  D --> R[rescue_calls]
  style T fill:#FFF3BF,stroke:#F59F00
  style D fill:#E7F5FF,stroke:#1C7ED6
""")

st.markdown("## متى تحتاج Causal ML؟")
comparison_table([
    {"الموقف": "ترتيب العملاء حسب احتمال المغادرة", "النوع": "تنبؤ", "الأداة": "مصنف + معايرة"},
    {"الموقف": "هل يقلل الخصم المغادرة؟ وبكم؟", "النوع": "سببي (ATE)", "الأداة": "تجربة عشوائية أو DML"},
    {"الموقف": "لمن نعطي الخصم؟", "النوع": "سببي غير متجانس (CATE)", "الأداة": "Meta-learners / Causal forest"},
    {"الموقف": "ما المتغيرات المرتبطة بالمغادرة؟", "النوع": "وصف/تفسير", "الأداة": "نموذج قابل للتفسير + حذر"},
])
definition("Double Machine Learning باختصار",
           "يستخدم ML للتنبؤ بما هو **ليس** هدفنا (دوال الإزعاج مثل E[Y|X] وE[D|X])، ثم يبني درجة متعامدة وCross-fitting "
           "لتقدير الأثر السببي θ بفترات ثقة صالحة — تحت افتراضات تعريف سببية لا يختبرها ML.")
page_link("causal_foundations", "ابدأ مسار Causal ML وDML", ":material/device_hub:")

if at_least("advanced"):
    st.markdown("## متقدم: لماذا يفشل التفسير السببي لمعاملات نموذج تنبؤي؟")
    st.markdown("- **الإرباك:** المعامل يلتقط أثر المتغيرات المحذوفة المرتبطة.\n"
                "- **Bad controls:** إدخال نواتج المعالجة (Mediators/Colliders) يحيّز الأثر.\n"
                "- **التنظيم:** Lasso/Ridge يحيّزان المعاملات نحو الصفر عمدًا لتحسين التنبؤ.\n"
                "- **عدم التعريف:** نماذج مختلفة جدًا تعطي تنبؤات متقاربة (Rashomon effect).")
if at_least("research"):
    researcher_note(["Shmueli (2010): التنبؤ والتفسير يتطلبان خيارات مختلفة في كل مرحلة، من اختيار المتغيرات إلى التقييم.",
                     "Mullainathan & Spiess (2017): مسائل ŷ مقابل مسائل β̂؛ ML ممتاز في الأولى ويتطلب حذرًا في الثانية.",
                     "Chernozhukov et al. (2018): يوفّق بين الاثنين باستخدام ML للإزعاج فقط."])
    st.markdown(cite("shmueli2010", "mullainathan2017", "varian2014", "chernozhukov2018"))

real_world(["اكتب سؤالك بصيغة: «ماذا يحدث لو...» (سببي) أو «ما قيمة...» (تنبؤي).",
            "هل ستتخذ قرار تدخّل بناءً على النموذج؟ إذن السؤال سببي.",
            "هل بعض المتغيرات تُقاس بعد المعالجة؟ لا تستخدمها كضوابط في تحليل سببي."])
page_footer("prediction_vs_causality",
            takeaways=["التنبؤ يستغل كل الارتباطات؛ السببية تحتاج افتراضات تعريف.",
                       "أهمية الخاصية ≠ أثرها السببي.", "DML يستخدم ML للإزعاج ويبقي الاستدلال على θ صالحًا."],
            mistakes=["تفسير Feature importance كأثر سببي.", "إدخال متغيرات بعد المعالجة كضوابط.",
                      "استنتاج سياسة من نموذج تنبؤي دون تجربة أو تصميم سببي."])
