import pandas as pd
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.dataset_viewer import dataset_card
from config import DATA_SCIENCE_PLATFORM_URL
from core.page import page_footer, page_header, page_link
from core.state import at_least
from utils.datasets import DATASET_INFO, load_dataset
from utils.preprocessing import readiness_frame, readiness_report

page_header("ml_readiness")

with st.container(key="ml-dsplat-readiness"):
    st.markdown("**:material/link: جسر من منصة علم البيانات**")
    st.markdown("مررت في [DSplat](" + DATA_SCIENCE_PLATFORM_URL + ") بجودة البيانات، القيم المفقودة، التكرارات، القيم "
                "الشاذة، الترميز والقياس. هنا نعيد طرح الأسئلة نفسها **بعين النمذجة**: هل هذه البيانات صالحة لتعلّم نموذج "
                "سيُستخدم على حالات مستقبلية؟")

st.markdown("## قائمة الجاهزية: 13 بندًا")
items = [
    ("Target definition", "هل الهدف معرّف بدقة، ومقاس للجميع، وفي الأفق الزمني المطلوب؟"),
    ("Unit of analysis", "ماذا يمثل الصف؟ هل يتكرر الكيان نفسه؟"),
    ("Availability at prediction time", "هل كل خاصية معروفة لحظة التنبؤ؟"),
    ("Missingness", "كم؟ أين؟ ولماذا؟ (MCAR/MAR/MNAR)"),
    ("Outliers", "أخطاء أم حالات حقيقية نادرة؟"),
    ("Feature types", "عددية، فئوية، نصية، زمنية؛ الكاردينالية."),
    ("Leakage", "هل توجد خاصية «تعرف» الإجابة؟"),
    ("Duplicates", "صفوف مكررة ستقع في التدريب والاختبار معًا؟"),
    ("Imbalance", "نسبة الفئة النادرة؟"),
    ("Temporal order", "هل الترتيب الزمني مهم للتقسيم؟"),
    ("Grouped observations", "مرضى/عملاء/مدارس متكررة؟"),
    ("Sample size", "هل n كافٍ مقارنة بـp وتعقيد النموذج؟"),
    ("Shift risk", "هل ستختلف بيانات النشر عن التدريب؟"),
]
st.dataframe(pd.DataFrame(items, columns=["البند", "السؤال"]), hide_index=True, width="stretch")

st.markdown("## مختبر: فحص آلي للجاهزية")
st.caption("الفحص الآلي يغطي ما يمكن قياسه؛ البنود التي تحتاج معرفة المجال تُعلَّم «يحتاج حكمك».")
options = ["mixed", "imbalanced", "classification", "panel", "timeseries", "regression"]
c1, c2 = st.columns([2, 1])
name = c1.selectbox("مجموعة البيانات", options, format_func=lambda n: DATASET_INFO[n].title, key="rd_ds")
inject = c2.toggle("أضف مشكلات عمدًا (تسرب + تكرار)", value=False, key="rd_inject",
                   help="يضيف عمودًا مشتقًا من الهدف وصفوفًا مكررة لترى كيف يلتقطها الفحص.")
df = load_dataset(name).copy()
target = DATASET_INFO[name].target
if name == "timeseries":
    df = df.assign(target_next=df["sales"].shift(-1)).dropna()
    target = "target_next"
if inject:
    df["days_since_decision"] = df[target].rank(method="first") * 0.01 if df[target].nunique() > 15 else \
        df[target] * 30 + (df.index % 3)
    df = pd.concat([df, df.sample(40, random_state=0)], ignore_index=True)
drop_truth = [c for c in ("tau", "true_ps", "true_att", "is_anomaly", "cluster_true") if c in df]
df = df.drop(columns=drop_truth)
id_col = "id" if "id" in df else None
time_col = "date" if "date" in df else ("t" if "t" in df else None)
group_col = "id" if "id" in df else None
checks = readiness_report(df.drop(columns=[c for c in ("date",) if c in df]), target, id_col=id_col,
                          time_col=time_col, group_col=group_col)
report = readiness_frame(checks)
summary = pd.Series([c.status for c in checks]).value_counts()
with st.container(horizontal=True):
    st.metric("جاهز", int(summary.get("ok", 0)), border=True)
    st.metric("انتبه", int(summary.get("warn", 0)), border=True)
    st.metric("مشكلة", int(summary.get("fail", 0)), border=True)
    st.metric("يحتاج حكمك", int(summary.get("manual", 0)), border=True)
st.dataframe(report, hide_index=True, width="stretch")
if summary.get("fail", 0):
    st.error("يوجد بند واحد على الأقل بحالة «مشكلة»: عالجه قبل النمذجة، وإلا فالأرقام اللاحقة غير موثوقة.",
             icon=":material/block:")
dataset_card(name, show_head=False)

why("لا يُستبدل حكم المجال بالفحص الآلي.",
    "الفحص يرى الأنماط الإحصائية فقط؛ لا يعرف متى تُسجَّل الخاصية أو كيف ستُستخدم التنبؤات. البنود اليدوية غالبًا هي "
    "مصدر أخطر أخطاء المشاريع (التسرب والانجراف).")

st.markdown("## إلى أين بعد الفحص؟")
c1, c2, c3 = st.columns(3)
with c1:
    page_link("leakage", "التسرب: بنود Leakage وAvailability", ":material/water_drop:")
with c2:
    page_link("cross_validation", "المجموعات والزمن: مخطط CV", ":material/view_week:")
with c3:
    page_link("imbalanced", "عدم التوازن", ":material/balance:")

if at_least("advanced"):
    st.markdown("## متقدم: وثيقة جاهزية للنموذج")
    st.code("""readiness = {
  "target": "approved (decision within 30 days of application)",
  "unit": "one loan application; customers may repeat -> GroupKFold by customer_id",
  "prediction_time": "at submission; exclude fields filled by the credit officer",
  "missing": {"income": "8%, MAR by employment", "credit_score": "5%"},
  "split": "time-based holdout (last 6 months) + grouped CV inside training",
  "imbalance": "approved = 41%",
  "shift_risk": "new cities after expansion -> monitor PSI monthly",
}""", language="python")
if at_least("research"):
    researcher_note(["قائمة الجاهزية جزء من «Datasheets for datasets» و«Model cards»: وثّقها كملحق.",
                     "أي قرار تنظيف يعتمد على الهدف (مثل حذف الشواذ حسب y) يجب أن يحدث داخل CV."])
real_world(["من يملك تعريف الهدف في المؤسسة؟", "متى تُحدَّث كل خاصية في قاعدة البيانات؟",
            "هل تغيّرت السياسات أو الأنظمة خلال فترة البيانات؟"])
page_footer("ml_readiness",
            takeaways=["13 بندًا تسبق أي نموذج.", "الآلي يكشف الإحصائي؛ اليدوي يكشف المنطقي والزمني.",
                       "بند واحد «مشكلة» يكفي لإبطال كل النتائج اللاحقة."],
            mistakes=["القفز إلى النمذجة بعد df.info() فقط.", "تعويض الهدف المفقود.", "تجاهل تكرار الكيانات."])
