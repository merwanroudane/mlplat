import io
import json
import time

import joblib
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import xy
from utils.preprocessing import ordinal_categoricals
from utils.validation import package_versions

page_header("production_ml")

st.markdown("## النموذج جزء صغير من النظام")
mermaid("""
flowchart LR
  subgraph Data
    S[Sources] --> V[Validation / contracts] --> F[Feature pipeline]
  end
  subgraph Training
    F --> T[Train + evaluate] --> R[(Model registry)]
  end
  subgraph Serving
    R --> B[Batch scoring job]
    R --> O[Online API]
  end
  B & O --> L[(Prediction logs)]
  L --> M[Monitoring: data / performance drift]
  M -->|alert| T
  M -->|rollback| R
  style T fill:#F3F0FF,stroke:#7048E8
""")
intuition("في Sculley et al. (2015) يظهر كود النموذج مربعًا صغيرًا وسط بنية ضخمة: جمع البيانات والتحقق والخصائص والخدمة والمراقبة. "
          "معظم الأعطال في الإنتاج تأتي من هذه البنية لا من الخوارزمية.")
comparison_table([
    {"المفهوم": "Model artifact", "المعنى": "الـPipeline المدرّب كاملًا (معالجة + نموذج) + بيانات وصفية"},
    {"المفهوم": "Static vs dynamic training", "المعنى": "تدريب مرة وتجميد مقابل إعادة تدريب دورية/مستمرة"},
    {"المفهوم": "Batch inference", "المعنى": "تنبؤات مجدولة لكل الحالات (ليلًا) — بسيطة ورخيصة"},
    {"المفهوم": "Online inference", "المعنى": "API يتنبأ عند الطلب بزمن استجابة منخفض"},
    {"المفهوم": "Serving skew", "المعنى": "اختلاف المعالجة أو البيانات بين التدريب والخدمة"},
    {"المفهوم": "Rollback", "المعنى": "العودة لإصدار سابق حين يسوء الأداء"},
    {"المفهوم": "Logging", "المعنى": "تسجيل المدخلات والمخرجات والإصدار لكل تنبؤ (للمراقبة والتدقيق)"},
])

st.markdown("## بناء Artifact قابل للنشر")
X, y = xy("mixed")
Xc = ordinal_categoricals(X)


@st.cache_resource(show_spinner="يدرّب الـPipeline…")
def _train():
    pipe = Pipeline([("model", HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0))]).fit(Xc, y)
    return pipe


pipe = _train()
meta = {"model": "HistGradientBoostingClassifier", "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_rows": int(len(Xc)), "features": list(Xc.columns), "target": "approved", "versions": package_versions(
            ("scikit-learn", "numpy", "pandas")), "categories": {c: list(Xc[c].cat.categories) for c in Xc.select_dtypes("category")}}
buf = io.BytesIO()
joblib.dump(pipe, buf)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**model_card.json (بيانات وصفية)**")
    st.json({k: v for k, v in meta.items() if k != "categories"}, expanded=False)
    st.download_button("نزّل model_card.json", json.dumps(meta, ensure_ascii=False, indent=2), "model_card.json", "application/json",
                       icon=":material/download:")
with c2:
    st.metric("حجم الـartifact (joblib)", f"{buf.getbuffer().nbytes / 1e6:.2f} MB")
    st.caption("لا نوفّر تنزيل ملف pickle/joblib من التطبيق العام عمدًا (انظر التحذير أدناه).")
warning("**مخاطر التسلسل (Serialization):** pickle/joblib يمكن أن ينفّذ كودًا عشوائيًا عند التحميل. **لا تحمّل أبدًا ملف pickle من مصدر "
        "غير موثوق** — ولهذا لا تقبل هذه المنصة رفع نماذج. بدائل أكثر أمانًا: تنسيقات مثل skops.io (مع قائمة أنواع موثوقة) أو ONNX "
        "للاستدلال؛ ووقّع الـartifacts وخزّنها في سجل موثوق.", title=":material/gpp_maybe: أمان")
why("احفظ مع النموذج: الإصدارات، الأعمدة وأنواعها، الفئات المعروفة، ومقاييس التحقق.",
    "تحميل نموذج بإصدار مكتبة مختلف قد يغيّر السلوك أو يفشل؛ والفئات الجديدة في الإنتاج تحتاج معالجة معروفة مسبقًا.")

st.markdown("## Batch مقابل Online")
comparison_table([
    {"": "Batch", "الزمن": "دقائق/ساعات", "البنية": "مهمة مجدولة + جدول مخرجات", "الخصائص": "يمكن حسابها بكلفة", "المراقبة": "أسهل"},
    {"": "Online", "الزمن": "ميلي ثوانٍ", "البنية": "API + خصائص فورية (Feature store)", "الخصائص": "يجب أن تتوفر لحظيًا", "المراقبة": "أصعب"},
])
new = Xc.sample(5, random_state=3)
t0 = time.perf_counter()
proba = pipe.predict_proba(new)[:, 1]
st.dataframe(new.assign(**{"P(approved)": proba.round(3)}), hide_index=True, width="stretch")
st.caption(f"زمن تنبؤ 5 صفوف: {(time.perf_counter() - t0) * 1000:.1f} ms — «خدمة» مصغرة داخل الصفحة.")

if at_least("advanced"):
    st.markdown("## متقدم: عقد البيانات (Data contract)")
    st.code("""contract = {
  "age": {"dtype": "int", "min": 18, "max": 100},
  "income": {"dtype": "float", "min": 0, "nullable": True},
  "employment": {"dtype": "category", "allowed": ["salaried", "self_employed", "public", "unemployed"]},
}
# validate every scoring batch against the contract BEFORE predict; log and quarantine violations""", language="python")
if at_least("research"):
    researcher_note(["Sculley et al. (2015): Hidden technical debt — Glue code، Pipeline jungles، Feedback loops.",
                     "Mitchell et al. (2019): Model cards لتوثيق الاستخدام المقصود والقيود والأداء حسب المجموعات."])
    st.markdown(cite("sculley2015", "mitchell2019", "mlcc"))
mistakes(["حفظ النموذج دون المعالجة.", "تحميل pickle غير موثوق.", "خصائص في التدريب غير متوفرة لحظة الخدمة.", "لا تسجيل للتنبؤات."])
page_footer("production_ml",
            takeaways=["احفظ الـPipeline كاملًا مع بيانات وصفية وإصدارات.", "Batch أبسط؛ Online يتطلب خصائص فورية.",
                       "pickle غير آمن مع مصادر غير موثوقة."])
