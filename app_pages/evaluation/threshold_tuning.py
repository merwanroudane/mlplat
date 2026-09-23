import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold, TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import hyperparameter_table
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from components.metric_explorer import threshold_explorer
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import xy

page_header("threshold_tuning")

st.markdown("## 0.5 ليست قاعدة مقدسة")
formula(r"\hat y = \mathbb 1\big[\hat p(x) \ge t\big],\qquad t^* = \frac{C_{FP}}{C_{FP}+C_{FN}}\ \ \text{(calibrated probabilities)}",
        title="Decision rule and the cost-optimal threshold (Elkan, 2001)",
        symbols={"t": "العتبة", "C_FP, C_FN": "تكلفة الإنذار الكاذب والحالة الفائتة"},
        intuition="قرّر «موجب» إذا كانت التكلفة المتوقعة لقول «سالب» (p·C_FN) أكبر من تكلفة قول «موجب» ((1 − p)·C_FP).",
        example="C_FN = 9، C_FP = 1 ⇒ t* = 0.1: يكفي احتمال 10% لإطلاق الإنذار.")

X, y = xy("imbalanced")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, random_state=0, stratify=ytr)


@st.cache_data(show_spinner=False)
def _proba():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, ya)
    return m.predict_proba(Xv)[:, 1]


st.markdown("## Threshold Lab (على مجموعة التحقق)")
threshold_explorer(yv.to_numpy(), _proba(), key="tt_thr", default_cost_fp=1.0, default_cost_fn=10.0)

st.markdown("## TunedThresholdClassifierCV: الضبط داخل CV")
c1, c2 = st.columns(2)
cfp = c1.number_input("C_FP", 0.1, 100.0, 1.0, 0.5, key="tt_cfp")
cfn = c2.number_input("C_FN", 0.1, 100.0, 10.0, 0.5, key="tt_cfn")


def neg_cost(y_true, y_pred, cfp=1.0, cfn=10.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return -(cfp * fp + cfn * fn) / len(y_true)


@st.cache_data(show_spinner="يضبط العتبة بـ5 طيات…", max_entries=16)
def _tune(cfp: float, cfn: float):
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scorer = make_scorer(neg_cost, cfp=cfp, cfn=cfn)
    tuned = TunedThresholdClassifierCV(base, scoring=scorer, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                       store_cv_results=True, random_state=0).fit(Xtr, ytr)
    fixed = FixedThresholdClassifier(base, threshold=0.5).fit(Xtr, ytr)
    rows = []
    for name, m in (("default threshold 0.5", fixed), (f"tuned threshold {tuned.best_threshold_:.3f}", tuned)):
        pred = m.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        rows.append({"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "cost per case (test)": (cfp * fp + cfn * fn) / len(yte)})
    return pd.DataFrame(rows), tuned.best_threshold_, cfp / (cfp + cfn)


res, best_t, theory = _tune(cfp, cfn)
st.dataframe(res.round(4), hide_index=True, width="stretch")
st.markdown(f"العتبة المضبوطة بـCV = **{best_t:.3f}** (النظرية مع احتمالات معايرة: {theory:.3f}). الاختبار استُخدم مرة واحدة للتقييم "
            "النهائي فقط.")
st.code("""from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer
tuned = TunedThresholdClassifierCV(pipeline, scoring=make_scorer(neg_cost, cfp=1, cfn=10), cv=5)
tuned.fit(X_train, y_train)          # threshold chosen by internal CV — test set untouched
tuned.best_threshold_, tuned.predict(X_new)""", language="python")
hyperparameter_table("TunedThresholdClassifierCV")
warning("لا تضبط العتبة على Test. العتبة معامل فائق مثل C؛ اختيارها على Test يجعل تقدير الأداء متفائلًا.")

comparison_table([
    {"": "Threshold tuning", "ماذا يغيّر": "القرار فقط", "الاحتمالات": "كما هي", "متى": "تكاليف غير متماثلة"},
    {"": "Calibration", "ماذا يغيّر": "الاحتمالات", "الاحتمالات": "تصبح موثوقة", "متى": "حين تُستخدم الاحتمالات نفسها"},
    {"": "class_weight", "ماذا يغيّر": "التدريب", "الاحتمالات": "تُشوَّه", "متى": "بديل تقريبي؛ يفقد المعايرة"},
])
why("افصل: درّب نموذجًا جيد الترتيب، عايره إن لزم، ثم اضبط العتبة من التكاليف.",
    "كل خطوة تعالج مشكلة مختلفة؛ خلطها (مثل class_weight وحده) يصعّب التفسير وتغيير السياسة لاحقًا.")
intuition("تغيير التكاليف لا يحتاج إعادة تدريب؛ يكفي تحريك العتبة. لذلك احفظ الاحتمالات لا القرارات.")

if at_least("advanced"):
    st.markdown("## متقدم: عتبات مقيدة")
    st.markdown("- «Recall ≥ 90% بأقل عدد إنذارات» ⇒ اختر أعلى عتبة تحقق القيد على التحقق.\n"
                "- «نراجع 100 حالة يوميًا فقط» ⇒ القرار يصبح Top-k لا عتبة ثابتة؛ قيّم بـPrecision@k.\n"
                "- TunedThresholdClassifierCV يقبل أي scorer بما فيها scorers مخصصة بقيود.")
if at_least("research"):
    researcher_note(["العتبة المثلى نظريًا تفترض معايرة جيدة وتكاليف ثابتة؛ في الواقع التكاليف تعتمد على الحالة (Example-dependent costs).",
                     "في السياسات العامة، العتبة قرار معياري يجب توثيقه وتبريره، لا تفصيل تقني."])
    st.markdown(cite("elkan2001"))
mistakes(["0.5 افتراضيًا دون سؤال عن التكاليف.", "اختيار العتبة على Test.", "تغيير العتبة ثم مقارنة AUC (AUC لا يتغير)."])
page_footer("threshold_tuning",
            takeaways=["العتبة قرار أعمال لا خاصية نموذج.", "t* = C_FP/(C_FP + C_FN) مع احتمالات معايرة.",
                       "TunedThresholdClassifierCV يضبطها داخل CV."])
