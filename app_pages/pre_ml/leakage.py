import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

from components.callouts import definition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import bars, plot

page_header("leakage")

definition("Data leakage", "أي معلومات تدخل التدريب أو التقييم **لن تكون متاحة لحظة التنبؤ الحقيقي**، فتجعل الأداء المقدَّر "
           "أفضل من الأداء الفعلي (Kaufman et al., 2012).")

st.markdown("## تسعة أنواع")
comparison_table([
    {"النوع": "Target leakage", "مثال": "«تاريخ إغلاق الحساب» للتنبؤ بالمغادرة", "العلاج": "احذف ما يُسجَّل بعد الحدث"},
    {"النوع": "Preprocessing leakage", "مثال": "fit للمعالجة على كل البيانات", "العلاج": "Pipeline داخل CV"},
    {"النوع": "Scaling leakage", "مثال": "StandardScaler().fit(X) قبل التقسيم", "العلاج": "Scaler داخل Pipeline"},
    {"النوع": "Imputation leakage", "مثال": "المتوسط محسوب مع الاختبار", "العلاج": "SimpleImputer داخل Pipeline"},
    {"النوع": "Feature-selection leakage", "مثال": "اختيار أفضل 20 خاصية على كل البيانات", "العلاج": "الاختيار داخل كل طية"},
    {"النوع": "Target-encoding leakage", "مثال": "متوسط الهدف لكل فئة يتضمن الصف نفسه", "العلاج": "TargetEncoder (cross-fitting داخلي)"},
    {"النوع": "Group leakage", "مثال": "زيارات المريض نفسه في التدريب والاختبار", "العلاج": "GroupKFold"},
    {"النوع": "Temporal leakage", "مثال": "بيانات المستقبل تتنبأ بالماضي", "العلاج": "تقسيم زمني + خصائص Lag فقط"},
    {"النوع": "HPO leakage", "مثال": "ضبط المعاملات على نفس الطيات التي تبلغ عنها", "العلاج": "Nested CV أو Test معزول"},
])

st.markdown("## مختبر: نتيجة ممتازة من ضجيج خالص")
st.caption("n صغير وp ضخم وتسميات **عشوائية تمامًا**: لا يوجد أي نمط حقيقي؛ الدقة الصحيحة ≈ 50%. قارن الإجراء الخاطئ والصحيح.")
c1, c2, c3 = st.columns(3)
n = c1.slider("n (ملاحظات)", 40, 300, 80, 20, key="lk_n")
p = c2.slider("p (خصائص عشوائية)", 100, 5000, 2000, 100, key="lk_p")
k = c3.slider("عدد الخصائص المختارة k", 5, 100, 20, 5, key="lk_k")


@st.cache_data(show_spinner="يحسب…", max_entries=16)
def _selection_leak(n: int, p: int, k: int):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    # WRONG: select on all data, then cross-validate
    X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
    wrong = cross_val_score(LogisticRegression(max_iter=2000), X_sel, y, cv=cv)
    # RIGHT: selection inside the pipeline, refit in every fold
    right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000)), X, y, cv=cv)
    return wrong.mean(), wrong.std(), right.mean(), right.std()


w, ws, r, rs = _selection_leak(n, p, k)
c1, c2 = st.columns([1.2, 1])
with c1:
    plot(bars(["WRONG: select, then CV", "RIGHT: select inside CV"], [w, r], errors=[ws, rs],
              title="Cross-validated accuracy on pure noise", color=[PALETTE["coral"], PALETTE["teal"]]), height=340)
with c2:
    st.metric("الإجراء الخاطئ", f"{w:.1%}", "يبدو «نموذجًا ممتازًا»", delta_color="off")
    st.metric("الإجراء الصحيح", f"{r:.1%}", "≈ الصدفة — وهي الحقيقة", delta_color="off")
warning("الاختيار على كل البيانات «رأى» تسميات الطيات التي ستُستخدم للتقييم؛ فاختار خصائص ترتبط صدفةً بها. مع p كبير توجد "
        "دائمًا خصائص كهذه. هذا بالضبط ما حدث في دراسات جينومية منشورة (Ambroise & McLachlan, 2002).")
st.code("""# WRONG — leakage
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)      # saw every label
cross_val_score(LogisticRegression(), X_sel, y, cv=5)

# RIGHT — selection is part of the model
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
cross_val_score(pipe, X, y, cv=5)                              # re-selected inside every fold""", language="python")

st.markdown("## مختبر ثانٍ: Group leakage وTemporal leakage")
kind = st.segmented_control("النوع", ["Group leakage", "Temporal leakage", "Target encoding"], default="Group leakage",
                            key="lk_kind", required=True)


@st.cache_data(show_spinner=False)
def _group_leak():
    rng = np.random.default_rng(1)
    n_groups, per = 60, 8
    g = np.repeat(np.arange(n_groups), per)
    signature = rng.normal(size=(n_groups, 20))[g]          # each patient has a unique "fingerprint"
    y = rng.integers(0, 2, n_groups)[g]                      # label is constant within patient, unrelated to X
    X = signature + rng.normal(scale=0.3, size=signature.shape)
    from sklearn.neighbors import KNeighborsClassifier
    m = KNeighborsClassifier(1)
    rnd = cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    grp = cross_val_score(m, X, y, cv=GroupKFold(5), groups=g).mean()
    return rnd, grp


@st.cache_data(show_spinner=False)
def _temporal_leak():
    rng = np.random.default_rng(2)
    n = 600
    walk = np.cumsum(rng.normal(size=n))
    y = (np.diff(np.r_[0, walk]) > 0).astype(int)            # next-step direction: unpredictable
    X = np.c_[walk, np.r_[walk[1:], walk[-1]]]              # includes tomorrow's level (future information!)
    m = LogisticRegression(max_iter=2000)
    leak = cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    honest = cross_val_score(m, X[:, :1], y, cv=TimeSeriesSplit(5)).mean()
    return leak, honest


@st.cache_data(show_spinner=False)
def _te_leak():
    rng = np.random.default_rng(3)
    n = 1000
    cat = rng.integers(0, 400, n).astype(str)               # 400 levels, ~2.5 rows each
    y = rng.integers(0, 2, n)                               # unrelated to the category
    df = pd.DataFrame({"cat": cat, "y": y})
    naive = df.groupby("cat")["y"].transform("mean").to_numpy()[:, None]   # includes the row's own label
    m = LogisticRegression()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    wrong = cross_val_score(m, naive, y, cv=cv).mean()
    right = cross_val_score(make_pipeline(TargetEncoder(random_state=0), LogisticRegression()), df[["cat"]], y, cv=cv).mean()
    return wrong, right


if kind == "Group leakage":
    a, b = _group_leak()
    st.markdown("كل «مريض» له 8 زيارات وبصمة فريدة، والتسمية ثابتة للمريض ولا علاقة لها بالبصمة. نموذج 1-NN:")
    c1, c2 = st.columns(2)
    c1.metric("Random KFold", f"{a:.1%}", "يحفظ المريض", delta_color="off")
    c2.metric("GroupKFold (by patient)", f"{b:.1%}", "مرضى جدد", delta_color="off")
elif kind == "Temporal leakage":
    a, b = _temporal_leak()
    st.markdown("اتجاه الخطوة التالية في Random walk **غير قابل للتنبؤ**، لكن خاصية تتضمن «قيمة الغد» تكشفه:")
    c1, c2 = st.columns(2)
    c1.metric("خاصية مستقبلية + KFold عشوائي", f"{a:.1%}", delta_color="off")
    c2.metric("ماضٍ فقط + TimeSeriesSplit", f"{b:.1%}", delta_color="off")
else:
    a, b = _te_leak()
    st.markdown("400 فئة بمعدل صفين ونصف لكل فئة، والهدف عشوائي. ترميز الهدف الساذج يتضمن تسمية الصف نفسه:")
    c1, c2 = st.columns(2)
    c1.metric("Target mean encoding ساذج", f"{a:.1%}", delta_color="off")
    c2.metric("TargetEncoder داخل Pipeline", f"{b:.1%}", delta_color="off")
    st.caption("`TargetEncoder` في scikit-learn يستخدم Cross-fitting داخليًا في fit_transform (منذ 1.3). ملاحظة: "
               "المعاملان shuffle وrandom_state فيه مهجوران في 1.9 لصالح تمرير cv.")

st.markdown("## كيف تمنع التسرب هيكليًا")
mermaid("""
flowchart TD
  A[Raw data] --> B{Hold out test first}
  B --> T[(Test — locked)]
  B --> C[Development data]
  C --> D[Pipeline = all learned preprocessing + model]
  D --> E[CV splitter matching reality<br/>Group / Time / Stratified]
  E --> F[Tuning inside CV]
  F --> G[Refit on development data]
  G --> H[Evaluate ONCE on test]
  T --> H
""")
why("اجعل كل ما «يتعلم» من البيانات جزءًا من الـPipeline.",
    "عندها يعيد CV تدريبه داخل كل طية آليًا، فلا يمكن هيكليًا أن يرى بيانات التقييم.")
page_link("pipelines", "التالي: Pipelines", ":material/account_tree:")

if at_least("advanced"):
    st.markdown("## متقدم: كشف التسرب")
    st.markdown("- أداء «أفضل من اللازم» مقارنة بالأدبيات أو الخبراء.\n"
                "- خاصية واحدة تسيطر على الأهمية.\n- انهيار الأداء عند تقسيم زمني/جماعي.\n"
                "- Adversarial validation: إن استطاع نموذج تمييز التدريب عن الاختبار، فالتوزيعان مختلفان.")
if at_least("research"):
    researcher_note(["أبلغ عن كل خطوة معالجة وموضعها بالنسبة للتقسيم؛ هذا مطلب شائع في قوائم المراجعة (مثل TRIPOD).",
                     "في DML، Cross-fitting يمنع شكلًا آخر من «التسرب»: استخدام نفس الملاحظة في تقدير الإزعاج والدرجة."])
    st.markdown(cite("kaufman2012", "varma2006"))
mistakes(["fit للـScaler/Imputer على كل البيانات.", "اختيار الخصائص قبل CV.", "KFold عشوائي لكيانات متكررة أو بيانات زمنية.",
          "ضبط المعاملات والإبلاغ على نفس الطيات."])
page_footer("leakage",
            takeaways=["التسرب يجعل الأداء المقدَّر أفضل من الحقيقي.", "كل خطوة تتعلم من البيانات تنتمي إلى الـPipeline.",
                       "مخطط التقسيم يجب أن يطابق البنية (مجموعات/زمن)."])
