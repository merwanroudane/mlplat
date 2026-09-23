import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from components.sampler_viz import sampler_visualizer
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least
from utils.datasets import xy
from utils.imbalance import cv_evaluate, make_sampler
from utils.plotting import bars, plot

page_header("imb_oversampling")
imb = optional("imblearn")

st.markdown("Oversampling يزيد أمثلة الأقلية في **بيانات التدريب فقط**. الفرق بين الطرق: **أين** تُولَّد الأمثلة الجديدة وكيف.")
comparison_table([
    {"الطريقة": "RandomOverSampler", "كيف": "تكرار أمثلة الأقلية عشوائيًا مع الإرجاع", "أين": "على الأمثلة نفسها",
     "متى": "خط أساس؛ نماذج لا تتأثر بالتكرار كثيرًا"},
    {"الطريقة": "ROS + shrinkage", "كيف": "Smoothed bootstrap: تكرار + ضجيج Gaussian", "أين": "حول الأمثلة", "متى": "بديل ناعم للتكرار"},
    {"الطريقة": "SMOTE", "كيف": "استيفاء خطي بين مثال وأحد أقرب k جيران من الأقلية", "أين": "على القطع بين الجيران",
     "متى": "خصائص عددية متصلة"},
    {"الطريقة": "Borderline-SMOTE 1/2", "كيف": "SMOTE من أمثلة «الخطر» فقط", "أين": "قرب الحد", "متى": "حين يهم الحد الفاصل"},
    {"الطريقة": "SVMSMOTE", "كيف": "SMOTE حول متجهات الدعم من SVM", "أين": "منطقة الهامش", "متى": "حدود واضحة نسبيًا"},
    {"الطريقة": "ADASYN", "كيف": "عدد نقاط لكل مثال ∝ صعوبته (نسبة الأغلبية بين جيرانه)", "أين": "المناطق الصعبة",
     "متى": "حدود معقدة؛ حذر مع الضجيج"},
    {"الطريقة": "KMeansSMOTE", "كيف": "K-Means ثم SMOTE داخل العناقيد ذات الأقلية الكافية", "أين": "داخل عناقيد",
     "متى": "أقلية على شكل جزر صغيرة"},
    {"الطريقة": "SMOTENC / SMOTEN", "كيف": "SMOTE لبيانات مختلطة / فئوية: الفئة الجديدة = الأكثر تكرارًا بين الجيران",
     "أين": "مثل SMOTE", "متى": "خصائص فئوية"},
])

st.markdown("## الصيغ")
formula(r"x_{\text{new}} = x_i + \lambda\,(x_{zi} - x_i),\qquad \lambda \sim U(0,1)", title="SMOTE (Chawla et al., 2002)",
        symbols={"x_i": "مثال من الأقلية", "x_{zi}": "أحد أقرب k جيرانه من الأقلية (k_neighbors = 5 افتراضيًا)"})
formula(r"\text{DANGER} = \{x_i : \tfrac{m}{2} \le m'_i < m\}", title="Borderline-SMOTE (Han et al., 2005)",
        symbols={"m": "عدد الجيران من كل الفئات (m_neighbors = 10)", "m'_i": "عدد جيران x_i من الأغلبية"},
        intuition="m'_i = m ⇒ ضجيج (يُتجاهل)؛ m'_i < m/2 ⇒ آمن؛ بينهما ⇒ على الحد: هنا فقط نولّد.")
formula(r"r_i = \frac{\Delta_i}{k},\quad \hat r_i = \frac{r_i}{\sum_j r_j},\quad g_i = \hat r_i\, G",
        title="ADASYN (He et al., 2008)",
        symbols={r"\Delta_i": "عدد جيران x_i من الأغلبية بين أقرب k", "G": "إجمالي النقاط المطلوبة", "g_i": "نقاط تُولَّد من x_i"})

st.markdown("## مُصوِّر الـSamplers · Sampler Visualizer")
st.caption("النجوم = نقاط اصطناعية، الحلقات = تكرارات. غيّر التداخل وعدد العناقيد وقارن SMOTE بـBorderline وADASYN.")
sampler_visualizer("over", families=("over",), default="SMOTE")
intuition("مع تداخل كبير يضع SMOTE نقاطًا اصطناعية داخل منطقة الأغلبية (استيفاء بين مثالين نادرين يمر عبر الأغلبية). "
          "Borderline وADASYN يركّزان على الحد — مفيد حين يكون الحد حقيقيًا، وضار حين تكون أمثلة الحد ضجيجًا.")

st.markdown("## الترتيب الصحيح: داخل الـPipeline")
mermaid("flowchart LR\n  S[StratifiedKFold] --> T[fold train] --> O[Oversampler.fit_resample] --> M[model.fit]\n"
        "  S --> V[fold validation: untouched] --> E[evaluate]\n  M --> E")
st.code("""from imblearn.pipeline import make_pipeline        # NOT sklearn.pipeline: it must call fit_resample
from imblearn.over_sampling import SMOTE

pipe = make_pipeline(StandardScaler(), SMOTE(k_neighbors=5, random_state=0), LogisticRegression(max_iter=2000))
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="average_precision")
# during predict / score the sampler is skipped: validation data keep their real prevalence""", language="python")
warning("SMOTE قبل التقسيم يضع نقاطًا مستوفاة من أمثلة التحقق داخل التدريب ⇒ درجات مضخّمة. imblearn.pipeline يطبق "
        "fit_resample على التدريب فقط ويتخطاه عند التنبؤ.")

st.markdown("## مقارنة داخل CV على بيانات الاحتيال")


@st.cache_data(show_spinner="5-fold CV لعدة Oversamplers…")
def _compare() -> pd.DataFrame:
    from imblearn.pipeline import make_pipeline as imb_pipe
    X, y = xy("imbalanced")
    lr = lambda: LogisticRegression(max_iter=2000)  # noqa: E731
    models = {"no resampling": make_pipeline(StandardScaler(), lr())}
    for name in ("RandomOverSampler", "ROS + shrinkage (smoothed bootstrap)", "SMOTE", "BorderlineSMOTE-1",
                 "SVMSMOTE", "ADASYN"):
        models[name] = imb_pipe(StandardScaler(), make_sampler(name), lr())
    rows = []
    for name, m in models.items():
        r = cv_evaluate(m, X, y, n_splits=5).mean(numeric_only=True)
        rows.append({"strategy": name, **{k: r[k] for k in ("PR-AUC (AP)", "ROC-AUC", "recall", "precision", "F1", "MCC",
                                                             "Brier", "mean p")}})
    return pd.DataFrame(rows)


if imb is None:
    missing_notice("imblearn")
else:
    if st.button("شغّل المقارنة", key="over_run", type="primary", icon=":material/play_arrow:"):
        st.session_state["over_done"] = True
    if st.session_state.get("over_done"):
        res = _compare()
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        plot(bars(res["strategy"], res["PR-AUC (AP)"], title="PR-AUC by oversampler (threshold-free)", horizontal=True),
             height=340)
        base = res.iloc[0]
        st.markdown(f"القراءة: بلا إعادة عيّنة PR-AUC = {base['PR-AUC (AP)']:.3f} ومتوسط الاحتمال {base['mean p']:.3f} "
                    f"(≈ الانتشار). الطرق الأخرى ترفع Recall عند 0.5 وترفع متوسط الاحتمال بعيدًا عن الانتشار، بينما جودة "
                    "الترتيب (PR-AUC/ROC-AUC) تتغير قليلًا أو تنخفض. أي أن معظم الأثر = تحريك نقطة التشغيل.")

st.markdown("## البيانات الفئوية: SMOTENC وSMOTEN")
st.markdown("SMOTE على أعمدة One-hot يولّد قيمًا كسرية مثل city_B = 0.37 — فئة غير موجودة. SMOTENC يستوفي الأعمدة العددية "
            "فقط، ويعطي كل عمود فئوي **القيمة الأكثر تكرارًا بين الجيران**.")


@st.cache_data(show_spinner=False)
def _nc_demo():
    from imblearn.over_sampling import SMOTE, SMOTENC
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"amount": rng.gamma(2, 50, n).round(1), "age": rng.integers(18, 80, n),
                       "city": rng.choice(["A", "B", "C"], n)})
    y = np.r_[np.zeros(180, int), np.ones(20, int)]
    onehot = pd.get_dummies(df, columns=["city"], dtype=float)
    Xs, _ = SMOTE(random_state=0).fit_resample(onehot, y)
    Xn, _ = SMOTENC(categorical_features=[2], random_state=0).fit_resample(df, y)
    return Xs.iloc[n:n + 5].round(2), Xn.iloc[n:n + 5]


if imb is not None:
    a, b = _nc_demo()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SMOTE على One-hot (خطأ)**")
        st.dataframe(a, hide_index=True, width="stretch")
    with c2:
        st.markdown("**SMOTENC (صحيح)**")
        st.dataframe(b, hide_index=True, width="stretch")
definition("SMOTEN", "لبيانات **كلها فئوية**: يستخدم مسافة Value Difference Metric بين الفئات، والقيمة الجديدة لكل عمود هي الأكثر "
                     "تكرارًا بين الجيران.")

if at_least("advanced"):
    st.markdown("## متقدم: متى يضر Oversampling؟")
    st.markdown("- **ضجيج التسميات:** SMOTE وADASYN يضاعفان أمثلة الأقلية الخاطئة ويبنيان حولها مناطق كاملة.\n"
                "- **أبعاد عالية:** الاستيفاء بين نقاط متباعدة يولّد نقاطًا بلا معنى؛ مسافات أقرب الجيران تفقد التمييز.\n"
                "- **نماذج الأشجار المعززة القوية:** غالبًا لا تحتاج Oversampling؛ class_weight أو عتبة تكفي بتكلفة أقل.\n"
                "- **الاحتمالات:** كل Oversampling يرفع الانتشار الفعلي في التدريب ⇒ احتمالات أعلى من الحقيقة.")
if at_least("research"):
    researcher_note([
        "sampling_strategy كـfloat (مثل 0.3) يعطي موازنة جزئية؛ اضبطها كمعامل فائق داخل CV بدل 1:1 التلقائية.",
        "بلّغ عن نتائج «بلا إعادة عيّنة + عتبة مضبوطة» كخط أساس لأي ادعاء بتفوق طريقة Oversampling.",
        "استخدم random_state ثابتًا؛ تقلب SMOTE بين البذور قد يعادل الفروق بين الطرق.",
    ])
why("جرّب Oversampling كمعامل فائق داخل Pipeline، لا كخطوة معالجة مسبقة ثابتة.",
    "بهذا يحكم التحقق المتقاطع إن كان مفيدًا لبياناتك، ويبقى التقييم على الانتشار الحقيقي.")
st.markdown("### المراجع")
st.markdown(cite("chawla2002", "han2005", "he2008adasyn", "batista2004", "lemaitre2017"))
mistakes(["SMOTE قبل التقسيم أو قبل CV.", "sklearn.pipeline بدل imblearn.pipeline.", "SMOTE على أعمدة One-hot.",
          "k_neighbors أكبر من عدد أمثلة الأقلية في الطية.", "تفسير الاحتمالات بعد SMOTE كمخاطر حقيقية."])
page_footer("imb_oversampling",
            takeaways=["ROS يكرر، SMOTE يستوفي، Borderline/SVMSMOTE يركزان على الحد، ADASYN على الأمثلة الصعبة.",
                       "SMOTENC/SMOTEN للبيانات الفئوية.", "دائمًا داخل imblearn.pipeline وعلى تدريب كل طية.",
                       "الأثر الرئيسي غالبًا تحريك نقطة التشغيل وتشويه الاحتمالات."])
