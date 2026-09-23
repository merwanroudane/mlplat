import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.registry import missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import plot, scatter_classes

page_header("imbalanced")

st.markdown("هذه الصفحة نظرة عامة ومختبر سريع؛ المسار الكامل للتصنيف غير المتوازن يتفرع إلى صفحات تفصيلية:")
_TRACK = (("imb_nature", "لماذا تفشل النماذج؟ IR، الندرة المطلقة، التداخل، قاعدة بايز"),
          ("imb_metrics", "ROC مقابل PR، MCC، G-mean، F-beta، فترات الثقة"),
          ("imb_cost_sensitive", "مصفوفة التكلفة، عتبة Elkan، class_weight، Focal loss"),
          ("imb_oversampling", "ROS، SMOTE، Borderline، SVMSMOTE، ADASYN، SMOTENC"),
          ("imb_undersampling", "NearMiss، Tomek، ENN، CNN، OSS، NCR، الهجين"),
          ("imb_ensembles", "BalancedRandomForest، BalancedBagging، EasyEnsemble، RUSBoost"),
          ("imb_probabilities", "المعايرة، تصحيح الأولوية، تقدير انتشار جديد بـEM"),
          ("imb_multiclass_extreme", "متعدد الفئات، الندرة القصوى، كشف الشذوذ"),
          ("imb_workflow", "منهجية من 10 خطوات ومختبر مقارنة شامل"))
cols = st.columns(3)
for i, (mid, desc) in enumerate(_TRACK):
    with cols[i % 3]:
        page_link(mid)
        st.caption(desc)

st.markdown("## الأدوات")
comparison_table([
    {"الاستراتيجية": "Stratification", "أين": "التقسيم وCV", "الأثر": "نسب ثابتة في كل طية"},
    {"الاستراتيجية": "class_weight='balanced'", "أين": "التدريب", "الأثر": "وزن الفئة النادرة ∝ 1/انتشارها؛ يشوّه الاحتمالات"},
    {"الاستراتيجية": "sample_weight", "أين": "fit", "الأثر": "أوزان فردية (تكاليف لكل حالة)"},
    {"الاستراتيجية": "Threshold tuning", "أين": "القرار", "الأثر": "يحرك نقطة التشغيل دون تغيير النموذج"},
    {"الاستراتيجية": "Random undersampling", "أين": "بيانات التدريب داخل الطية", "الأثر": "يضيع بيانات الأغلبية"},
    {"الاستراتيجية": "Random oversampling", "أين": "بيانات التدريب داخل الطية", "الأثر": "تكرار ⇒ خطر حفظ"},
    {"الاستراتيجية": "SMOTE", "أين": "بيانات التدريب داخل الطية", "الأثر": "نقاط اصطناعية بالاستيفاء بين الجيران"},
])
formula(r"x_{\text{new}} = x_i + \lambda\,(x_{nn} - x_i),\quad \lambda\sim U(0,1)", title="SMOTE (Chawla et al., 2002)",
        symbols={"x_i": "مثال من الفئة النادرة", "x_{nn}": "أحد أقرب k جيران له من الفئة نفسها"},
        intuition="يولّد أمثلة على القطع المستقيمة بين الأمثلة النادرة وجيرانها — لا ينسخها حرفيًا.")
warning("Resampling يحدث **داخل الـPipeline وعلى بيانات التدريب فقط في كل طية**. Oversampling قبل CV يضع نسخًا (أو نقاطًا "
        "مستوفاة) من الأمثلة نفسها في التدريب والتحقق ⇒ تسرب ودرجات مضخّمة.")
mermaid("""
flowchart LR
  A[fold train] --> S[SMOTE / undersample] --> M[fit model]
  B[fold validation — untouched, real prevalence] --> E[evaluate]
  M --> E
""")

st.markdown("## SMOTE بصريًا")
imb = optional("imblearn")
rng = np.random.default_rng(0)
Xs = np.vstack([rng.normal([0, 0], 1, (190, 2)), rng.normal([2.2, 2.2], 0.6, (10, 2))])
ys = np.r_[np.zeros(190, int), np.ones(10, int)]
if imb is None:
    missing_notice("imblearn")
else:
    from imblearn.over_sampling import SMOTE
    k = st.slider("k_neighbors", 1, 9, 5, key="imb_k")
    Xr, yr = SMOTE(k_neighbors=k, random_state=0).fit_resample(Xs, ys)
    new = np.arange(len(yr)) >= len(ys)
    fig = scatter_classes(Xs, ys)
    fig.add_trace(go.Scatter(x=Xr[new, 0], y=Xr[new, 1], mode="markers", name="synthetic (SMOTE)",
                             marker=dict(symbol="star", size=8, color=PALETTE["amber"], line=dict(width=0.5, color="#212529"))))
    fig.update_layout(title=f"SMOTE: {new.sum()} synthetic minority points between real neighbours", height=400)
    plot(fig)

st.markdown("## Imbalance Lab: مقارنة الاستراتيجيات بنفس الطيات")
X, y = xy("imbalanced")


def _strategies():
    base = lambda **kw: LogisticRegression(max_iter=2000, **kw)  # noqa: E731
    s = {"baseline (no treatment)": make_pipeline(StandardScaler(), base()),
         "class_weight='balanced'": make_pipeline(StandardScaler(), base(class_weight="balanced")),
         "threshold tuned (F1, inner CV)": TunedThresholdClassifierCV(make_pipeline(StandardScaler(), base()), scoring="f1", cv=3)}
    if imb is not None:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import make_pipeline as imb_pipe
        from imblearn.under_sampling import RandomUnderSampler
        s["SMOTE inside pipeline"] = imb_pipe(StandardScaler(), SMOTE(random_state=0), base())
        s["undersampling inside pipeline"] = imb_pipe(StandardScaler(), RandomUnderSampler(random_state=0), base())
    return s


@st.cache_data(show_spinner="5 طيات × عدة استراتيجيات…")
def _run(has_imb: bool):
    rows = []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    for name, model in _strategies().items():
        m = {"recall": [], "F1": [], "balanced acc": [], "PR-AUC": [], "ROC-AUC": [], "mean p": []}
        for tr, te in folds:
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            est = model.estimator_ if hasattr(model, "estimator_") else model
            p = est.predict_proba(X.iloc[te])[:, 1]
            yt = y.iloc[te]
            m["recall"].append(recall_score(yt, pred))
            m["F1"].append(f1_score(yt, pred))
            m["balanced acc"].append(balanced_accuracy_score(yt, pred))
            m["PR-AUC"].append(average_precision_score(yt, p))
            m["ROC-AUC"].append(roc_auc_score(yt, p))
            m["mean p"].append(p.mean())
        rows.append({"strategy": name, **{k: float(np.mean(v)) for k, v in m.items()}})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="imb_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["imb_done"] = True
if st.session_state.get("imb_done"):
    res = _run(imb is not None)
    st.dataframe(res.round(3), hide_index=True, width="stretch")
    st.caption(f"الانتشار الحقيقي ≈ {y.mean():.3f}. قارن «mean p»: class_weight وResampling ترفع متوسط الاحتمالات بعيدًا عن "
               "الانتشار الحقيقي (فقدان المعايرة)، بينما PR-AUC وROC-AUC (جودة الترتيب) تتغير قليلًا — معظم المكسب في Recall "
               "يأتي من إزاحة العتبة فعليًا.")
why("ابدأ بالأبسط: نموذج جيد + مقياس مناسب (PR-AUC) + عتبة مضبوطة.",
    "Resampling ليس علاجًا سحريًا؛ غالبًا ما يعادل أثره تحريك العتبة، مع تكلفة تشويه الاحتمالات.")
intuition("المشكلة الحقيقية غالبًا ليست «عدم التوازن» بل قلة الأمثلة الموجبة المطلقة. 50 مثالًا نادرًا من 1000 أصعب من 5000 من 100,000.")

if at_least("advanced"):
    st.markdown("## متقدم: تصحيح الاحتمالات بعد إعادة الموازنة")
    formula(r"p = \frac{p_s\,\pi/\pi_s}{p_s\,\pi/\pi_s + (1-p_s)(1-\pi)/(1-\pi_s)}", title="Prior-shift correction",
            symbols={r"\pi": "الانتشار الحقيقي", r"\pi_s": "الانتشار بعد إعادة الموازنة", "p_s": "الاحتمال من النموذج المُعاد موازنته"})
if at_least("research"):
    researcher_note(["van den Goorbergh et al. (2022) وجدوا في النمذجة السريرية أن تصحيح عدم التوازن أفسد المعايرة دون تحسين التمييز.",
                     "أبلغ عن الانتشار في كل مجموعة، والمقاييس المستقلة عن العتبة، ونقطة التشغيل المختارة وسببها."])
    st.markdown(cite("chawla2002", "he2009", "vandengoorbergh2022", "lemaitre2017"))
mistakes(["SMOTE قبل CV.", "Accuracy كمقياس.", "تقييم على بيانات مُعاد موازنتها بدل الانتشار الحقيقي.",
          "استخدام احتمالات نموذج class_weight كاحتمالات حقيقية."])
page_footer("imbalanced",
            takeaways=["الأدوات: الطبقية، الأوزان، العتبة، إعادة العيّنة.", "Resampling داخل الـPipeline وعلى التدريب فقط.",
                       "كثيرًا ما تكفي العتبة المضبوطة + PR-AUC."])
