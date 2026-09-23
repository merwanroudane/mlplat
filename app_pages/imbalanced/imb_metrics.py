import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import xy
from utils.imbalance import bootstrap_ci, imbalance_metrics
from utils.plotting import plot

page_header("imb_metrics")

st.markdown("## مفارقة الدقة · The accuracy paradox")
st.markdown("مع 1% احتيال، مصنّف يقول «ليس احتيالًا» دائمًا يحقق Accuracy = 99% ولا يكتشف حالة واحدة. كل مقياس يجيب عن سؤال "
            "مختلف؛ المهم أن تختار السؤال الصحيح.")
comparison_table([
    {"المقياس": "Precision", "الصيغة": "TP/(TP+FP)", "السؤال": "من الإنذارات، كم كان صحيحًا؟", "حساس للانتشار؟": "نعم بشدة"},
    {"المقياس": "Recall (Sensitivity)", "الصيغة": "TP/(TP+FN)", "السؤال": "من الحالات النادرة، كم التقطنا؟", "حساس للانتشار؟": "لا"},
    {"المقياس": "Specificity", "الصيغة": "TN/(TN+FP)", "السؤال": "من السالبة، كم تركنا بسلام؟", "حساس للانتشار؟": "لا"},
    {"المقياس": "F-beta", "الصيغة": "(1+β²)PR/(β²P+R)", "السؤال": "توازن P وR؛ β = 2 يعطي Recall وزنًا أكبر", "حساس للانتشار؟": "نعم"},
    {"المقياس": "Balanced accuracy", "الصيغة": "(Recall + Specificity)/2", "السؤال": "متوسط الدقة لكل فئة", "حساس للانتشار؟": "لا"},
    {"المقياس": "G-mean", "الصيغة": "√(Recall × Specificity)", "السؤال": "يعاقب إهمال أي فئة", "حساس للانتشار؟": "لا"},
    {"المقياس": "MCC", "الصيغة": "انظر أدناه", "السؤال": "ارتباط التنبؤ بالحقيقة باستخدام الخلايا الأربع", "حساس للانتشار؟": "نعم"},
    {"المقياس": "ROC-AUC", "الصيغة": "P(s⁺ > s⁻)", "السؤال": "جودة الترتيب بين موجب وسالب عشوائيين", "حساس للانتشار؟": "لا"},
    {"المقياس": "PR-AUC / Average precision", "الصيغة": "Σ (Rₙ − Rₙ₋₁) Pₙ", "السؤال": "جودة الترتيب من منظور الإنذارات",
     "حساس للانتشار؟": "نعم (خط الأساس = π)"},
    {"المقياس": "Brier / Log loss", "الصيغة": "متوسط (p − y)² / −log p_y", "السؤال": "جودة الاحتمالات نفسها", "حساس للانتشار؟": "نعم"},
])
formula(r"\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}",
        title="Matthews correlation coefficient",
        intuition="بين −1 و+1؛ صفر لمصنف عشوائي أو ثابت. لا يستطيع أن «يتجاهل» أي خلية من المصفوفة.",
        example="TP = 30، FN = 20، FP = 70، TN = 880: MCC = (26400 − 1400)/√(100·50·950·900) ≈ 0.38، بينما Accuracy = 0.91.")

st.markdown("## مختبر الانتشار · Prevalence Lab")
st.caption("نثبّت قدرة النموذج على الفصل (الدرجات: السالبة ~ N(0,1)، الموجبة ~ N(d,1)) ونغيّر الانتشار فقط. أي المقاييس يتغير؟")
d = st.slider("قوة الفصل d", 0.5, 3.0, 1.5, 0.25, key="prev_d")


@st.cache_data(show_spinner=False, max_entries=10)
def _prevalence(d: float):
    rng = np.random.default_rng(0)
    out, curves = [], {}
    for pi in (0.5, 0.2, 0.05, 0.01, 0.002):
        n = 50000
        n1 = max(int(n * pi), 60)
        s = np.r_[rng.normal(0, 1, n - n1), rng.normal(d, 1, n1)]
        y = np.r_[np.zeros(n - n1, int), np.ones(n1, int)]
        thr = d / 2  # the same operating point for every prevalence
        m = imbalance_metrics(y, 1 / (1 + np.exp(-(s - thr) * 3)), threshold=0.5)
        out.append({"prevalence": pi, "ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
                    "precision @ fixed thr": m["precision"], "recall @ fixed thr": m["recall"], "F1": m["F1"],
                    "MCC": m["MCC"], "balanced acc": m["balanced acc"]})
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        k = np.linspace(0, len(fpr) - 1, 300).astype(int)
        j = np.linspace(0, len(pr) - 1, 300).astype(int)
        curves[pi] = (fpr[k], tpr[k], rc[j], pr[j])
    return pd.DataFrame(out), curves


res, curves = _prevalence(d)
c1, c2 = st.columns(2)
roc_fig, pr_fig = go.Figure(), go.Figure()
for i, (pi, (fpr, tpr, rc, pr)) in enumerate(curves.items()):
    col = SEQUENCE[i % len(SEQUENCE)]
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"π = {pi}", line=dict(color=col, width=2)))
    pr_fig.add_trace(go.Scatter(x=rc, y=pr, name=f"π = {pi}", line=dict(color=col, width=2)))
roc_fig.update_layout(title="ROC: practically identical", xaxis_title="FPR", yaxis_title="TPR", height=380)
pr_fig.update_layout(title="Precision–Recall: collapses with prevalence", xaxis_title="Recall", yaxis_title="Precision", height=380)
with c1:
    plot(roc_fig)
with c2:
    plot(pr_fig)
st.dataframe(res.round(3), hide_index=True, width="stretch")
intuition("ROC يقسم على عدد السالبة (FPR) فلا يرى أن 1% من مليون سالب = 10,000 إنذار كاذب. Precision يراها مباشرة. "
          "لذلك مع أحداث نادرة، PR-AUC وPrecision عند Recall مطلوب يصفان تجربة المستخدم الحقيقية.")
definition("خط الأساس لـPR-AUC",
           "مصنّف عشوائي يحقق Average precision ≈ π (الانتشار)، لا 0.5. قارن PR-AUC دائمًا بالانتشار: 0.30 مع π = 0.01 ممتاز، "
           "و0.30 مع π = 0.25 ضعيف.")

st.markdown("## اختيار المقياس")
comparison_table([
    {"الحالة": "التكاليف معروفة (مبالغ، وقت محقق)", "المقياس المقترح": "التكلفة المتوقعة عند العتبة المختارة"},
    {"الحالة": "قائمة أولويات للمراجعة (Top-k)", "المقياس المقترح": "Precision@k أو Recall@k"},
    {"الحالة": "لا عتبة بعد؛ مقارنة نماذج للأحداث النادرة", "المقياس المقترح": "PR-AUC (Average precision) + ROC-AUC"},
    {"الحالة": "الاحتمالات ستُستخدم كمخاطر (طب، ائتمان)", "المقياس المقترح": "Brier / Log loss + Reliability diagram"},
    {"الحالة": "مقياس واحد متوازن لقرار ثنائي", "المقياس المقترح": "MCC أو Balanced accuracy"},
    {"الحالة": "تفويت الحالة أسوأ بكثير من الإنذار", "المقياس المقترح": "F2 أو Recall عند Precision أدنى محدد"},
])

st.markdown("## عدم اليقين حين تكون الموجبات قليلة")
st.caption("بيانات الاحتيال (5%)، Logistic regression. كلما صغر جزء الاختبار قلّ عدد الموجبات واتسعت فترة Bootstrap.")
test_frac = st.select_slider("حجم الاختبار", [0.1, 0.2, 0.3, 0.5], value=0.2, key="ci_frac")


@st.cache_data(show_spinner="Bootstrap طبقي…", max_entries=8)
def _ci(test_frac: float):
    X, y = xy("imbalanced")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=0, stratify=y)
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    yt = y_te.to_numpy()
    rows = []
    for name, fn in (("PR-AUC", average_precision_score), ("ROC-AUC", roc_auc_score)):
        lo, hi = bootstrap_ci(yt, p, fn, n_boot=300)
        rows.append({"metric": name, "estimate": fn(yt, p), "95% CI low": lo, "95% CI high": hi, "width": hi - lo})
    return pd.DataFrame(rows), int(yt.sum())


ci, npos = _ci(test_frac)
st.metric("عدد الموجبات في الاختبار", npos, border=True)
st.dataframe(ci.round(3), hide_index=True, width="stretch")
st.caption("فرق 0.03 في PR-AUC بين نموذجين قد يكون ضجيجًا تمامًا مع بضع عشرات من الموجبات. استخدم Repeated stratified CV "
           "وبلّغ عن الفترات.")

if at_least("advanced"):
    st.markdown("## متقدم: العلاقة بين Precision والانتشار")
    formula(r"\mathrm{Precision} = \frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi + \mathrm{FPR}\,(1-\pi)}",
            intuition="TPR وFPR خصائص للنموذج والعتبة؛ الانتشار π يأتي من العالم. نفس النموذج يعطي Precision مختلفًا في بيئة "
                      "بانتشار مختلف.")
if at_least("research"):
    researcher_note([
        "Average precision في scikit-learn مجموع خطوات (لا استيفاء شبه منحرف) — لا تقارن أرقامه بأرقام PR-AUC المحسوبة بالاستيفاء.",
        "اشتقاق Bootstrap طبقي يحفظ عدد الموجبات في كل عينة، فيعكس عدم اليقين دون تغيير الانتشار.",
        "MCC وF1 يعتمدان على الانتشار؛ عند مقارنة دراسات بانتشارات مختلفة استخدم مقاييس مستقلة عنه (TPR، FPR، ROC-AUC) إلى جانبها.",
    ])
why("بلّغ دائمًا عن: الانتشار، مقياس ترتيب (PR-AUC)، ونقطة التشغيل المختارة بمقاييسها (Precision، Recall).",
    "قارئ التقرير يحتاج أن يعرف كم إنذارًا سيراجع وكم حالة ستفوته.")
st.markdown("### المراجع")
st.markdown(cite("he2009", "lemaitre2017"))
mistakes(["Accuracy مع بيانات غير متوازنة.", "مقارنة PR-AUC بـ0.5 بدل الانتشار.",
          "الاكتفاء بـROC-AUC للأحداث النادرة جدًا.", "إعلان فوز نموذج بفرق أصغر من عرض فترة الثقة."])
page_footer("imb_metrics",
            takeaways=["ROC-AUC لا يتأثر بالانتشار؛ PR-AUC وPrecision يتأثران بقوة.",
                       "خط أساس PR-AUC هو الانتشار.",
                       "MCC وBalanced accuracy وG-mean مقاييس أحادية أكثر إنصافًا من Accuracy.",
                       "مع موجبات قليلة: فترات ثقة وRepeated CV."])
