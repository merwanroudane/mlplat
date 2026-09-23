import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from components.metric_explorer import metrics_table, threshold_explorer
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils import metrics as M
from utils.datasets import xy
from utils.plotting import plot

page_header("classification_metrics")

st.markdown("## مصفوفة الالتباس وكل ما يُشتق منها")
formula(r"\text{Precision}=\frac{TP}{TP+FP},\ \ \text{Recall}=\frac{TP}{TP+FN},\ \ \text{Specificity}=\frac{TN}{TN+FP},\ \ "
        r"F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}",
        title="Threshold-dependent metrics",
        symbols={"TP": "موجب متنبأ به موجبًا", "FP": "سالب متنبأ به موجبًا (إنذار كاذب)", "FN": "موجب فاتنا", "TN": "سالب صحيح",
                 r"\beta": "β > 1 يعطي Recall وزنًا أكبر"},
        intuition="Precision: «حين أقول نعم، كم مرة أصيب؟». Recall: «من كل الحالات الحقيقية، كم التقطت؟».",
        example="TP=40, FP=10, FN=20, TN=930 ⇒ P = 0.80، R = 0.667، F1 = 0.727، Accuracy = 0.97 (مضللة).")
comparison_table([
    {"المقياس": "Accuracy", "يعتمد على العتبة": "نعم", "حساس للانتشار": "جدًا", "متى": "فئات متوازنة وتكاليف متساوية"},
    {"المقياس": "Balanced accuracy", "يعتمد على العتبة": "نعم", "حساس للانتشار": "لا", "متى": "عدم توازن، أهمية متساوية للفئتين"},
    {"المقياس": "Precision / Recall / F1", "يعتمد على العتبة": "نعم", "حساس للانتشار": "Precision نعم", "متى": "التركيز على الفئة الموجبة"},
    {"المقياس": "MCC", "يعتمد على العتبة": "نعم", "حساس للانتشار": "قليلًا", "متى": "ملخص متوازن لكل خلايا المصفوفة"},
    {"المقياس": "ROC-AUC", "يعتمد على العتبة": "لا", "حساس للانتشار": "لا", "متى": "جودة الترتيب العامة"},
    {"المقياس": "Average precision (PR-AUC)", "يعتمد على العتبة": "لا", "حساس للانتشار": "نعم", "متى": "الفئة الموجبة نادرة ومهمة"},
    {"المقياس": "Log loss / Brier", "يعتمد على العتبة": "لا", "حساس للانتشار": "—", "متى": "حين تهم الاحتمالات نفسها"},
])

X, y = xy("imbalanced")
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)


@st.cache_data(show_spinner=False)
def _scores():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


proba = _scores()
yb_np = yb.to_numpy()

st.markdown("## Metric Explorer: حرّك العتبة")
st.caption(f"مجموعة تقييم: {len(yb)} معاملة، الاحتيال {yb.mean():.1%}. النموذج: Logistic Regression.")
threshold_explorer(yb_np, proba, key="cm_thr")

st.markdown("## ROC مقابل Precision–Recall")
fpr, tpr, _ = roc_curve(yb_np, proba)
prec, rec, _ = precision_recall_curve(yb_np, proba)
auc, ap = roc_auc_score(yb_np, proba), average_precision_score(yb_np, proba)
fpr_s, tpr_s, _ = M.roc_points(yb_np, proba)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, name=f"model (AUC {auc:.3f})", line=dict(color=PALETTE["sky"], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="random (0.5)", line=dict(color=PALETTE["muted"], dash="dot")))
    fig.update_layout(title="ROC curve", xaxis_title="FPR = 1 − specificity", yaxis_title="TPR = recall", height=360)
    plot(fig)
with c2:
    fig = go.Figure(go.Scatter(x=rec, y=prec, name=f"model (AP {ap:.3f})", line=dict(color=PALETTE["coral"], width=3)))
    fig.add_hline(y=yb.mean(), line=dict(color=PALETTE["muted"], dash="dot"), annotation_text=f"random = prevalence {yb.mean():.3f}")
    fig.update_layout(title="Precision–Recall curve", xaxis_title="recall", yaxis_title="precision", height=360)
    plot(fig)
st.caption(f"تحقق: AUC من تنفيذنا بالتعريف (utils/metrics.roc_points + شبه المنحرف) = {M.auc_trapezoid(fpr_s, tpr_s):.4f}، "
           f"وscikit-learn = {auc:.4f}.")
definition("ROC-AUC كاحتمال", "AUC = احتمال أن تحصل حالة موجبة مختارة عشوائيًا على درجة أعلى من حالة سالبة عشوائية.")
why("مع الندرة الشديدة أبلغ عن PR-AUC إلى جانب ROC-AUC.",
    "ROC يستخدم FPR ومقامه عدد السالبات الضخم، فيبقى صغيرًا حتى مع آلاف الإنذارات الكاذبة. خط الأساس لـPR هو الانتشار لا 0.5.")

st.markdown("## جدول ملخص")
summary = {"accuracy": (proba >= 0.5).astype(int).__eq__(yb_np).mean(),
           "balanced accuracy": balanced_accuracy_score(yb_np, proba >= 0.5), "F1": f1_score(yb_np, proba >= 0.5),
           "MCC": matthews_corrcoef(yb_np, proba >= 0.5), "ROC-AUC": auc, "average precision": ap,
           "log loss": log_loss(yb_np, proba), "Brier": brier_score_loss(yb_np, proba)}
st.dataframe(pd.DataFrame([summary]).round(4), hide_index=True, width="stretch")
st.dataframe(metrics_table(yb_np, proba).round(3), hide_index=True, width="stretch")

st.markdown("## Metric Selector")
c1, c2, c3 = st.columns(3)
imbalance = c1.segmented_control("الانتشار", ["متوازن", "نادر (<10%)"], default="نادر (<10%)", key="ms_imb", required=True)
cost = c2.segmented_control("التكلفة", ["متساوية", "FN أغلى", "FP أغلى"], default="FN أغلى", key="ms_cost", required=True)
need = c3.segmented_control("المخرج المطلوب", ["قرار", "ترتيب", "احتمال"], default="قرار", key="ms_need", required=True)
rec_ = []
if need == "احتمال":
    rec_ += ["Log loss وBrier + Reliability diagram (وحدة المعايرة)"]
if need == "ترتيب":
    rec_ += ["PR-AUC (Average precision)" if imbalance.startswith("نادر") else "ROC-AUC", "Precision@k إن كانت الموارد محدودة بعدد ثابت"]
if need == "قرار":
    if cost == "FN أغلى":
        rec_ += ["Recall عند Precision أدنى مقبول، أو F2", "التكلفة المتوقعة مع عتبة مضبوطة"]
    elif cost == "FP أغلى":
        rec_ += ["Precision عند Recall أدنى مقبول، أو F0.5", "التكلفة المتوقعة"]
    else:
        rec_ += ["Balanced accuracy أو MCC" if imbalance.startswith("نادر") else "Accuracy", "F1"]
st.success("**مقاييس مقترحة:**\n" + "\n".join(f"- {r}" for r in rec_) + "\n\nوأبلغ دائمًا عن Baseline ومصفوفة الالتباس.")

if at_least("advanced"):
    st.markdown("## متقدم: متعدد الفئات")
    st.markdown("- **Macro:** متوسط المقياس لكل فئة (كل فئة بالوزن نفسه).\n- **Micro:** اجمع TP/FP/FN عبر الفئات ثم احسب.\n"
                "- **Weighted:** Macro مرجّح بعدد عينات كل فئة.\n- ROC-AUC متعدد: `roc_auc_score(y, P, multi_class='ovr')`.")
if at_least("research"):
    researcher_note(["Log loss وBrier قواعد تسجيل مناسبة (Proper)؛ AUC ليس كذلك ولا يقيس المعايرة.",
                     "قارن AUC لنموذجين على نفس العينة باختبار DeLong أو Bootstrap مزدوج، لا بتداخل الفترات.",
                     "Chicco & Jurman (2020) يدافعان عن MCC كمقياس أحادي لمصفوفة ثنائية."])
mistakes(["Accuracy مع الندرة.", "مقارنة PR-AUC بين مجموعات بيانات بانتشار مختلف.", "اختيار العتبة على Test.",
          "اعتبار AUC مرتفعًا دليلًا على احتمالات جيدة."])
page_footer("classification_metrics",
            takeaways=["كل المقاييس المعتمدة على العتبة تُشتق من مصفوفة الالتباس.", "PR-AUC أنسب من ROC-AUC مع الندرة.",
                       "اختر المقياس من التكلفة والانتشار ونوع المخرج."])
