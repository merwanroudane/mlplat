import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.algorithm_profile import hyperparameter_table
from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import xy
from utils.plotting import plot

page_header("calibration")

definition("Calibration", "المصنّف معايَر إذا كان من بين كل الحالات التي أعطاها احتمال 0.7، تكون 70% منها موجبة فعلًا: "
           "$P(Y=1 \\mid \\hat p(X) = p) = p$.")
formula(r"\text{Brier} = \frac1n\sum_i(\hat p_i - y_i)^2,\qquad \text{ECE} = \sum_b \frac{n_b}{n}\big|\bar y_b - \bar p_b\big|",
        title="Brier score and expected calibration error",
        intuition="Brier يمزج المعايرة والتمييز؛ ECE يقيس الفجوة بين الاحتمال المتوسط والتكرار الفعلي في كل صندوق.")
comparison_table([
    {"الطريقة": "sigmoid (Platt)", "الشكل": "p' = σ(a·s + b) — معلمتان", "البيانات المطلوبة": "قليلة", "متى": "تشوّه على شكل S (SVM، التعزيز)"},
    {"الطريقة": "isotonic", "الشكل": "دالة درجية رتيبة غير معلمية", "البيانات المطلوبة": "كثيرة (> ~1000)", "متى": "أي تشوه رتيب"},
    {"الطريقة": "temperature (منذ 1.8)", "الشكل": "softmax(z / T) — معلمة واحدة", "البيانات المطلوبة": "قليلة", "متى": "متعدد الفئات؛ الثقة المفرطة"},
])
hyperparameter_table("CalibratedClassifierCV")

st.markdown("## Calibration Lab")
X, y = xy("classification")
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
c1, c2 = st.columns(2)
base_name = c1.selectbox("النموذج الأساسي", ["GaussianNB", "SVC (decision_function)", "RandomForest", "LogisticRegression"],
                         key="cal_base")
method = c2.segmented_control("طريقة المعايرة", ["sigmoid", "isotonic", "temperature"], default="sigmoid", key="cal_method",
                              required=True)


def _base(name):
    return {"GaussianNB": GaussianNB(), "SVC (decision_function)": make_pipeline(StandardScaler(), SVC()),
            "RandomForest": RandomForestClassifier(200, random_state=0, n_jobs=1),
            "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}[name]


@st.cache_data(show_spinner="يعاير…", max_entries=32)
def _run(base_name: str, method: str):
    base = _base(base_name).fit(Xa, ya)
    if hasattr(base, "predict_proba"):
        p_raw = base.predict_proba(Xb)[:, 1]
    else:
        s = base.decision_function(Xb)
        p_raw = (s - s.min()) / (s.max() - s.min())  # min-max scaled scores: NOT probabilities
    cal = CalibratedClassifierCV(_base(base_name), method=method, cv=5).fit(Xa, ya)
    p_cal = cal.predict_proba(Xb)[:, 1]
    return p_raw, p_cal


p_raw, p_cal = _run(base_name, method)
fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(color=PALETTE["muted"], dash="dot")))
rows = []
for i, (name, p) in enumerate((("uncalibrated", p_raw), (f"calibrated ({method})", p_cal))):
    frac, mean_p = calibration_curve(yb, p, n_bins=10, strategy="quantile")
    fig.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers", name=name, line=dict(color=SEQUENCE[i], width=3),
                             marker=dict(symbol=["circle", "diamond"][i], size=9)))
    bins = np.clip(np.digitize(p, np.quantile(p, np.linspace(0, 1, 11)[1:-1])), 0, 9)
    ece = sum((bins == b).mean() * abs(yb.to_numpy()[bins == b].mean() - p[bins == b].mean()) for b in range(10) if (bins == b).any())
    rows.append({"version": name, "Brier": brier_score_loss(yb, p), "log loss": log_loss(yb, np.clip(p, 1e-6, 1 - 1e-6)),
                 "ECE": ece, "ROC-AUC": roc_auc_score(yb, p)})
fig.update_layout(title="Reliability diagram (quantile bins)", xaxis_title="mean predicted probability",
                  yaxis_title="observed fraction of positives", height=420)
c1, c2 = st.columns([1.4, 1])
with c1:
    plot(fig)
with c2:
    st.dataframe(pd.DataFrame(rows).set_index("version").T.round(4), width="stretch")
    fig = go.Figure()
    for i, (name, p) in enumerate((("uncalibrated", p_raw), ("calibrated", p_cal))):
        fig.add_trace(go.Histogram(x=p, nbinsx=25, name=name, opacity=0.6, marker_color=SEQUENCE[i]))
    fig.update_layout(barmode="overlay", title="Distribution of predicted probabilities", height=260)
    plot(fig)
intuition("المعايرة لا تغيّر الترتيب كثيرًا (AUC شبه ثابت) لكنها تجعل الأرقام قابلة للاستخدام كاحتمالات. NB مفرط الثقة (نقاطه "
          "تحت القطر عند الأطراف)، وRF معتدل الثقة (يتجنب الأطراف)، وSVM لا يعطي احتمالات أصلًا.")
why("عاير على بيانات لم يُدرَّب عليها النموذج.", "CalibratedClassifierCV يفعل ذلك بـcv؛ معايرة على بيانات التدريب نفسها تتعلم "
    "ثقة مفرطة. ومع ensemble=False يُدرَّب مصنف واحد على كل البيانات ويُعاير بتنبؤات Cross-validation.")

if at_least("advanced"):
    st.markdown("## متقدم: أي النماذج تحتاج معايرة؟")
    comparison_table([
        {"النموذج": "LogisticRegression", "المعايرة الأصلية": "جيدة غالبًا (تُحسّن Log loss مباشرة)"},
        {"النموذج": "Naive Bayes", "المعايرة الأصلية": "مفرطة الثقة"},
        {"النموذج": "Random Forest", "المعايرة الأصلية": "ثقة ناقصة قرب 0 و1"},
        {"النموذج": "Gradient boosting", "المعايرة الأصلية": "مقبولة إلى جيدة مع Log loss؛ تتدهور مع التوقف المتأخر"},
        {"النموذج": "SVM", "المعايرة الأصلية": "لا احتمالات: تحتاج Platt/isotonic"},
        {"النموذج": "Neural networks", "المعايرة الأصلية": "الحديثة مفرطة الثقة غالبًا ⇒ Temperature scaling"},
    ])
if at_least("research"):
    researcher_note(["في DML-IRM، الأوزان D/m̂(X) حساسة جدًا لمعايرة درجة الميل قرب 0 و1؛ عاير أو قص (trimming).",
                     "Guo et al. (2017): Temperature scaling بمعامل واحد يكفي غالبًا للشبكات العميقة؛ أضيف إلى scikit-learn في 1.8."])
    st.markdown(cite("niculescu2005", "zadrozny2002", "guo2017"))
mistakes(["معايرة على بيانات التدريب.", "isotonic مع بيانات قليلة.", "استخدام درجات SVM كاحتمالات.",
          "الحكم على المعايرة بـAUC."])
page_footer("calibration",
            takeaways=["المعايرة: الاحتمال 0.7 يعني 70% فعلًا.", "Reliability diagram وBrier وECE أدوات التشخيص.",
                       "sigmoid للبيانات القليلة، isotonic للكثيرة، temperature لمتعدد الفئات (منذ 1.8)."])
