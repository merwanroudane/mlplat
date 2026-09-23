import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from components.algorithm_profile import algorithm_profile, hyperparameter_table
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset
from utils.plotting import grid_for, plot

page_header("anomaly_detection")
algorithm_profile("isolation_forest")

st.markdown("## خمسة مفاهيم لا تُخلط")
comparison_table([
    {"المفهوم": "Outlier", "التعريف": "ملاحظة بعيدة إحصائيًا عن البقية", "مثال": "راتب 50 ضعف الوسيط", "الإجراء": "تحقق، قد تكون حقيقية"},
    {"المفهوم": "Anomaly", "التعريف": "نمط لا يتوافق مع السلوك المتوقع (معنى مجالي)", "مثال": "معاملة من بلدين في دقيقة", "الإجراء": "تحقيق"},
    {"المفهوم": "Novelty", "التعريف": "جديد مقارنة ببيانات تدريب «نظيفة»", "مثال": "نوع عطل لم يُرَ", "الإجراء": "novelty detection (تدريب على الطبيعي)"},
    {"المفهوم": "Data error", "التعريف": "خطأ إدخال/قياس", "مثال": "عمر = 999", "الإجراء": "تصحيح/تنظيف (DSplat)"},
    {"المفهوم": "Rare event", "التعريف": "حدث حقيقي نادر وله تسميات", "مثال": "احتيال مُعلَّم", "الإجراء": "تصنيف غير متوازن إن توفرت التسميات"},
])
why("إن كانت لديك تسميات كافية للحالات النادرة فاستخدم تصنيفًا (وحدة عدم التوازن).",
    "كشف الشذوذ غير الموجّه يفترض أن الغريب = المهم، وهذا ليس صحيحًا دائمًا.")

st.markdown("## Isolation Forest: Animation للعزل")
df = load_dataset("anomalies")
X = df[["x1", "x2"]].to_numpy()
truth = df["is_anomaly"].to_numpy()
target_idx = int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))  # a far point
normal_idx = int(np.argmin(np.linalg.norm(X - X[:300].mean(0), axis=1)))  # a central point
which = st.segmented_control("النقطة المراد عزلها", ["شاذة (بعيدة)", "طبيعية (مركزية)"], default="شاذة (بعيدة)", key="ad_which",
                             required=True)
idx = target_idx if which.startswith("شاذة") else normal_idx


@st.cache_data(show_spinner=False)
def _isolation_path(idx: int, seed: int = 0, max_steps: int = 14):
    rng = np.random.default_rng(seed)
    box = np.array([X.min(0), X.max(0)])
    pts = np.arange(len(X))
    steps = [(box.copy(), pts.copy(), None)]
    for _ in range(max_steps):
        if len(pts) <= 1:
            break
        j = rng.integers(2)
        lo, hi = X[pts, j].min(), X[pts, j].max()
        if lo == hi:
            break
        t = rng.uniform(lo, hi)
        side = X[idx, j] <= t
        pts = pts[(X[pts, j] <= t) == side]
        box = box.copy()
        if side:
            box[1, j] = t
        else:
            box[0, j] = t
        steps.append((box.copy(), pts.copy(), (j, t)))
    return steps


steps = _isolation_path(idx)


def _iso_frame(i: int) -> None:
    box, pts, cut = steps[i]
    fig = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color="#CED4DA", size=5), name="all points"))
    fig.add_trace(go.Scatter(x=X[pts, 0], y=X[pts, 1], mode="markers", marker=dict(color=PALETTE["sky"], size=7), name="still together"))
    fig.add_shape(type="rect", x0=box[0, 0], y0=box[0, 1], x1=box[1, 0], y1=box[1, 1], line=dict(color=PALETTE["purple"], width=2))
    fig.add_trace(go.Scatter(x=[X[idx, 0]], y=[X[idx, 1]], mode="markers", name="target",
                             marker=dict(symbol="star", size=18, color=PALETTE["coral"])))
    fig.update_layout(title=f"Random split {i}: {len(pts)} point(s) remain in the target's cell" +
                      (" — isolated!" if len(pts) == 1 else ""), height=420)
    plot(fig)


stepper(f"iso_{idx}", len(steps), _iso_frame, labels=[f"split {i}" for i in range(len(steps))])
intuition(f"عزل النقطة الشاذة احتاج {len(steps) - 1} تقسيمًا عشوائيًا فقط؛ النقطة المركزية تحتاج عادة أكثر بكثير. متوسط طول المسار "
          "عبر أشجار كثيرة = درجة الشذوذ: قصير ⇒ شاذ.")

st.markdown("## Anomaly Lab: أربع خوارزميات على البيانات نفسها")
contamination = st.slider("contamination (النسبة المتوقعة)", 0.01, 0.15, 0.05, 0.01, key="ad_cont")
Xs = StandardScaler().fit_transform(X)


@st.cache_resource(show_spinner="يدرّب الخوارزميات الأربع…", max_entries=16)
def _models(contamination: float):
    return {
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=contamination, random_state=0).fit(Xs),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True).fit(Xs),
        "One-Class SVM": OneClassSVM(nu=contamination, gamma="scale").fit(Xs),
        "Elliptic Envelope": EllipticEnvelope(contamination=contamination, random_state=0).fit(Xs),
    }


models = _models(contamination)
xx, yy, grid = grid_for(Xs, steps=100, pad=0.6)
rows = []
cols = st.columns(2)
for i, (name, m) in enumerate(models.items()):
    pred = m.predict(Xs) == -1
    score = -m.decision_function(Xs)
    rows.append({"method": name, "flagged": int(pred.sum()), "precision": precision_score(truth, pred, zero_division=0),
                 "recall": recall_score(truth, pred), "AP (score)": average_precision_score(truth, score)})
    Zg = m.decision_function(grid).reshape(xx.shape)
    fig = go.Figure(go.Contour(x=xx[0], y=yy[:, 0], z=Zg, colorscale=[[0, "#FFE8CC"], [0.5, "#FCFCFF"], [1, "#D0EBFF"]],
                               contours=dict(showlines=False), showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Zg, contours=dict(start=0, end=0, size=1, coloring="none"),
                             line=dict(color="#212529", width=2), showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=Xs[~pred, 0], y=Xs[~pred, 1], mode="markers", marker=dict(color=PALETTE["sky"], size=5), name="normal"))
    fig.add_trace(go.Scatter(x=Xs[pred, 0], y=Xs[pred, 1], mode="markers", marker=dict(color=PALETTE["coral"], symbol="x", size=8),
                             name="flagged"))
    fig.add_trace(go.Scatter(x=Xs[truth == 1, 0], y=Xs[truth == 1, 1], mode="markers", name="true anomaly",
                             marker=dict(color="rgba(0,0,0,0)", size=12, line=dict(color="#212529", width=1.5))))
    fig.update_layout(title=name, height=360, showlegend=i == 0)
    with cols[i % 2]:
        plot(fig)
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
st.caption("التقييم ممكن هنا فقط لأن البيانات اصطناعية بشذوذ معروف. AP يقيّم الدرجة المستمرة (مستقل عن contamination)؛ "
           "Precision/Recall تعتمد على العتبة التي يحددها contamination.")
comparison_table([
    {"الخوارزمية": "Isolation Forest", "الفكرة": "طول مسار العزل", "يناسب": "أبعاد عالية، كبير الحجم", "حدود": "شذوذ محلي بين كتل"},
    {"الخوارزمية": "LOF", "الفكرة": "كثافة نسبية للجيران", "يناسب": "شذوذ محلي، كثافات متفاوتة", "حدود": "O(n²)، novelty=True للبيانات الجديدة"},
    {"الخوارزمية": "One-Class SVM", "الفكرة": "حدود حول الطبيعي في فضاء النواة", "يناسب": "حدود مرنة", "حدود": "حساس لـnu وgamma، مكلف"},
    {"الخوارزمية": "Elliptic Envelope", "الفكرة": "تغاير متين (MCD) + مسافة Mahalanobis", "يناسب": "بيانات طبيعية أحادية الكتلة",
     "حدود": "يفشل مع كتل متعددة"},
])
hyperparameter_table("IsolationForest")

if at_least("advanced"):
    st.markdown("## متقدم: الدرجة")
    st.latex(r"s(x, n) = 2^{-\mathbb E[h(x)]/c(n)},\qquad c(n) = 2H(n-1) - \tfrac{2(n-1)}{n}")
    st.markdown("h(x) طول المسار، وc(n) متوسط طول المسار في شجرة بحث ثنائية ⇒ s قرب 1 شاذ، قرب 0.5 عادي (Liu et al., 2008). "
                "في scikit-learn: `score_samples` = −s، و`decision_function` = score_samples − offset_.")
if at_least("research"):
    researcher_note(["contamination يحدد العتبة فقط؛ الدرجة المستمرة لا تتغير — أبلغ عن AP/درجات لا عن تسميات فقط.",
                     "Outlier detection (التدريب ملوث) مقابل Novelty detection (التدريب نظيف) مسألتان مختلفتان إحصائيًا."])
    st.markdown(cite("liu2008", "breunig2000", "scholkopf2001"))
mistakes(["اعتبار كل شذوذ خطأ وحذفه.", "Elliptic Envelope على بيانات متعددة الكتل.", "LOF دون novelty=True للتنبؤ بجديد.",
          "ضبط contamination على Test."])
page_footer("anomaly_detection",
            takeaways=["فرّق بين Outlier وAnomaly وNovelty وData error وRare event.", "Isolation Forest: الشاذ يُعزل بسرعة.",
                       "كل خوارزمية تفترض تعريفًا مختلفًا لـ«الطبيعي»."])
