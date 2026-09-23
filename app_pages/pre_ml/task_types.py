import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, real_world, researcher_note
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset, toy_2d
from utils.plotting import plot, scatter_classes

page_header("task_types")

st.markdown("## خريطة أنواع المسائل")
comparison_table([
    {"المسألة": "Regression", "المخرج": "عدد حقيقي", "مثال": "سعر منزل", "Loss شائعة": "MSE, MAE, Huber",
     "مقياس": "RMSE, MAE, R²"},
    {"المسألة": "Binary classification", "المخرج": "فئة من اثنتين + احتمال", "مثال": "احتيال/سليم",
     "Loss شائعة": "Log loss, Hinge", "مقياس": "ROC-AUC, PR-AUC, F1"},
    {"المسألة": "Multiclass", "المخرج": "فئة من k", "مثال": "نوع الوثيقة", "Loss شائعة": "Cross-entropy",
     "مقياس": "Macro-F1, accuracy"},
    {"المسألة": "Multilabel", "المخرج": "مجموعة فئات", "مثال": "وسوم مقال", "Loss شائعة": "Log loss لكل وسم",
     "مقياس": "Hamming loss, micro-F1"},
    {"المسألة": "Ordinal", "المخرج": "فئة مرتبة", "مثال": "تقييم 1–5 نجوم", "Loss شائعة": "Ordinal logit / MAE",
     "مقياس": "MAE على الرتب، Kendall τ"},
    {"المسألة": "Ranking", "المخرج": "ترتيب", "مثال": "نتائج بحث", "Loss شائعة": "Pairwise / LambdaRank",
     "مقياس": "NDCG, MAP"},
    {"المسألة": "Clustering", "المخرج": "مجموعات بلا تسميات", "مثال": "تجزئة العملاء", "Loss شائعة": "Inertia, likelihood",
     "مقياس": "Silhouette + حكم المجال"},
    {"المسألة": "Anomaly detection", "المخرج": "درجة شذوذ", "مثال": "عطل آلة", "Loss شائعة": "حسب الطريقة",
     "مقياس": "Precision@k إن توفرت تسميات"},
    {"المسألة": "Forecasting", "المخرج": "قيم مستقبلية", "مثال": "مبيعات الأسبوع القادم", "Loss شائعة": "MSE/MAE/Pinball",
     "مقياس": "MASE, sMAPE (بحذر)"},
])

st.markdown("## أشكال الإشراف")
comparison_table([
    {"النوع": "Supervised", "ماذا يتوفر؟": "X وy", "مثال": "تصنيف رسائل مزعجة بأمثلة مُعلَّمة"},
    {"النوع": "Unsupervised", "ماذا يتوفر؟": "X فقط", "مثال": "اكتشاف مجموعات عملاء"},
    {"النوع": "Semi-supervised", "ماذا يتوفر؟": "X كثير + y لجزء صغير", "مثال": "آلاف الصور و100 تسمية"},
    {"النوع": "Self-supervised", "ماذا يتوفر؟": "X ومهمة مصطنعة منها", "مثال": "توقّع الكلمة المحجوبة"},
    {"النوع": "Reinforcement", "ماذا يتوفر؟": "مكافآت بعد أفعال", "مثال": "وكيل يتعلم لعبة"},
])

st.markdown("## صورة كل مسألة")
kind = st.segmented_control("اعرض", ["Regression", "Classification", "Clustering", "Anomaly detection"],
                            default="Classification", key="tt_kind", required=True)
if kind == "Regression":
    df = load_dataset("regression")
    fig = go.Figure(go.Scatter(x=df["x1"], y=df["y"], mode="markers", marker=dict(color=PALETTE["sky"], opacity=0.6)))
    fig.update_layout(title="Regression: continuous y against a feature", xaxis_title="x1", yaxis_title="y")
    plot(fig, height=380)
elif kind == "Classification":
    X, y = toy_2d("moons", n=300)
    plot(scatter_classes(X, y).update_layout(title="Classification: discrete labels (colour + shape)"), height=380)
elif kind == "Clustering":
    df = load_dataset("blobs")
    fig = go.Figure(go.Scatter(x=df["x1"], y=df["x2"], mode="markers", marker=dict(color=PALETTE["muted"], opacity=0.7)))
    fig.update_layout(title="Clustering: no labels — the algorithm must find structure", xaxis_title="x₁", yaxis_title="x₂")
    plot(fig, height=380)
else:
    df = load_dataset("anomalies")
    fig = go.Figure(go.Scatter(x=df["x1"], y=df["x2"], mode="markers",
                               marker=dict(color=np.where(df["is_anomaly"] == 1, PALETTE["coral"], PALETTE["sky"]),
                                           symbol=np.where(df["is_anomaly"] == 1, "x", "circle"), size=8)))
    fig.update_layout(title="Anomaly detection: rare points far from the bulk (× = true anomaly)",
                      xaxis_title="x₁", yaxis_title="x₂")
    plot(fig, height=380)

st.markdown("## تمرين تصنيف سريع")
scenarios = {
    "توقّع عدد الزيارات إلى موقع غدًا": "Forecasting (Regression على بيانات زمنية)",
    "اختيار أي 10 منتجات نعرضها لكل مستخدم": "Ranking",
    "تحديد ما إذا كانت صورة الأشعة تحتوي كسرًا": "Binary classification",
    "تجميع المقالات دون أي تسميات": "Clustering",
    "اكتشاف معاملات بطاقة غير معتادة دون أمثلة احتيال": "Anomaly detection",
    "تقييم رضا العميل من 1 إلى 5": "Ordinal classification",
    "وسم مقال بعدة موضوعات": "Multilabel classification",
}
choice = st.selectbox("اختر سيناريو", list(scenarios), key="tt_sc")
with st.expander("اعرض نوع المسألة"):
    st.success(scenarios[choice])
intuition("السؤال الحاسم: **ما شكل المخرج؟** رقم، فئة، مجموعة فئات، ترتيب، أو لا تسميات أصلًا. ثم: هل للزمن دور؟")

if at_least("advanced"):
    st.markdown("## متقدم: المهمة تحدد الـLoss والـMetric والتقسيم")
    st.markdown("- المسائل الزمنية تفرض **تقسيمًا زمنيًا** مهما كانت الخوارزمية.\n"
                "- تحويل Regression إلى Classification (مثل «مبيعات > 100») يضيع معلومات ويخلق عتبة اعتباطية.\n"
                "- التصنيف متعدد الفئات يحتاج قرارًا: Macro أم Micro averaging، حسب أهمية الفئات النادرة.")
if at_least("research"):
    researcher_note(["في البحث التطبيقي، صِف المهمة بصيغة رسمية: فضاء المدخلات، فضاء المخرجات، والخسارة.",
                     "Ordinal outcomes تُعالج خطأً كثيرًا كـMulticlass؛ فقدان الترتيب يضيع قوة إحصائية."])
real_world(["ما القرار الذي سيُتخذ بناءً على المخرج؟", "هل التسميات متوفرة وموثوقة؟", "هل المخرج مستقبلي؟"])
page_footer("task_types",
            takeaways=["شكل المخرج يحدد نوع المسألة.", "نوع المسألة يحدد الخسارة والمقياس ومخطط التقسيم.",
                       "Unsupervised لا يعني بلا تقييم: يحتاج معرفة المجال."],
            mistakes=["معاملة Ordinal كـMulticlass عادي.", "تحويل Regression إلى تصنيف دون سبب.",
                      "تجاهل البعد الزمني في Forecasting."])
