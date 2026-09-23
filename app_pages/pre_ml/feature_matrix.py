import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.dataset_viewer import dataset_card
from components.formulas import formula
from core.page import dsplat_link, page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset, xy
from utils.plotting import plot

page_header("feature_matrix")

st.markdown("## من الجدول إلى X وy")
definition("Design / feature matrix", "المصفوفة $X \\in \\mathbb{R}^{n\\times p}$: الصف $i$ هو الملاحظة $x_i$، والعمود $j$ "
           "هو الخاصية $j$. والهدف $y \\in \\mathbb{R}^n$ (أو فئات).")
formula(r"X=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p}\\ x_{21} & x_{22} & \cdots & x_{2p}\\ \vdots & & \ddots & \vdots\\ "
        r"x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix},\qquad y=\begin{bmatrix}y_1\\ \vdots\\ y_n\end{bmatrix}",
        title="الشكل الرياضي",
        symbols={"n": "عدد الملاحظات (الصفوف)", "p": "عدد الخصائص (الأعمدة)", "x_{ij}": "قيمة الخاصية j للملاحظة i",
                 "y_i": "هدف الملاحظة i"},
        intuition="كل خوارزمية في هذا المقرر تستقبل X بهذا الشكل: أرقام فقط، بلا فئات نصية وبلا قيم مفقودة (غالبًا).",
        example="جدول قروض فيه 2000 طلب و8 أعمدة ⇒ بعد الترميز قد يصبح X بحجم 2000 × 60 (One-hot للمدن).")

dataset_card("mixed")
df = load_dataset("mixed")
X_raw, y = xy("mixed")
c1, c2, c3 = st.columns(3)
c1.metric("n (صفوف)", f"{len(X_raw):,}")
c2.metric("p قبل الترميز", X_raw.shape[1])
num = X_raw.select_dtypes("number").columns.tolist()
cat = [c for c in X_raw.columns if c not in num]
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_raw[cat].astype(str))
c3.metric("p بعد One-hot", len(num) + enc.get_feature_names_out().size)
st.markdown(f"الأعمدة العددية: `{', '.join(num)}` · الفئوية: `{', '.join(cat)}` · الهدف: `approved`.")
why("افصل y عن X قبل أي معالجة، وأخرج المعرّفات (IDs) والأعمدة المستقبلية.",
    "المعرّف يسمح بالحفظ لا التعلّم، والأعمدة المسجلة بعد الحدث تسرّب الهدف.")

st.markdown("## مختبر Feature Space: الملاحظات نقاط في فضاء")
st.caption("اختر خاصيتين عدديتين؛ كل نقطة ملاحظة، ولونها وشكلها الفئة. غيّر القياس لترى كيف تتغير الهندسة.")
c1, c2, c3 = st.columns(3)
fx = c1.selectbox("المحور x", num, index=num.index("credit_score"), key="fm_x")
fy = c2.selectbox("المحور y", num, index=num.index("income"), key="fm_y")
scale = c3.segmented_control("القياس", ["raw", "standardized", "log(income-like)"], default="raw", key="fm_s",
                             required=True)
d = df[[fx, fy, "approved"]].dropna().sample(800, random_state=0)
A = d[[fx, fy]].to_numpy(float)
if scale == "standardized":
    A = StandardScaler().fit_transform(A)
elif scale == "log(income-like)":
    A = np.sign(A) * np.log1p(np.abs(A))
fig = go.Figure()
for cls, color, sym in ((0, PALETTE["sky"], "circle"), (1, PALETTE["coral"], "diamond")):
    m = d["approved"].to_numpy() == cls
    fig.add_trace(go.Scatter(x=A[m, 0], y=A[m, 1], mode="markers", name=f"approved={cls}",
                             marker=dict(color=color, symbol=sym, size=7, opacity=0.65)))
fig.update_layout(title=f"Feature space ({scale})", xaxis_title=fx, yaxis_title=fy, height=420)
if scale == "raw":
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
plot(fig)
if scale == "raw":
    st.warning("بمقياس خام متساوي المحاور، يسيطر المتغير ذو المدى الأكبر على أي مسافة إقليدية — هذا ما «تراه» kNN وK-Means "
               "وSVM دون قياس.", icon=":material/straighten:")
intuition("الخوارزمية لا ترى «دخلًا» و«درجة ائتمان»؛ ترى نقاطًا في فضاء. القياس والترميز والتحويل تغيّر شكل هذا الفضاء، "
          "وبالتالي تغيّر ما يمكن تعلّمه.")

st.markdown("## الأبعاد: n وp ونسبتهما")
ratio = st.slider("n / p", 1, 100, 10, key="fm_ratio")
st.markdown(f"مع n/p = **{ratio}**: " + (
    "خطر شديد: نماذج مرنة ستحفظ الضجيج؛ استخدم تنظيمًا قويًا ونماذج بسيطة." if ratio < 5 else
    "معقول لنماذج خطية منظمة؛ كن حذرًا مع النماذج المرنة جدًا." if ratio < 20 else
    "مريح لمعظم الطرق؛ ما زال التحقق ضروريًا."))

if at_least("advanced"):
    st.markdown("## متقدم: الترميز يغيّر الرتبة والهندسة")
    st.markdown("- One-hot مع حد ثابت ينتج أعمدة مرتبطة خطيًا (Dummy trap): $X$ ليست كاملة الرتبة ⇒ OLS غير فريد؛ "
                "التنظيم أو `drop='first'` يحل ذلك.\n"
                "- فئات عالية الكاردينالية ⇒ p ضخم ومتفرق؛ استخدم `min_frequency` أو TargetEncoder داخل CV.\n"
                "- الخصائص المتطابقة تقريبًا تضخم تباين المعاملات (Multicollinearity) دون أن تضر التنبؤ بالضرورة.")
    Xo = OneHotEncoder(sparse_output=False).fit_transform(df[["region"]])
    Xfull = np.c_[np.ones(len(Xo)), Xo]
    st.code(f"rank([1, one_hot(region)]) = {np.linalg.matrix_rank(Xfull)}  while columns = {Xfull.shape[1]}", language="text")
if at_least("research"):
    researcher_note(["عرّف وحدة التحليل قبل X: هل الصف طلب أم عميل؟ هذا يحدد الاستقلال والتقسيم.",
                     "وثّق خريطة الأعمدة (Data dictionary) وتوقيت توفر كل خاصية في ملحق الورقة."])
dsplat_link("الترميز والقياس")
mistakes(["ترك عمود المعرّف داخل X.", "ترميز الفئات بأرقام 0..k لخوارزمية تعامل الأرقام كمسافات.",
          "حساب مقاييس القياس على كل البيانات قبل التقسيم."])
page_footer("feature_matrix",
            takeaways=["X رقمية n×p وy منفصلة.", "الملاحظات نقاط في فضاء الخصائص؛ الترميز والقياس يغيّران هندسته.",
                       "نسبة n/p تحدد كم يمكن أن يكون النموذج مرنًا."])
