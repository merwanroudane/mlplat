import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("linear_algebra")

st.markdown("## لماذا الجبر الخطي؟")
comparison_table([
    {"المفهوم": "Dot product", "أين يظهر؟": "كل نموذج خطي: ŷ = wᵀx، النوى في SVM، الانتباه في Transformers"},
    {"المفهوم": "Norms", "أين يظهر؟": "التنظيم L1/L2، معايير التوقف، المسافات"},
    {"المفهوم": "Distances / cosine", "أين يظهر؟": "kNN، K-Means، DBSCAN، استرجاع النصوص"},
    {"المفهوم": "Matrix multiplication", "أين يظهر؟": "Xw للتنبؤ بكل الصفوف دفعة واحدة، طبقات الشبكات"},
    {"المفهوم": "Rank / inverse", "أين يظهر؟": "وجود حل OLS فريد، التعدد الخطي"},
    {"المفهوم": "Pseudo-inverse", "أين يظهر؟": "حل المربعات الصغرى حين XᵀX منفردة"},
    {"المفهوم": "Eigen / SVD", "أين يظهر؟": "PCA، Ridge، تحليل التعدد الخطي، LSA للنصوص"},
])

st.markdown("## المتجهات والجداء النقطي")
formula(r"\mathbf{w}^\top\mathbf{x} = \sum_{j=1}^p w_j x_j = \|\mathbf{w}\|\,\|\mathbf{x}\|\cos\theta",
        title="Dot product",
        symbols={r"\mathbf{w}, \mathbf{x}": "متجهان في ℝᵖ", r"\theta": "الزاوية بينهما", r"\|\cdot\|": "الطول الإقليدي"},
        intuition="يقيس «كم يشير x في اتجاه w». النموذج الخطي يحسب هذا لكل صف: الإشارة تحدد الجانب من المستوى الفاصل.",
        example="w = (2, −1), x = (3, 4) ⇒ wᵀx = 6 − 4 = 2.")

st.markdown("## مختبر: المسافات والتشابه")
c1, c2 = st.columns(2)
a = np.array([c1.slider("a₁", -5.0, 5.0, 3.0, 0.5, key="la_a1"), c1.slider("a₂", -5.0, 5.0, 1.0, 0.5, key="la_a2")])
b = np.array([c2.slider("b₁", -5.0, 5.0, 1.0, 0.5, key="la_b1"), c2.slider("b₂", -5.0, 5.0, 3.0, 0.5, key="la_b2")])
eu = np.linalg.norm(a - b)
man = np.abs(a - b).sum()
cheb = np.abs(a - b).max()
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
fig = go.Figure()
for vec, name, color in ((a, "a", PALETTE["sky"]), (b, "b", PALETTE["coral"])):
    fig.add_trace(go.Scatter(x=[0, vec[0]], y=[0, vec[1]], mode="lines+markers+text", name=name, text=["", name],
                             textposition="top center", line=dict(color=color, width=3), marker=dict(size=[0, 12])))
fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", name="Euclidean", line=dict(dash="dot", color=PALETTE["purple"])))
fig.add_trace(go.Scatter(x=[a[0], b[0], b[0]], y=[a[1], a[1], b[1]], mode="lines", name="Manhattan path",
                         line=dict(dash="dash", color=PALETTE["teal"])))
fig.update_layout(height=420, xaxis=dict(range=[-6, 6], zeroline=True), yaxis=dict(range=[-6, 6], scaleanchor="x"),
                  title="Vectors, distances and angle")
c1, c2 = st.columns([1.6, 1])
with c1:
    plot(fig)
with c2:
    st.metric("Euclidean ‖a−b‖₂", f"{eu:.3f}")
    st.metric("Manhattan ‖a−b‖₁", f"{man:.3f}")
    st.metric("Chebyshev ‖a−b‖∞", f"{cheb:.3f}")
    st.metric("Cosine similarity", f"{cos:.3f}")
    st.metric("a · b", f"{a @ b:.2f}")
intuition("Cosine يتجاهل الطول ويقيس الاتجاه فقط: وثيقتان بنفس توزيع الكلمات وبأطوال مختلفة «متشابهتان» جيبيًا، "
          "بعيدتان إقليديًا. لذلك يُفضَّل Cosine في النصوص.")

st.markdown("## المصفوفة كتحويل")
st.caption("كل مصفوفة 2×2 تنقل الدائرة الواحدية إلى قطع ناقص؛ المحاور الرئيسية هي المتجهات المفردة، وأطوالها القيم المفردة.")
c1, c2, c3, c4 = st.columns(4)
M = np.array([[c1.number_input("m₁₁", value=2.0, step=0.5, key="la_m11"), c2.number_input("m₁₂", value=1.0, step=0.5, key="la_m12")],
              [c3.number_input("m₂₁", value=0.5, step=0.5, key="la_m21"), c4.number_input("m₂₂", value=1.0, step=0.5, key="la_m22")]])
t = np.linspace(0, 2 * np.pi, 200)
circle = np.c_[np.cos(t), np.sin(t)]
ell = circle @ M.T
U, S, Vt = np.linalg.svd(M)
fig = go.Figure()
fig.add_trace(go.Scatter(x=circle[:, 0], y=circle[:, 1], mode="lines", name="unit circle", line=dict(color=PALETTE["muted"], dash="dot")))
fig.add_trace(go.Scatter(x=ell[:, 0], y=ell[:, 1], mode="lines", name="M · circle", line=dict(color=PALETTE["purple"], width=3)))
for k, color in ((0, PALETTE["coral"]), (1, PALETTE["teal"])):
    v = U[:, k] * S[k]
    fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers", name=f"σ{k + 1}·u{k + 1}",
                             line=dict(color=color, width=3)))
lim = max(3, float(np.abs(ell).max()) + 0.5)
fig.update_layout(height=420, xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim], scaleanchor="x"),
                  title="A matrix maps the unit circle to an ellipse (SVD axes)")
c1, c2 = st.columns([1.6, 1])
with c1:
    plot(fig)
with c2:
    det = np.linalg.det(M)
    st.metric("det(M)", f"{det:.3f}")
    st.metric("rank(M)", int(np.linalg.matrix_rank(M)))
    st.metric("σ₁, σ₂", f"{S[0]:.2f}, {S[1]:.2f}")
    st.metric("condition number σ₁/σ₂", f"{S[0] / max(S[1], 1e-12):.1f}" if S[1] > 1e-12 else "∞")
    if abs(det) < 1e-9:
        st.warning("المصفوفة منفردة: تسحق البعد إلى خط؛ لا معكوس لها — هذا ما يحدث لـXᵀX مع خصائص مرتبطة تمامًا.")

st.markdown("## المعكوس والمعكوس الزائف والمربعات الصغرى")
formula(r"\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top y \quad\text{(if } X^\top X \text{ invertible)},\qquad "
        r"\hat{\boldsymbol\beta} = X^{+}y = V\Sigma^{+}U^\top y",
        title="Normal equations and pseudo-inverse",
        symbols={"X^+": "المعكوس الزائف (Moore–Penrose)", r"\Sigma^+": "مقلوب القيم المفردة غير الصفرية فقط"},
        intuition="حين تكون الخصائص مرتبطة تمامًا لا يوجد حل فريد؛ المعكوس الزائف يختار الحل ذا أصغر معيار ‖β‖₂.",
        example=lambda: st.code(
            f"X = [[1,2],[2,4],[3,6]] (column 2 = 2 × column 1)\nrank = {np.linalg.matrix_rank(np.array([[1, 2], [2, 4], [3, 6]]))}"
            f"\npinv(X) @ [1,2,3] = {np.round(np.linalg.pinv(np.array([[1., 2], [2, 4], [3, 6]])) @ np.array([1., 2, 3]), 3)}",
            language="text"))

st.markdown("## القيم الذاتية وSVD")
formula(r"A\mathbf{v} = \lambda\mathbf{v},\qquad X = U\Sigma V^\top",
        title="Eigen-decomposition and SVD",
        symbols={r"\lambda, \mathbf v": "قيمة ومتجه ذاتي: اتجاه لا يدور، يتمدد بمعامل λ",
                 "U, V": "مصفوفتان متعامدتان (دوران)", r"\Sigma": "قيم مفردة σ₁ ≥ σ₂ ≥ … ≥ 0 (تمدد)"},
        intuition="كل مصفوفة = دوران ← تمدد على المحاور ← دوران. PCA هو SVD لمصفوفة البيانات المركزية: V اتجاهات "
                  "التباين، وσ²/(n−1) التباين على كل اتجاه.",
        example="لمصفوفة التغاير Σ = [[2,1],[1,2]]: λ = 3 على (1,1)/√2 و λ = 1 على (1,−1)/√2.")

if at_least("advanced"):
    st.markdown("## متقدم: التكييف العددي ولماذا لا نحسب (XᵀX)⁻¹")
    st.markdown("رقم التكييف $\\kappa(X^\\top X) = \\kappa(X)^2$. إن كان $\\kappa(X) = 10^6$ فإن حساب المعكوس صراحة يخسر ~12 "
                "رقمًا عشريًا. لذلك تستخدم المكتبات QR أو SVD (`np.linalg.lstsq`) أو Cholesky لمسائل مُكيَّفة جيدًا.")
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    rows = []
    for eps in (1.0, 1e-2, 1e-4, 1e-6):
        X = np.c_[x1, x1 + eps * rng.normal(size=200)]
        rows.append({"noise between x1, x2": eps, "κ(X)": np.linalg.cond(X), "κ(XᵀX)": np.linalg.cond(X.T @ X)})
    st.dataframe(pd.DataFrame(rows).style.format({"noise between x1, x2": "{:g}", "κ(X)": "{:.2e}", "κ(XᵀX)": "{:.2e}"}),
                 hide_index=True)
if at_least("research"):
    researcher_note(["Ridge = إضافة λ لكل قيمة ذاتية لـXᵀX: (XᵀX + λI) تُحسّن التكييف وتنكمش الاتجاهات ذات التباين "
                     "المنخفض أكثر (ESL §3.4.1).",
                     "في الأبعاد العالية (p > n) تكون XᵀX منفردة دائمًا ⇒ الحاجة إلى التنظيم ليست خيارًا بل ضرورة للتعريف."])
mistakes(["استخدام المسافة الإقليدية على خصائص غير مقاسة.", "حساب المعكوس صراحة بدل lstsq/solve.",
          "افتراض أن Cosine وEuclidean متكافئتان."])
page_footer("linear_algebra",
            takeaways=["الجداء النقطي قلب النماذج الخطية.", "المسافة المختارة تحدد معنى «القرب».",
                       "SVD = دوران-تمدد-دوران، وأساس PCA وRidge والمعكوس الزائف."])
