import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, mistakes, researcher_note
from components.formulas import formula
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.optimization import quadratic
from utils.plotting import contour_surface, plot

page_header("calculus")

st.markdown("## المشتقة: الميل المحلي")
formula(r"f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}", title="Derivative",
        symbols={"f'(x)": "ميل المماس عند x", "h": "خطوة صغيرة"},
        intuition="إن كان f'(x) > 0 فالتحرك يمينًا يزيد f، فلنقصانها نتحرك يسارًا: هذا جوهر Gradient descent.",
        example="f(x) = (x − 2)² ⇒ f'(x) = 2(x − 2); عند x = 5 الميل = 6 ⇒ اتجه يسارًا.")

st.markdown("## مختبر المماس")
fn = st.segmented_control("الدالة", ["(x−2)² (convex)", "x⁴ − 3x² + x (non-convex)", "log(1+e^{−x}) (logistic loss)"],
                          default="(x−2)² (convex)", key="calc_fn", required=True)
funcs = {
    "(x−2)² (convex)": (lambda x: (x - 2) ** 2, lambda x: 2 * (x - 2)),
    "x⁴ − 3x² + x (non-convex)": (lambda x: x ** 4 - 3 * x ** 2 + x, lambda x: 4 * x ** 3 - 6 * x + 1),
    "log(1+e^{−x}) (logistic loss)": (lambda x: np.logaddexp(0, -x), lambda x: -1 / (1 + np.exp(x))),
}
f, df = funcs[fn]
x0 = st.slider("النقطة x₀", -3.0, 4.0, 0.5, 0.1, key="calc_x0")
h = st.select_slider("h للفرق المنتهي", [1.0, 0.5, 0.1, 0.01, 0.001], value=0.1, key="calc_h")
xs = np.linspace(-3, 4, 400)
fd = (f(x0 + h) - f(x0)) / h
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=f(xs), name="f(x)", line=dict(color=PALETTE["sky"], width=3)))
fig.add_trace(go.Scatter(x=xs, y=f(x0) + df(x0) * (xs - x0), name="tangent (exact)",
                         line=dict(color=PALETTE["coral"], dash="dash")))
fig.add_trace(go.Scatter(x=xs, y=f(x0) + fd * (xs - x0), name=f"secant (h={h})", line=dict(color=PALETTE["purple"], dash="dot")))
fig.add_trace(go.Scatter(x=[x0], y=[f(x0)], mode="markers", marker=dict(size=12, color="#212529"), showlegend=False))
fig.update_layout(height=380, yaxis=dict(range=[float(np.min(f(xs))) - 1, float(np.percentile(f(xs), 95)) + 1]),
                  title=f"f'(x₀) = {df(x0):.4f} · finite difference = {fd:.4f}")
plot(fig)
intuition("مع h صغير يقترب ميل القاطع من المشتقة. هكذا نتحقق عدديًا من صحة التدرجات (Gradient checking).")

st.markdown("## المشتقات الجزئية والتدرّج")
formula(r"\nabla f(\mathbf w) = \Big(\frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_p}\Big)^\top",
        title="Gradient",
        symbols={r"\partial f/\partial w_j": "ميل f في اتجاه w_j مع تثبيت البقية", r"\nabla f": "اتجاه أسرع صعود"},
        intuition="التدرج عمودي على خطوط الكنتور ويشير إلى الأعلى؛ −∇f يشير إلى أسرع نزول.",
        example="f(w) = ½(w₁² + 10w₂²) ⇒ ∇f = (w₁, 10w₂); عند (2, 1): (2, 10) ⇒ الانحدار في w₂ أشد بكثير.")
s = quadratic(1, 10)
c1, c2 = st.columns(2)
w1 = c1.slider("w₁", -3.5, 3.5, 2.0, 0.25, key="calc_w1")
w2 = c2.slider("w₂", -2.5, 2.5, 1.0, 0.25, key="calc_w2")
g = s.grad(np.array([w1, w2]))
fig = contour_surface(s.f, s.xlim, s.ylim)
scale = 0.15
fig.add_annotation(x=w1 - scale * g[0], y=w2 - scale * g[1], ax=w1, ay=w2, xref="x", yref="y", axref="x", ayref="y",
                   showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor=PALETTE["coral"], text="−∇f")
fig.add_trace(go.Scatter(x=[w1], y=[w2], mode="markers", marker=dict(size=12, color=PALETTE["purple"]), name="w"))
fig.update_layout(title=f"∇f(w) = ({g[0]:.2f}, {g[1]:.2f})")
plot(fig)
page_link("gradient_descent", "استخدم هذا التدرج لبناء Gradient descent", ":material/south_east:")

st.markdown("## قاعدة السلسلة")
formula(r"\frac{d}{dw}\,\ell\big(\sigma(wx)\big) = \ell'(\sigma(wx))\cdot\sigma'(wx)\cdot x",
        title="Chain rule",
        symbols={r"\sigma": "دالة Sigmoid", r"\ell": "الخسارة"},
        intuition="المشتقة عبر سلسلة دوال = حاصل ضرب المشتقات المحلية. Backpropagation هو قاعدة السلسلة مطبقة بذكاء "
                  "من المخرج إلى المدخل.",
        example="لـ Log loss مع Sigmoid تختصر السلسلة إلى (σ(wx) − y)·x — التدرج الأنيق للانحدار اللوجستي.")

st.markdown("## النهايات العظمى والصغرى")
st.markdown("- **شرط أول:** $\\nabla f(\\mathbf w^*) = 0$ (نقطة حرجة).\n"
            "- **شرط ثانٍ:** الهيسيان $H = \\nabla^2 f$ موجب التحديد ⇒ نهاية صغرى محلية؛ سالب ⇒ عظمى؛ مختلط ⇒ نقطة سرج.\n"
            "- **التحدب:** إن كانت $H \\succeq 0$ في كل مكان فكل نهاية صغرى محلية هي عالمية.")

if at_least("advanced"):
    st.markdown("## متقدم: التدرجات التي ستقابلها")
    st.latex(r"\text{MSE: } \nabla_{\beta}\tfrac1n\|y-X\beta\|^2 = -\tfrac2n X^\top(y-X\beta)")
    st.latex(r"\text{Log loss: } \nabla_{w}\tfrac1n\sum_i \ell_i = \tfrac1n X^\top(\sigma(Xw)-y)")
    st.latex(r"\text{Ridge Hessian: } \nabla^2 = \tfrac2n X^\top X + 2\lambda I \succ 0")
if at_least("research"):
    researcher_note(["التفاضل التلقائي (Autodiff) يحسب التدرجات بدقة الآلة؛ الفروق المنتهية للتحقق فقط.",
                     "في DML، شرط التعامد هو «مشتقة Gateaux للعزم بالنسبة للإزعاج = 0» — تعميم لنفس فكرة التدرج."])
    page_link("neyman_orthogonality", "انظر: تعامد Neyman", ":material/architecture:")
mistakes(["خلط اتجاه التدرج (صعود) مع اتجاه الخطوة (نزول).", "استخدام h كبير جدًا أو صغير جدًا في الفروق المنتهية.",
          "افتراض أن ∇f = 0 يعني نهاية صغرى عالمية في دالة غير محدبة."])
page_footer("calculus",
            takeaways=["المشتقة ميل محلي؛ التدرج اتجاه أسرع صعود.", "قاعدة السلسلة أساس Backpropagation.",
                       "التحدب يضمن أن النهاية المحلية عالمية."])
