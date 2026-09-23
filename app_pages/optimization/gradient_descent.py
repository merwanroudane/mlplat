import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import SGDRegressor

from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.code_lab import code_lab
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.optimization import (SURFACES, gradient_descent, learning_rate_limit, make_linear_data, mse_loss_surface,
                                ols_closed_form, sgd_linear)
from utils.plotting import add_path, contour_surface, lines, plot

page_header("gradient_descent")

formula(r"\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \eta\,\nabla J(\boldsymbol\theta_k)",
        title="Gradient descent update",
        symbols={r"\eta": "معدل التعلّم (Learning rate)", r"\nabla J": "التدرج: اتجاه أسرع صعود"},
        intuition="قف على جبل في الضباب: تحسس الميل تحت قدميك وخذ خطوة نحو الأسفل؛ طول الخطوة η.",
        example="J(θ) = θ²، θ₀ = 4، η = 0.1 ⇒ θ₁ = 4 − 0.1·8 = 3.2، θ₂ = 2.56 … يتناقص هندسيًا بمعامل 0.8.")

st.markdown("## مختبر Gradient Descent على أسطح اختبار")
c1, c2, c3 = st.columns(3)
surf_name = c1.selectbox("السطح", list(SURFACES), format_func=lambda k: SURFACES[k]().name, key="gd_surf")
surf = SURFACES[surf_name]()
lr = c2.select_slider("معدل التعلّم η", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.18, 0.2, 0.25, 0.5, 1.0],
                      value=0.1 if surf_name != "rosenbrock" else 0.002, key="gd_lr")
iters = c3.slider("عدد التكرارات", 5, 300, 60, 5, key="gd_iters")
c4, c5, c6, c7 = st.columns(4)
x0 = c4.slider("بداية w₁", float(surf.xlim[0]), float(surf.xlim[1]), -3.0 if surf_name == "quadratic" else -1.5, 0.1, key="gd_x0")
y0 = c5.slider("بداية w₂", float(surf.ylim[0]), float(surf.ylim[1]), 2.0 if surf_name != "rosenbrock" else 2.0, 0.1, key="gd_y0")
noise = c6.slider("ضجيج التدرج (SGD-like)", 0.0, 5.0, 0.0, 0.25, key="gd_noise")
momentum = c7.slider("Momentum β", 0.0, 0.95, 0.0, 0.05, key="gd_mom")
path = gradient_descent(surf.grad, np.array([x0, y0]), lr, iters, momentum=momentum, noise=noise, seed=0)
vals = surf.f(path[:, 0], path[:, 1])
diverged = (not np.all(np.isfinite(path))) or np.abs(path).max() > 1e3 or vals[-1] > vals[0] * 10


def _frame(i: int) -> None:
    k = int(round((i + 1) / 20 * (len(path) - 1)))
    sub = path[: k + 1]
    fig = contour_surface(surf.f, surf.xlim, surf.ylim)
    add_path(fig, np.clip(sub, -50, 50), name=f"GD path (η={lr})")
    fig.add_trace(go.Scatter(x=[surf.minimum[0]], y=[surf.minimum[1]], mode="markers", name="minimum",
                             marker=dict(symbol="x", size=14, color=PALETTE["teal"])))
    fig.update_xaxes(range=list(surf.xlim))
    fig.update_yaxes(range=list(surf.ylim))
    fig.update_layout(title=f"{surf.name} — iteration {k} / {len(path) - 1}")
    left, right = st.columns([1.5, 1])
    with left:
        plot(fig)
    with right:
        lf = lines(np.arange(k + 1), {"J(θ_k)": np.clip(vals[: k + 1], 1e-12, 1e8)}, title="Loss curve", xaxis="iteration",
                   yaxis="J")
        lf.update_yaxes(type="log")
        plot(lf, height=300)
        st.metric("J الحالي", f"{vals[k]:.4g}" if np.isfinite(vals[k]) else "∞")
        st.metric("المسافة إلى الحل", f"{np.linalg.norm(path[k] - np.array(surf.minimum)):.4f}"
                  if np.all(np.isfinite(path[k])) else "∞")


stepper(f"gd_{surf_name}_{lr}_{iters}_{x0}_{y0}_{noise}_{momentum}", 20, _frame, labels=[f"{5 * (i + 1)}%" for i in range(20)])
if surf_name == "quadratic":
    lim = learning_rate_limit(surf.hess(np.zeros(2)))
    st.info(f"للدالة التربيعية: GD يتقارب إذا 0 < η < 2/λ_max = **{lim:.3f}**، ويكون أسرع قرب η = 2/(λ_min + λ_max) "
            f"= {2 / (1 + 10):.3f}. جرّب η = 0.18 ثم 0.2 ثم 0.25.", icon=":material/calculate:")
if diverged:
    st.error("تباعد (Divergence): الخطوة أكبر من انحناء السطح فتقفز فوق الوادي وتبتعد. صغّر η.", icon=":material/trending_up:")
elif vals[-1] > vals[0] * 0.5 and noise == 0:
    st.warning("تقدم بطيء جدًا: η صغير أو السطح سيئ التكييف. زد η بحذر أو أضف Momentum.", icon=":material/hourglass_bottom:")
intuition("على السطح المستطيل (κ = 10) يتذبذب GD عبر الوادي الضيق ويتقدم ببطء على طوله. Momentum يراكم السرعة في الاتجاه "
          "الثابت ويخمد التذبذب. وعلى Double well تحدد نقطة البداية أي حد أدنى تصل إليه.")

st.markdown("## Batch مقابل SGD مقابل Mini-batch على بيانات حقيقية")
formula(r"\nabla J(\theta) \approx \frac{1}{|B|}\sum_{i\in B}\nabla \ell_i(\theta)",
        title="Stochastic gradient from a mini-batch B",
        intuition="بدل حساب التدرج على كل n ملاحظة، نقدّره من دفعة صغيرة: أرخص بكثير لكنه متقلب (غير متحيز).")
c1, c2, c3, c4 = st.columns(4)
batch = c1.segmented_control("نمط الدفعات", ["Batch (n)", "Mini-batch (32)", "SGD (1)"], default="Mini-batch (32)",
                             key="gd_batch", required=True)
lr2 = c2.select_slider("η", [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05, key="gd_lr2")
epochs = c3.slider("Epochs", 1, 30, 5, key="gd_ep")
decay = c4.slider("تناقص η (decay)", 0.0, 0.05, 0.0, 0.005, key="gd_decay")
Xd, yd = make_linear_data(256, noise=0.8, seed=3)
bs = {"Batch (n)": None, "Mini-batch (32)": 32, "SGD (1)": 1}[batch]
p_sgd, l_sgd = sgd_linear(Xd, yd, np.array([-3.0, 3.0]), lr2, epochs, batch_size=bs, seed=0, decay=decay)
beta = ols_closed_form(Xd, yd)
fs = mse_loss_surface(Xd, yd)
c1, c2 = st.columns([1.3, 1])
with c1:
    fig = contour_surface(fs, (beta[0] - 5, beta[0] + 5), (beta[1] - 5, beta[1] + 5))
    add_path(fig, np.clip(p_sgd, -20, 20), name=batch, color=PALETTE["purple"])
    fig.add_trace(go.Scatter(x=[beta[0]], y=[beta[1]], mode="markers", name="OLS optimum",
                             marker=dict(symbol="star", size=15, color=PALETTE["coral"])))
    fig.update_layout(title=f"{batch}: {len(p_sgd) - 1} parameter updates in {epochs} epochs")
    plot(fig)
with c2:
    lf = lines(np.arange(len(l_sgd)), {"full-data MSE": np.clip(l_sgd, 1e-6, 1e6)}, title="Loss per update",
               xaxis="update", yaxis="MSE")
    lf.update_yaxes(type="log")
    lf.add_hline(y=float(np.mean((yd - Xd @ beta) ** 2)), line=dict(dash="dot", color=PALETTE["teal"]),
                 annotation_text="optimum")
    plot(lf, height=330)
    st.metric("المعاملات النهائية", f"({p_sgd[-1, 0]:.3f}, {p_sgd[-1, 1]:.3f})")
    st.caption(f"OLS: ({beta[0]:.3f}, {beta[1]:.3f})")
if st.button("سجّل هذا التشغيل", key="gd_log", icon=":material/history:", type="tertiary"):
    log_experiment("Gradient Descent Lab", f"linear regression by {batch}", {"lr": lr2, "epochs": epochs, "decay": decay},
                   {"final_mse": float(l_sgd[-1])}, seed=0, dataset="make_linear_data(n=256, noise=0.8, seed=3)")
    st.toast("سُجّل.", icon=":material/check:")
why("استخدم Mini-batch مع تناقص معدل التعلّم للبيانات الكبيرة.",
    "SGD بمعدل ثابت يبقى «يرتجف» حول الحل بتباين يتناسب مع η؛ تناقص η (شروط Robbins–Monro) يسمح بالتقارب.")

st.markdown("## التنفيذ من الصفر مقابل المكتبة")


def _code(p):
    return ("# from scratch: batch gradient descent for linear regression\n"
            "w = np.zeros(2)\n"
            f"for _ in range({p['n_iter']}):\n"
            "    grad = -2 * X.T @ (y - X @ w) / len(y)\n"
            f"    w -= {p['lr']} * grad\n\n"
            "# library: SGDRegressor (squared loss, no penalty)\n"
            f"SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0={p['lr']},\n"
            "             max_iter=50, tol=None, random_state=0).fit(x, y)")


def _run(lr, n_iter):
    w = np.zeros(2)
    for _ in range(n_iter):
        w -= lr * (-2 * Xd.T @ (yd - Xd @ w) / len(yd))
    sk = SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant", eta0=lr, max_iter=50, tol=None,
                      random_state=0).fit(Xd[:, :1], yd)
    return (f"| method | w₁ (slope) | w₂ (intercept) |\n|---|---|---|\n| scratch GD | {w[0]:.4f} | {w[1]:.4f} |\n"
            f"| SGDRegressor | {sk.coef_[0]:.4f} | {sk.intercept_[0]:.4f} |\n| closed form (OLS) | {beta[0]:.4f} | {beta[1]:.4f} |")


code_lab("gd_code", "Scratch GD vs SGDRegressor vs closed form", _code, _run,
         lambda: {"lr": st.select_slider("η", [0.01, 0.05, 0.1, 0.2], value=0.1, key="gdc_lr"),
                  "n_iter": st.slider("iterations", 10, 500, 200, 10, key="gdc_it")},
         explanation="الثلاثة يقتربون من الحل نفسه لأن المسألة محدبة؛ SGDRegressor يعالج صفًا واحدًا في كل تحديث. "
                     "ملاحظة: منذ scikit-learn 1.8 يجب أن يكون eta0 موجبًا تمامًا.")

if at_least("advanced"):
    st.markdown("## متقدم: لماذا يتذبذب GD على الوادي الضيق؟")
    st.markdown("على $J = \\tfrac12 \\sum_j \\lambda_j \\theta_j^2$ يتطور كل إحداثي مستقلًا: $\\theta_j^{(k)} = (1-\\eta\\lambda_j)^k\\theta_j^{(0)}$. "
                "الاتجاه ذو $\\lambda$ الكبير يفرض $\\eta < 2/\\lambda_{max}$ للاستقرار، فيتقدم الاتجاه ذو $\\lambda$ الصغير "
                "بمعامل $(1-2\\lambda_{min}/\\lambda_{max})$ فقط ⇒ عدد التكرارات ∝ رقم التكييف κ. لذلك **القياس (Standardization)** "
                "يسرّع GD كثيرًا: يقرّب κ من 1.")
if at_least("research"):
    researcher_note(["Robbins & Monro (1951): شروط Σηₖ = ∞ وΣηₖ² < ∞ تضمن تقارب SGD.",
                     "SGD كتنظيم ضمني: الضجيج يفضّل حلولًا «مسطحة» في الشبكات العميقة (موضوع بحث نشط).",
                     "في المسائل المحدبة القوية، يتقارب SGD بمعدل O(1/k) مقابل الخطي لـGD الكامل، لكن بتكلفة O(1) لكل خطوة."])
    st.markdown(cite("robbins1951", "kingma2015"))
mistakes(["η كبير ⇒ تباعد؛ صغير ⇒ لا تقارب في الزمن المتاح.", "نسيان القياس قبل GD.",
          "مقارنة SGD وGD بعدد التكرارات بدل عدد الـEpochs أو الزمن."])
page_footer("gradient_descent",
            takeaways=["GD: خطوة بحجم η عكس التدرج.", "η > 2/λ_max ⇒ تباعد؛ الأسطح سيئة التكييف بطيئة.",
                       "SGD/Mini-batch أرخص لكل خطوة ومتقلب؛ Momentum والتناقص يساعدان."])
