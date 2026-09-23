
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import xy
from utils.optimization import SURFACES, adam, gradient_descent, lasso_coordinate_descent, newton
from utils.plotting import add_path, contour_surface, lines, plot

page_header("advanced_optimizers")

st.markdown("## سباق المحسّنات على نفس السطح")
c1, c2 = st.columns(2)
surf_name = c1.selectbox("السطح", list(SURFACES), format_func=lambda k: SURFACES[k]().name, key="ao_surf")
iters = c2.slider("عدد التكرارات", 5, 200, 40, 5, key="ao_it")
surf = SURFACES[surf_name]()
w0 = np.array([-1.5, 2.0]) if surf_name == "rosenbrock" else np.array([-3.0, 2.0]) if surf_name == "quadratic" else np.array([1.5, 1.5])
lr = {"quadratic": 0.15, "rosenbrock": 0.0015, "double_well": 0.05}[surf_name]
paths = {
    f"GD (η={lr})": gradient_descent(surf.grad, w0, lr, iters),
    "GD + momentum 0.9": gradient_descent(surf.grad, w0, lr * 0.5, iters, momentum=0.9),
    "Adam (η=0.1)": adam(surf.grad, w0, 0.1, iters),
    "Newton (damped 1.0)": newton(surf.grad, surf.hess, w0, min(iters, 30)),
}
trace = [w0]
minimize(lambda w: float(surf.f(np.array(w[0]), np.array(w[1]))), w0, jac=surf.grad, method="L-BFGS-B",
         options={"maxiter": iters}, callback=lambda xk: trace.append(np.array(xk)))
paths["L-BFGS (SciPy)"] = np.array(trace)
fig = contour_surface(surf.f, surf.xlim, surf.ylim)
for i, (name, p) in enumerate(paths.items()):
    add_path(fig, np.clip(p, -10, 10), name=name, color=SEQUENCE[i])
fig.update_xaxes(range=list(surf.xlim))
fig.update_yaxes(range=list(surf.ylim))
fig.update_layout(title=f"Optimizer race on {surf.name}", legend=dict(orientation="h", y=-0.15))
plot(fig, height=500)
rows = []
for name, p in paths.items():
    v = surf.f(p[:, 0], p[:, 1])
    rows.append({"optimizer": name, "steps": len(p) - 1, "final J": float(v[-1]) if np.isfinite(v[-1]) else np.inf,
                 "distance to minimum": float(np.linalg.norm(p[-1] - np.array(surf.minimum)))})
st.dataframe(pd.DataFrame(rows).style.format({"final J": "{:.3g}", "distance to minimum": "{:.4f}"}), hide_index=True,
             width="stretch")
intuition("Newton يستخدم الانحناء (الهيسيان) فيقفز مباشرة إلى حد الدالة التربيعية في خطوة واحدة؛ L-BFGS يقرّب الهيسيان "
          "من تاريخ التدرجات دون حسابه. أما على Double well فقد يصل Newton إلى نقطة حرجة غير صغرى لأن الهيسيان غير موجب هناك.")

st.markdown("## Newton وL-BFGS")
formula(r"\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - H_k^{-1}\nabla J(\boldsymbol\theta_k)", title="Newton step",
        symbols={"H_k": "الهيسيان ∇²J: مصفوفة المشتقات الثانية"},
        intuition="قرّب J محليًا بقطع مكافئ وانتقل إلى قاعه. تكلفة الهيسيان O(p²) ذاكرة وO(p³) حلًا.",
        example="هذا ما يفعله solver='newton-cholesky' في LogisticRegression (ممتاز حين n ≫ p وp معتدل).")
comparison_table([
    {"Solver (LogisticRegression)": "lbfgs (default)", "النوع": "Quasi-Newton", "متى": "افتراضي جيد؛ L2 أو بلا عقوبة"},
    {"Solver (LogisticRegression)": "newton-cholesky", "النوع": "Newton", "متى": "n ≫ p، خصائص One-hot كثيرة معتدلة"},
    {"Solver (LogisticRegression)": "newton-cg", "النوع": "Newton + CG", "متى": "L2، مسائل متوسطة"},
    {"Solver (LogisticRegression)": "liblinear", "النوع": "Coordinate descent", "متى": "بيانات صغيرة، L1 ثنائي"},
    {"Solver (LogisticRegression)": "sag / saga", "النوع": "Stochastic average gradient", "متى": "n كبير؛ saga لـL1/Elastic Net"},
])

st.markdown("## Coordinate descent: لماذا يستخدمه Lasso؟")
formula(r"w_j \leftarrow \frac{S\big(\tfrac1n x_j^\top r^{(j)},\ \alpha\big)}{\tfrac1n\|x_j\|^2},\qquad S(z,t) = \operatorname{sign}(z)\max(|z|-t, 0)",
        title="Lasso coordinate update with soft-thresholding",
        symbols={"r^{(j)}": "البواقي دون مساهمة الخاصية j", "S": "Soft-thresholding: يُصفِّر ما دون العتبة"},
        intuition="عقوبة L1 غير قابلة للاشتقاق عند 0؛ لكن مسألة البعد الواحد لها حل مغلق (Soft-threshold) ⇒ نحدّث خاصية "
                  "واحدة في كل مرة.")
X, y = xy("high_dim")
Xc = (X - X.mean()) / X.std(ddof=0)
Xc = Xc.to_numpy()[:, :20]
yc = (y - y.mean()).to_numpy()
alpha = st.select_slider("α", [0.01, 0.05, 0.1, 0.3, 0.5, 1.0], value=0.3, key="ao_alpha")
w_cd, hist = lasso_coordinate_descent(Xc, yc, alpha, n_sweeps=60)
w_sk = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=100000).fit(Xc, yc).coef_
H = np.array(hist)
fig = go.Figure()
for j in range(Xc.shape[1]):
    fig.add_trace(go.Scatter(x=np.arange(len(H)), y=H[:, j], mode="lines", showlegend=False,
                             line=dict(color=SEQUENCE[j % len(SEQUENCE)], width=1.8)))
fig.update_layout(title=f"Coefficients per sweep (20 features) · max |scratch − sklearn| = {np.abs(w_cd - w_sk).max():.2e}",
                  xaxis_title="sweep", yaxis_title="w_j", height=360)
plot(fig)
st.caption(f"معاملات غير صفرية: {np.count_nonzero(np.abs(w_cd) > 1e-10)} من 20 — Soft-thresholding يُصفّر المعاملات فعلًا.")

st.markdown("## Early stopping كتنظيم")
st.caption("HistGradientBoosting: نراقب خسارة التحقق بعد كل شجرة، ونتوقف حين تتوقف عن التحسن.")


@st.cache_data(show_spinner=False)
def _early_stopping():
    Xr, yr = xy("regression")
    Xa, Xb, ya, yb = train_test_split(Xr, yr, test_size=0.3, random_state=0)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.2, max_leaf_nodes=31, early_stopping=False,
                                      random_state=0).fit(Xa, ya)
    tr = [np.mean((ya - p) ** 2) for p in m.staged_predict(Xa)]
    va = [np.mean((yb - p) ** 2) for p in m.staged_predict(Xb)]
    m2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.2, early_stopping=True, n_iter_no_change=10,
                                       validation_fraction=0.2, random_state=0).fit(Xa, ya)
    return tr, va, m2.n_iter_


tr, va, n_stop = _early_stopping()
fig = lines(np.arange(1, len(tr) + 1), {"training MSE": tr, "validation MSE": va}, title="Boosting iterations: early stopping",
            xaxis="iteration", yaxis="MSE")
fig.add_vline(x=int(np.argmin(va)) + 1, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text="best validation")
fig.add_vline(x=n_stop, line=dict(color=PALETTE["purple"], dash="dot"), annotation_text=f"early_stopping stopped at {n_stop}",
              annotation_position="bottom right")
plot(fig, height=340)
why("استخدم Early stopping على بيانات تحقق داخلية، لا على Test.", "عدد التكرارات معامل فائق؛ اختياره على Test تسرّب.")

if at_least("advanced"):
    st.markdown("## متقدم: Momentum وAdam")
    st.latex(r"v_{k+1} = \beta v_k - \eta\nabla J(\theta_k),\quad \theta_{k+1}=\theta_k+v_{k+1}")
    st.latex(r"m_k=\beta_1 m_{k-1}+(1-\beta_1)g_k,\ s_k=\beta_2 s_{k-1}+(1-\beta_2)g_k^2,\ "
             r"\theta_{k+1}=\theta_k-\eta\,\frac{\hat m_k}{\sqrt{\hat s_k}+\epsilon}")
    st.markdown("Adam يكيّف الخطوة لكل معامل حسب حجم تدرجاته؛ شائع في التعلّم العميق، ونادر في النماذج المحدبة الكلاسيكية "
                "حيث L-BFGS وNewton أفضل.")
if at_least("research"):
    researcher_note(["Friedman, Hastie & Tibshirani (2010): Coordinate descent مع Warm starts على مسار α هو أساس glmnet "
                     "وLasso في scikit-learn؛ منذ 1.8 أضيف Gap safe screening للتسريع.",
                     "Early stopping في GD للانحدار الخطي يعادل تقريبًا Ridge بـλ ∝ 1/(η·k)."])
    st.markdown(cite("liu1989", "friedman2010", "kingma2015"))
mistakes(["استخدام Adam لمسألة محدبة صغيرة بدل L-BFGS.", "إيقاف مبكر على Test.", "نسيان أن Newton يحتاج هيسيانًا موجبًا."])
page_footer("advanced_optimizers",
            takeaways=["Newton/L-BFGS تستخدم الانحناء: خطوات أقل.", "Coordinate descent + Soft-threshold يعطي ندرة Lasso.",
                       "Early stopping تنظيم يُضبط بالتحقق."])
