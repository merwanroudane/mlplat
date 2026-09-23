import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d, toy_regression_1d, true_f_1d, xy
from utils.plotting import lines, plot

page_header("boosting")
algorithm_profile("gradient_boosting")

st.markdown("## الفكرة: متعلمون ضعفاء يصحح بعضهم بعضًا")
formula(r"F_m(x) = F_{m-1}(x) + \nu\, h_m(x),\qquad h_m \approx \arg\min_h \sum_i \ell\big(y_i, F_{m-1}(x_i) + h(x_i)\big)",
        title="Stage-wise additive model",
        symbols={"F_m": "النموذج بعد m مرحلة", "h_m": "متعلم ضعيف (شجرة صغيرة)", r"\nu": "معدل التعلّم (Shrinkage)"},
        intuition="كل شجرة جديدة لا تتعلم y بل «ما تبقى من الخطأ». النموذج النهائي مجموع مئات الأشجار الصغيرة.")
formula(r"r_{im} = -\Big[\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\Big]_{F=F_{m-1}}",
        title="Gradient boosting = gradient descent in function space (Friedman, 2001)",
        intuition="نحسب التدرج السالب للخسارة عند كل نقطة (الـPseudo-residuals) ونلائم شجرة عليه. مع MSE: rᵢ = yᵢ − F(xᵢ) حرفيًا.",
        example="MSE: الشجرة تتعلم البواقي. Log loss: تتعلم yᵢ − pᵢ. Absolute: تتعلم sign(yᵢ − F(xᵢ)).")

st.markdown("## مختبر مراحل التعزيز (Boosting Stage Lab)")
c1, c2, c3 = st.columns(3)
lr = c1.select_slider("learning_rate ν", [0.05, 0.1, 0.3, 0.5, 1.0], value=0.3, key="bo_lr")
depth = c2.slider("عمق كل شجرة", 1, 4, 1, key="bo_depth")
n_stages = c3.slider("عدد المراحل", 5, 60, 30, 5, key="bo_n")
x, y = toy_regression_1d(80, 0.3, seed=2)
grid = np.linspace(0, 1, 300)


@st.cache_data(show_spinner=False, max_entries=32)
def _stages(lr: float, depth: int, n: int):
    F = np.full_like(y, y.mean())
    Fg = np.full_like(grid, y.mean())
    out = [(Fg.copy(), y - F, np.zeros_like(grid))]
    for _ in range(n):
        r = y - F
        h = DecisionTreeRegressor(max_depth=depth).fit(x[:, None], r)
        F = F + lr * h.predict(x[:, None])
        hg = h.predict(grid[:, None])
        Fg = Fg + lr * hg
        out.append((Fg.copy(), y - F, hg))
    return out


stages = _stages(lr, depth, n_stages)


def _frame(i: int) -> None:
    Fg, resid, hg = stages[i]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=PALETTE["muted"], opacity=0.6), name="data"))
        fig.add_trace(go.Scatter(x=grid, y=true_f_1d(grid), name="truth", line=dict(color=PALETTE["teal"], dash="dash")))
        fig.add_trace(go.Scatter(x=grid, y=Fg, name=f"F_{i}", line=dict(color=PALETTE["purple"], width=3)))
        fig.update_layout(title=f"Ensemble after {i} stages · train MSE {np.mean(resid ** 2):.3f}", height=360,
                          yaxis=dict(range=[-2, 2]))
        plot(fig)
    with c2:
        fig = go.Figure(go.Scatter(x=x, y=resid, mode="markers", marker=dict(color=PALETTE["coral"]), name="residuals"))
        if i > 0:
            fig.add_trace(go.Scatter(x=grid, y=hg, name=f"tree h_{i}", line=dict(color=PALETTE["sky"], width=2.5, shape="hv")))
        fig.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dot"))
        fig.update_layout(title="Current residuals and the tree fitted to them", height=360, yaxis=dict(range=[-2, 2]))
        plot(fig)


stepper(f"bo_{lr}_{depth}_{n_stages}", len(stages), _frame, labels=[f"stage {i}" for i in range(len(stages))])
intuition("مع depth = 1 (Stumps) كل شجرة خطوة واحدة؛ الجمع التدريجي يبني منحنى معقدًا. ν صغير = خطوات حذرة تحتاج مراحل أكثر.")

st.markdown("## learning_rate × n_estimators")


@st.cache_data(show_spinner="يدرّب 3 نماذج بـ500 مرحلة…")
def _lr_curves():
    X, y2 = xy("regression")
    Xa, Xb, ya, yb = train_test_split(X, y2, test_size=0.3, random_state=0)
    out = {}
    for v in (1.0, 0.3, 0.05):
        m = GradientBoostingRegressor(n_estimators=500, learning_rate=v, max_depth=3, random_state=0).fit(Xa, ya)
        out[f"ν = {v}"] = [np.mean((yb - p) ** 2) for p in m.staged_predict(Xb)]
    return out


curves = _lr_curves()
fig = lines(np.arange(1, 501), curves, title="Held-out MSE vs stages: small ν needs more trees but reaches lower error",
            xaxis="n_estimators", yaxis="MSE")
fig.update_yaxes(type="log")
plot(fig, height=340)
why("اضبط learning_rate صغيرًا نسبيًا واختر عدد المراحل بالتوقف المبكر.",
    "على عكس Random Forest، المراحل الزائدة في Boosting **تسبب** Overfitting؛ المقايضة بين ν وعدد المراحل شبه عكسية.")

st.markdown("## AdaBoost: إعادة الوزن")
formula(r"\alpha_m = \tfrac12\log\frac{1-\text{err}_m}{\text{err}_m},\qquad w_i \leftarrow w_i\,e^{-\alpha_m y_i h_m(x_i)}",
        title="AdaBoost (discrete, y ∈ {−1, +1})",
        intuition="الأمثلة التي أُخطئ فيها يتضاعف وزنها فتركز عليها الشجرة التالية. يكافئ Gradient boosting بخسارة أسية.")
ada_n = st.slider("عدد Stumps في AdaBoost", 1, 200, 20, key="bo_ada")
Xm, ym = toy_2d("moons", n=300, noise=0.3)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=ada_n, random_state=0).fit(Xm, ym)
boundary_chart(ada, Xm, ym, title=f"AdaBoost with {ada_n} stumps · train accuracy {ada.score(Xm, ym):.3f}")
comparison_table([
    {"": "AdaBoost", "الخسارة": "Exponential", "آلية التصحيح": "أوزان الأمثلة", "حساسية الضجيج": "عالية"},
    {"": "Gradient Boosting", "الخسارة": "أي خسارة قابلة للاشتقاق", "آلية التصحيح": "ملاءمة التدرج السالب", "حساسية الضجيج": "أقل مع Huber/absolute"},
])
hyperparameter_table("GradientBoostingClassifier")
hyperparameter_table("AdaBoostClassifier")
st.caption("منذ scikit-learn 1.9 أصبح معامل criterion في GradientBoosting مهجورًا (friedman_mse وsquared_error كانا متطابقين).")

if at_least("advanced"):
    st.markdown("## متقدم: أدوات التنظيم في التعزيز")
    comparison_table([
        {"الأداة": "Shrinkage (ν)", "الأثر": "خطوات أصغر ⇒ تعميم أفضل"},
        {"الأداة": "subsample < 1 (Stochastic GB)", "الأثر": "عشوائية + سرعة"},
        {"الأداة": "max_depth صغير", "الأثر": "يحد رتبة التفاعلات"},
        {"الأداة": "min_samples_leaf", "الأثر": "أوراق مستقرة"},
        {"الأداة": "Early stopping (n_iter_no_change)", "الأثر": "عدد مراحل تلقائي"},
    ])
if at_least("research"):
    researcher_note(["Friedman (2001) صاغ التعزيز كنزول تدرجي في فضاء الدوال؛ Freund & Schapire (1997) أسسا AdaBoost نظريًا.",
                     "المكتبات الحديثة (XGBoost/LightGBM/CatBoost) تضيف تقريبًا من الرتبة الثانية ومدرجات وتنظيمًا صريحًا."])
    st.markdown(cite("friedman2001", "freund1997", "esl"))
template_checklist({3: "النموذج الجمعي المرحلي", 6: "الخسارة وPseudo-residuals", 7: "Animation المراحل",
                    9: "جداول المعاملات الفائقة", 19: "منحنيات ν × المراحل", 22: "GradientBoosting/AdaBoost", 23: "Boosting Stage Lab"})
mistakes(["اعتبار المراحل الزائدة آمنة كما في RF.", "ν كبير مع أشجار عميقة.", "AdaBoost مع تسميات كثيرة الأخطاء."])
page_footer("boosting",
            takeaways=["التعزيز يجمع أشجارًا صغيرة يصحح كل منها البواقي.", "Gradient boosting = نزول تدرجي في فضاء الدوال.",
                       "ν والمراحل متقايضان؛ استخدم التوقف المبكر."])
