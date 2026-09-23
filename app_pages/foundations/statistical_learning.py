import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from components.callouts import definition, intuition, mistakes, researcher_note
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_regression_1d, true_f_1d
from utils.plotting import lines, plot, regression_fit

page_header("statistical_learning")

st.markdown("## الإطار")
formula(r"Y = f(X) + \varepsilon,\qquad \mathbb E[\varepsilon\mid X]=0,\ \operatorname{Var}(\varepsilon)=\sigma^2",
        title="The statistical learning model (ISLP §2.1)",
        symbols={"f": "الدالة المنتظمة المجهولة", r"\varepsilon": "ضجيج لا يمكن التنبؤ به من X"},
        intuition="نقدّر f بـf̂ من عينة؛ لا نستطيع أبدًا التنبؤ بـε.")
formula(r"\mathbb E\big[(Y-\hat f(X))^2\big] = \underbrace{\mathbb E\big[(f(X)-\hat f(X))^2\big]}_{\text{reducible}} + "
        r"\underbrace{\sigma^2}_{\text{irreducible}}",
        title="Reducible vs irreducible error",
        intuition="أفضل نموذج ممكن لا ينزل تحت σ². إن كانت درجة التدريب أقل من σ² فأنت تحفظ الضجيج.",
        example="في بيانات الانحدار الاصطناعية σ² = 1 ⇒ لا يمكن أن يكون MSE الاختبار المتوقع أقل من 1.")

st.markdown("## المخاطرة المتوقعة والتجريبية")
definition("Empirical risk minimization", "$\\hat f = \\arg\\min_{f\\in\\mathcal F} \\hat R_n(f)$ حيث "
           "$\\hat R_n(f) = \\frac1n\\sum_i \\ell(y_i, f(x_i))$. التعميم الجيد يعني أن $\\hat R_n(\\hat f) \\approx R(\\hat f)$.")
intuition("خطأ التدريب يقدّر $\\hat R_n$ لنموذج اختير لأنه يقللها ⇒ متفائل دائمًا. خطأ التحقق يقدّر $R$ لنموذج لم يرَ "
          "تلك البيانات.")

st.markdown("## مختبر: السعة والتعميم")
c1, c2, c3 = st.columns(3)
n = c1.slider("n", 15, 300, 40, 5, key="sl_n")
noise = c2.slider("σ (ضجيج)", 0.05, 1.0, 0.3, 0.05, key="sl_noise")
deg = c3.slider("درجة كثير الحدود (السعة)", 1, 20, 3, key="sl_deg")
x, y = toy_regression_1d(n, noise, seed=1)
xt, yt = toy_regression_1d(400, noise, seed=99)
grid = np.linspace(0, 1, 300)
model = make_pipeline(PolynomialFeatures(deg), Ridge(alpha=1e-8)).fit(x[:, None], y)
tr = np.mean((y - model.predict(x[:, None])) ** 2)
te = np.mean((yt - model.predict(xt[:, None])) ** 2)
c1, c2 = st.columns([1.7, 1])
with c1:
    fig = regression_fit(x, y, grid, {f"degree {deg}": model.predict(grid[:, None])}, truth=true_f_1d(grid),
                         title="Fit vs truth f(x) = sin(2πx)")
    fig.update_yaxes(range=[-2.5, 2.5])
    plot(fig, height=360)
with c2:
    st.metric("Training MSE", f"{tr:.3f}")
    st.metric("Test MSE", f"{te:.3f}", f"{te - tr:+.3f} gap", delta_color="off")
    st.metric("Irreducible σ²", f"{noise ** 2:.3f}")
    if tr < noise ** 2 * 0.6:
        st.warning("خطأ التدريب أقل بكثير من σ²: النموذج يحفظ الضجيج.", icon=":material/warning:")


@st.cache_data(show_spinner=False)
def _vc(n: int, noise: float):
    x, y = toy_regression_1d(n, noise, seed=1)
    xt, yt = toy_regression_1d(400, noise, seed=99)
    degs = np.arange(1, 16)
    tr, te = [], []
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-8)).fit(x[:, None], y)
        tr.append(np.mean((y - m.predict(x[:, None])) ** 2))
        te.append(np.mean((yt - m.predict(xt[:, None])) ** 2))
    return degs, tr, te


degs, trs, tes = _vc(n, noise)
fig = lines(degs, {"training MSE": trs, "test MSE": tes}, title="The U-shape: error vs capacity", xaxis="degree",
            yaxis="MSE", markers=True)
fig.add_hline(y=noise ** 2, line=dict(dash="dot", color=PALETTE["muted"]), annotation_text="σ² (irreducible)")
fig.update_yaxes(type="log")
plot(fig, height=340)

st.markdown("## Learning curves وValidation curves في scikit-learn")
kind = st.segmented_control("المنحنى", ["Learning curve (n)", "Validation curve (max_depth)"],
                            default="Learning curve (n)", key="sl_curve", required=True)
X_big, y_big = toy_regression_1d(600, 0.3, seed=5)


@st.cache_data(show_spinner="يحسب المنحنى…")
def _curves(kind: str):
    X = X_big[:, None]
    if kind.startswith("Learning"):
        out = {}
        for depth in (2, 12):
            sizes, tr, va = learning_curve(DecisionTreeRegressor(max_depth=depth, random_state=0), X, y_big, cv=5,
                                           train_sizes=np.linspace(0.1, 1, 8), scoring="neg_mean_squared_error",
                                           shuffle=True, random_state=0)
            out[depth] = (sizes, -tr.mean(1), -va.mean(1))
        return out
    depths = np.arange(1, 16)
    tr, va = validation_curve(DecisionTreeRegressor(random_state=0), X, y_big, param_name="max_depth", param_range=depths,
                              cv=5, scoring="neg_mean_squared_error")
    return depths, -tr.mean(1), -va.mean(1)


res = _curves(kind)
if kind.startswith("Learning"):
    fig = go.Figure()
    for depth, color in ((2, PALETTE["sky"]), (12, PALETTE["coral"])):
        sizes, tr, va = res[depth]
        fig.add_trace(go.Scatter(x=sizes, y=tr, name=f"train (depth {depth})", line=dict(color=color, dash="dot")))
        fig.add_trace(go.Scatter(x=sizes, y=va, name=f"validation (depth {depth})", line=dict(color=color, width=3)))
    fig.update_layout(title="Learning curves: high bias (depth 2) vs high variance (depth 12)", xaxis_title="training size",
                      yaxis_title="MSE", height=380)
    plot(fig)
    st.markdown("- **تحيز عالٍ (depth 2):** المنحنيان يلتقيان عند خطأ مرتفع ⇒ المزيد من البيانات لن يساعد؛ زد السعة.\n"
                "- **تباين عالٍ (depth 12):** فجوة كبيرة تضيق ببطء ⇒ المزيد من البيانات أو تنظيم أقوى يساعدان.")
else:
    depths, tr, va = res
    plot(lines(depths, {"training": tr, "validation (5-fold)": va}, title="Validation curve for max_depth",
               xaxis="max_depth", yaxis="MSE", markers=True), height=360)
    st.markdown(f"أفضل max_depth على التحقق ≈ **{depths[int(np.argmin(va))]}**. يسار ذلك Underfitting، ويمينه Overfitting.")

st.markdown("## التنظيم = التحكم في السعة")
intuition("بدل اختيار درجة كثير الحدود، يمكن السماح بدرجة عالية مع عقوبة على حجم المعاملات. التنظيم يحرك النموذج على "
          "منحنى U باستمرار عبر معامل واحد (λ).")
page_link("bias_variance", "التالي: مقايضة التحيز والتباين بالمحاكاة", ":material/balance:")

if at_least("advanced"):
    st.markdown("## متقدم: التفاؤل (Optimism) في خطأ التدريب")
    st.latex(r"\mathbb E[\text{Err}_{\text{in}}] - \mathbb E[\overline{\text{err}}] = \frac{2}{n}\sum_{i=1}^n \operatorname{Cov}(\hat y_i, y_i)")
    st.markdown("كلما اعتمدت التنبؤات أكثر على y الملاحَظة نفسها (نموذج مرن)، زاد التفاؤل. للانحدار الخطي بـp خاصية: "
                "$2p\\sigma^2/n$ — أساس Cp وAIC (ESL §7.4).")
if at_least("research"):
    researcher_note(["Double descent: في النماذج شديدة الإفراط في المعاملات قد ينخفض خطأ الاختبار مجددًا بعد نقطة الاستيفاء؛ "
                     "لا يلغي منحنى U في الإعدادات الكلاسيكية لكنه يقيّد تعميمه.",
                     "أبلغ عن learning curves في الأوراق التطبيقية: تبيّن هل البيانات الإضافية ستفيد."])
    st.markdown(cite("islp", "esl"))
mistakes(["تفسير انخفاض خطأ التدريب كتحسّن.", "جمع مزيد من البيانات لنموذج متحيز جدًا.",
          "اختيار السعة على مجموعة الاختبار."])
page_footer("statistical_learning",
            takeaways=["الخطأ = قابل للاختزال + غير قابل.", "خطأ التدريب متفائل؛ التحقق يقدّر التعميم.",
                       "Learning/validation curves تشخّص التحيز مقابل التباين."])
