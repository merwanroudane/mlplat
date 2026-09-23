import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import toy_regression_1d, true_f_1d
from utils.plotting import lines, plot, regression_fit

page_header("polynomial_regression")

formula(r"\hat y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d = \phi(x)^\top\beta",
        title="Basis expansion",
        symbols={r"\phi(x)": "متجه الأساسات (1, x, x², …, x^d)", "d": "الدرجة — المعامل الفائق الذي يحدد السعة"},
        intuition="النموذج ما زال خطيًا في المعاملات β (فنستخدم OLS نفسه)، لكنه غير خطي في x. نغيّر الخصائص لا الخوارزمية.",
        example="d = 2 مع خاصيتين: PolynomialFeatures يولّد 1, x₁, x₂, x₁², x₁x₂, x₂² (يشمل التفاعلات).")

st.markdown("## مختبر الدرجة: كثير حدود مقابل Splines")
c1, c2, c3, c4 = st.columns(4)
n = c1.slider("n", 15, 300, 50, 5, key="pr_n")
noise = c2.slider("σ", 0.05, 1.0, 0.3, 0.05, key="pr_noise")
deg = c3.slider("درجة كثير الحدود", 1, 20, 9, key="pr_deg")
knots = c4.slider("عقد Spline", 3, 20, 6, key="pr_knots")
x, y = toy_regression_1d(n, noise, seed=3)
grid = np.linspace(0, 1, 300)
poly = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x[:, None], y)
spl = make_pipeline(SplineTransformer(n_knots=knots, degree=3), LinearRegression()).fit(x[:, None], y)
fig = regression_fit(x, y, grid, {f"polynomial d={deg}": poly.predict(grid[:, None]),
                                  f"cubic spline ({knots} knots)": spl.predict(grid[:, None])},
                     truth=true_f_1d(grid), title="Global polynomial vs local splines")
fig.update_yaxes(range=[-2.5, 2.5])
plot(fig, height=420)
intuition("كثير الحدود **عالمي**: كل معامل يؤثر في كل مكان، فيتأرجح بعنف عند الأطراف (ظاهرة Runge). Splines **محلية**: "
          "كل قطعة تتأثر بالبيانات القريبة فقط، فهي أكثر استقرارًا.")


@st.cache_data(show_spinner=False)
def _cv_curve(n: int, noise: float):
    x, y = toy_regression_1d(n, noise, seed=3)
    degs = np.arange(1, 16)
    scores = [-cross_val_score(make_pipeline(PolynomialFeatures(d), LinearRegression()), x[:, None], y, cv=5,
                               scoring="neg_mean_squared_error").mean() for d in degs]
    return degs, scores


degs, cv = _cv_curve(n, noise)
fig = lines(degs, {"5-fold CV MSE": cv}, title="Choose the degree by cross-validation", xaxis="degree", yaxis="MSE",
            markers=True)
fig.update_yaxes(type="log")
plot(fig, height=320)
st.markdown(f"أفضل درجة بالتحقق المتقاطع ≈ **{degs[int(np.argmin(cv))]}**.")
why("ضع PolynomialFeatures داخل Pipeline مع Scaler وتنظيم.",
    "x^15 على [0, 10] يصل إلى 10¹⁵: تكييف عددي كارثي. القياس ثم Ridge يجعل الدرجات العالية آمنة نسبيًا.")
st.code("""model = make_pipeline(PolynomialFeatures(degree=5, include_bias=False),
                      StandardScaler(),
                      RidgeCV(alphas=np.logspace(-4, 3, 30)))""", language="python")

if at_least("advanced"):
    st.markdown("## متقدم: انفجار عدد الخصائص")
    p = st.slider("عدد الخصائص الأصلية p", 2, 50, 10, key="pr_p")
    from math import comb
    counts = {d: comb(p + d, d) for d in range(1, 6)}
    st.dataframe({"degree": list(counts), "number of features (with bias)": list(counts.values())}, hide_index=True)
    st.markdown("عدد الحدود = C(p + d, d): ينمو بسرعة فائقة ⇒ النوى (Kernel trick) في SVM تتجنب بناءها صراحة.")
    Xm = np.random.default_rng(0).normal(size=(200, 3))
    ym = Xm[:, 0] * Xm[:, 1] + Xm[:, 2] ** 2 + np.random.default_rng(1).normal(scale=0.3, size=200)
    s1 = cross_val_score(LinearRegression(), Xm, ym, cv=5).mean()
    s2 = cross_val_score(make_pipeline(PolynomialFeatures(2), StandardScaler(), RidgeCV()), Xm, ym, cv=5).mean()
    st.markdown(f"مثال تفاعلات: y = x₁x₂ + x₃² ⇒ R² خطي = **{s1:.2f}**، وبعد PolynomialFeatures(2) = **{s2:.2f}**.")
if at_least("research"):
    researcher_note(["Natural cubic splines تفرض خطية خارج أطراف البيانات ⇒ استقراء أكثر أمانًا.",
                     "GAMs (Generalized Additive Models) = Splines لكل خاصية مع تنظيم النعومة: تفسير عالٍ ومرونة معتدلة."])
mistakes(["درجة عالية دون تنظيم أو قياس.", "استقراء كثير الحدود خارج مدى البيانات.", "اختيار الدرجة على الاختبار."])
page_footer("polynomial_regression",
            takeaways=["خطي في المعاملات، غير خطي في المدخلات.", "الدرجة معامل فائق يُختار بـCV.",
                       "Splines محلية وأكثر استقرارًا من كثير حدود عالمي."])
