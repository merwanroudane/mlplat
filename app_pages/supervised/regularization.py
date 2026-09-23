import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, RidgeCV, enet_path, lasso_path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least, log_experiment
from core.theme import PALETTE, SEQUENCE
from utils.datasets import xy
from utils.plotting import lines, plot

page_header("regularization")
algorithm_profile("ridge")

st.markdown("## الصياغات الثلاث")
formula(r"\min_\beta\ \tfrac{1}{2n}\|y-X\beta\|_2^2 + \alpha\Big(\rho\|\beta\|_1 + \tfrac{1-\rho}{2}\|\beta\|_2^2\Big)",
        title="Elastic Net objective (scikit-learn parametrisation)",
        symbols={r"\alpha": "قوة التنظيم (alpha)", r"\rho": "l1_ratio: 1 = Lasso، 0 = Ridge",
                 r"\|\beta\|_1": "مجموع القيم المطلقة", r"\|\beta\|_2^2": "مجموع المربعات"},
        intuition="نقايض قليلًا من الملاءمة على التدريب مقابل معاملات أصغر ⇒ تباين أقل وتعميم أفضل.",
        example="Ridge في scikit-learn يستخدم ‖y − Xβ‖² + α‖β‖² (دون 1/2n) — لذا قيم alpha غير قابلة للمقارنة مباشرة بين Ridge وLasso.")
comparison_table([
    {"": "Ridge (L2)", "الشكل الهندسي": "دائرة/كرة", "الحل": "مغلق", "الندرة": "لا", "الخصائص المترابطة": "يوزّع الوزن بينها",
     "متى": "كثير من الخصائص المفيدة قليلًا"},
    {"": "Lasso (L1)", "الشكل الهندسي": "ماسة", "الحل": "Coordinate descent", "الندرة": "نعم", "الخصائص المترابطة":
        "يختار واحدة اعتباطيًا", "متى": "قليل من الخصائص المهمة"},
    {"": "Elastic Net", "الشكل الهندسي": "ماسة مستديرة الحواف", "الحل": "Coordinate descent", "الندرة": "نعم",
     "الخصائص المترابطة": "يختار المجموعة معًا", "متى": "خصائص مترابطة + ندرة"},
])

st.markdown("## الهندسة: لماذا يعطي L1 أصفارًا؟")
t = st.slider("حجم القيد t (أكبر = تنظيم أضعف)", 0.3, 3.0, 1.0, 0.1, key="reg_t")
b_ols = np.array([2.0, 0.6])
A = np.array([[1.0, 0.5], [0.5, 1.2]])
g1, g2 = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1.5, 2, 200))
d = np.stack([g1 - b_ols[0], g2 - b_ols[1]], -1)
Z = np.einsum("...i,ij,...j->...", d, A, d)
pts = np.stack([g1.ravel(), g2.ravel()], 1)
inside_l1 = np.abs(pts).sum(1) <= t
inside_l2 = (pts ** 2).sum(1) <= t ** 2
z = Z.ravel()
b_l1 = pts[inside_l1][np.argmin(z[inside_l1])] if inside_l1.any() else np.zeros(2)
b_l2 = pts[inside_l2][np.argmin(z[inside_l2])] if inside_l2.any() else np.zeros(2)
fig = go.Figure(go.Contour(x=g1[0], y=g2[:, 0], z=Z, contours=dict(coloring="lines", showlabels=False),
                           line=dict(width=1), colorscale=[[0, "#ADB5BD"], [1, "#ADB5BD"]], showscale=False, hoverinfo="skip"))
th = np.linspace(0, 2 * np.pi, 200)
fig.add_trace(go.Scatter(x=t * np.cos(th), y=t * np.sin(th), mode="lines", name="L2 ball", line=dict(color=PALETTE["sky"], width=2.5)))
fig.add_trace(go.Scatter(x=[t, 0, -t, 0, t], y=[0, t, 0, -t, 0], mode="lines", name="L1 ball", line=dict(color=PALETTE["coral"], width=2.5)))
fig.add_trace(go.Scatter(x=[b_ols[0]], y=[b_ols[1]], mode="markers", name="OLS", marker=dict(symbol="star", size=14, color="#212529")))
fig.add_trace(go.Scatter(x=[b_l2[0]], y=[b_l2[1]], mode="markers", name="Ridge solution", marker=dict(size=13, color=PALETTE["sky"])))
fig.add_trace(go.Scatter(x=[b_l1[0]], y=[b_l1[1]], mode="markers", name="Lasso solution",
                         marker=dict(size=13, color=PALETTE["coral"], symbol="diamond")))
fig.update_layout(height=430, xaxis=dict(title="β₁", range=[-1, 3]), yaxis=dict(title="β₂", range=[-1.5, 2], scaleanchor="x"),
                  title=f"Lasso: β = ({b_l1[0]:.2f}, {b_l1[1]:.2f}) · Ridge: β = ({b_l2[0]:.2f}, {b_l2[1]:.2f})")
plot(fig)
intuition("الحل المنظَّم هو أول نقطة تلمسها خطوط تساوي الخسارة (قطوع ناقصة حول OLS) من منطقة القيد. زوايا الماسة تقع على "
          "المحاور، فكثيرًا ما يكون أول تماس عند زاوية ⇒ β₂ = 0 تمامًا. الدائرة بلا زوايا ⇒ انكماش دون أصفار.")

st.markdown("## مختبر مسارات التنظيم: alpha وl1_ratio")
X, y = xy("high_dim")
Xs = StandardScaler().fit_transform(X)
ys = (y - y.mean()).to_numpy()
c1, c2 = st.columns(2)
l1_ratio = c1.slider("l1_ratio (0 ≈ Ridge، 1 = Lasso)", 0.05, 1.0, 1.0, 0.05, key="reg_l1")
log_alpha = c2.slider("log₁₀ alpha", -3.0, 1.5, -0.5, 0.1, key="reg_alpha")
alpha = 10 ** log_alpha


@st.cache_data(show_spinner="يحسب المسار…", max_entries=32)
def _path(l1_ratio: float):
    alphas = np.logspace(1.5, -3, 60)
    if l1_ratio >= 0.999:
        al, coefs, _ = lasso_path(Xs, ys, alphas=alphas)
    else:
        al, coefs, _ = enet_path(Xs, ys, l1_ratio=l1_ratio, alphas=alphas)
    cv = [-cross_val_score(ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=20000), Xs, ys, cv=KFold(5, shuffle=True, random_state=0),
                           scoring="neg_mean_squared_error").mean() for a in alphas[::3]]
    return al, coefs, alphas[::3], cv


al, coefs, al_cv, cv = _path(l1_ratio)
fig = go.Figure()
true_idx = set(range(10))
for j in range(coefs.shape[0]):
    fig.add_trace(go.Scatter(x=np.log10(al), y=coefs[j], mode="lines", showlegend=False,
                             line=dict(color=SEQUENCE[j % len(SEQUENCE)] if j in true_idx else "rgba(92,103,125,0.25)",
                                       width=2.4 if j in true_idx else 1)))
fig.add_vline(x=log_alpha, line=dict(color="#212529", dash="dash"), annotation_text="current α")
fig.update_layout(title="Coefficient paths (coloured = 10 truly non-zero features, grey = 90 noise features)",
                  xaxis_title="log₁₀ α (← weaker · stronger →)", yaxis_title="β", height=420, xaxis=dict(autorange="reversed"))
plot(fig)
m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000).fit(Xs, ys)
nz = np.abs(m.coef_) > 1e-8
cv_now = -cross_val_score(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000), Xs, ys,
                          cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").mean()
with st.container(horizontal=True):
    st.metric("معاملات غير صفرية", int(nz.sum()), border=True)
    st.metric("منها حقيقية (من 10)", int(nz[:10].sum()), border=True)
    st.metric("ضجيج مختار (من 90)", int(nz[10:].sum()), border=True)
    st.metric("CV MSE", f"{cv_now:.2f}", border=True)
fig = lines(np.log10(al_cv), {"5-fold CV MSE": cv}, title="Validation score along the path", xaxis="log₁₀ α", yaxis="MSE",
            markers=True)
fig.add_vline(x=log_alpha, line=dict(color="#212529", dash="dash"))
fig.update_xaxes(autorange="reversed")
plot(fig, height=300)
if st.button("سجّل هذه التجربة", key="reg_log", icon=":material/history:", type="tertiary"):
    log_experiment("Regularization Path Lab", "ElasticNet", {"alpha": alpha, "l1_ratio": l1_ratio},
                   {"cv_mse": cv_now, "nonzero": int(nz.sum())}, seed=0, dataset="high_dim", split="KFold(5, shuffle, rs=0)")
    st.toast("سُجّلت.", icon=":material/check:")
why("اختر alpha بـLassoCV / RidgeCV / ElasticNetCV داخل Pipeline مع StandardScaler.",
    "العقوبة تعتمد على مقياس كل خاصية؛ دون القياس تُعاقَب الخصائص ذات الوحدات الصغيرة أكثر ظلمًا.")


@st.cache_data(show_spinner=False)
def _cv_models():
    cv = KFold(5, shuffle=True, random_state=0)
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 50)).fit(Xs, ys)
    lasso = LassoCV(alphas=50, cv=cv, max_iter=20000).fit(Xs, ys)
    enet = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95, 1.0], alphas=50, cv=cv, max_iter=20000).fit(Xs, ys)
    return [
        {"model": "RidgeCV", "chosen α": ridge.alpha_, "l1_ratio": 0.0, "non-zero": int(np.sum(np.abs(ridge.coef_) > 1e-8))},
        {"model": "LassoCV", "chosen α": lasso.alpha_, "l1_ratio": 1.0, "non-zero": int(np.sum(np.abs(lasso.coef_) > 1e-8))},
        {"model": "ElasticNetCV", "chosen α": enet.alpha_, "l1_ratio": enet.l1_ratio_,
         "non-zero": int(np.sum(np.abs(enet.coef_) > 1e-8))},
    ]


st.dataframe(pd.DataFrame(_cv_models()).style.format({"chosen α": "{:.4f}"}), hide_index=True, width="stretch")
st.caption("ملاحظة إصدار: في scikit-learn 1.9 أصبح `n_alphas` في lasso_path/enet_path مهجورًا؛ مرّر `alphas` كعدد صحيح أو مصفوفة.")

hyperparameter_table("Ridge")
hyperparameter_table("Lasso")
hyperparameter_table("ElasticNet")

if at_least("advanced"):
    st.markdown("## متقدم: الخصائص المترابطة")
    rho = st.slider("ρ بين خاصيتين متطابقتي الأثر", 0.0, 0.99, 0.95, 0.01, key="reg_rho")
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(30):
        Xc = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=150)
        yc = Xc[:, 0] + Xc[:, 1] + rng.normal(size=150)
        l = LassoCV(cv=5).fit(Xc, yc).coef_
        r = RidgeCV(alphas=np.logspace(-3, 3, 30)).fit(Xc, yc).coef_
        rows.append({"rep": rep, "lasso β1": l[0], "lasso β2": l[1], "ridge β1": r[0], "ridge β2": r[1]})
    d = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["lasso β1"], y=d["lasso β2"], mode="markers", name="Lasso", marker=dict(color=PALETTE["coral"], size=9)))
    fig.add_trace(go.Scatter(x=d["ridge β1"], y=d["ridge β2"], mode="markers", name="Ridge", marker=dict(color=PALETTE["sky"], size=9,
                                                                                                            symbol="diamond")))
    fig.update_layout(title="30 resamples: Lasso jumps between features, Ridge shares the weight", xaxis_title="β₁",
                      yaxis_title="β₂", height=380)
    plot(fig)
    st.markdown("**التعقيد:** Ridge بـSVD O(np²)؛ Lasso بـCoordinate descent O(np) لكل دورة؛ مسار كامل مع Warm starts سريع.")
if at_least("research"):
    researcher_note(["Lasso لا يعطي p-values صالحة بعد الاختيار؛ Post-double-selection (Belloni, Chernozhukov & Hansen, 2014) "
                     "هو الجسر إلى DML للاستدلال على معامل معالجة مع ضوابط عالية الأبعاد.",
                     "شروط استرجاع الدعم (Irrepresentable condition) نادرًا ما تتحقق مع الترابط القوي.",
                     "Adaptive Lasso وGroup Lasso امتدادات تستحق الاطلاع."])
    st.markdown(cite("hoerl1970", "tibshirani1996", "zou2005", "friedman2010", "belloni2014"))

template_checklist({3: "صيغة Elastic Net", 5: "الرسم الهندسي L1/L2", 6: "الهدف المنظَّم", 7: "Coordinate descent (وحدة المحسّنات)",
                    9: "جداول المعاملات الفائقة", 21: "Coordinate descent من الصفر (وحدة المحسّنات المتقدمة)",
                    22: "RidgeCV/LassoCV/ElasticNetCV", 23: "مختبر المسارات", 24: "منزلقا alpha وl1_ratio"})
mistakes(["تنظيم دون قياس.", "تفسير معاملات Lasso كمعاملات غير متحيزة.", "استنتاج أن الخاصية المُصفَّرة «غير مهمة».",
          "مقارنة alpha بين Ridge وLasso مباشرة."])
page_footer("regularization",
            takeaways=["L2 ينكمش، L1 يُصفِّر، Elastic Net يجمعهما.", "المسار يبيّن متى تدخل كل خاصية.",
                       "اختر alpha بـCV داخل Pipeline مع القياس."])
