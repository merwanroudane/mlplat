import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

from components.callouts import causal_caution, intuition, mistakes, researcher_note, why
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.inspection import ice_curves
from utils.plotting import plot

page_header("pdp_ice")

formula(r"\hat f_S(x_S) = \frac1n\sum_{i=1}^n \hat f\big(x_S, x_{C}^{(i)}\big)", title="Partial dependence (Friedman, 2001)",
        symbols={"x_S": "الخاصية (أو الخاصيتان) المدروسة", "x_C^{(i)}": "قيم بقية الخصائص للملاحظة i"},
        intuition="ثبّت الخاصية على قيمة، واترك البقية كما هي في البيانات، وخذ متوسط التنبؤ. ICE = المنحنى نفسه لكل ملاحظة دون متوسط.")

st.markdown("## PDP/ICE Lab")
st.caption("DGP معروف: y = 2·x1 + x1·x2 (تفاعل) + sin(3·x3) + ε. نموذج HistGradientBoosting.")
rng = np.random.default_rng(0)
n = 1500
corr = st.slider("ترابط x1 وx2", 0.0, 0.95, 0.0, 0.05, key="pdp_corr")
Z = rng.multivariate_normal([0, 0, 0], [[1, corr, 0], [corr, 1, 0], [0, 0, 1]], n)
y = 2 * Z[:, 0] + 1.5 * Z[:, 0] * Z[:, 1] + np.sin(3 * Z[:, 2]) + rng.normal(scale=0.3, size=n)


@st.cache_resource(show_spinner="يدرّب النموذج…", max_entries=8)
def _model(corr: float):
    return HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Z, y)


model = _model(corr)
feat = st.segmented_control("الخاصية", ["x1", "x2", "x3"], default="x1", key="pdp_feat", required=True)
j = int(feat[1]) - 1
grid = np.linspace(np.percentile(Z[:, j], 2), np.percentile(Z[:, j], 98), 40)
ice = ice_curves(model, Z, j, grid, n_lines=80, seed=0)
color_by = Z[np.random.default_rng(0).choice(n, 80, replace=False), 1 if j == 0 else 0]
fig = go.Figure()
for k in range(len(ice)):
    c = "rgba(247,103,7,0.35)" if color_by[k] > 0 else "rgba(28,126,214,0.35)"
    fig.add_trace(go.Scatter(x=grid, y=ice[k] - ice[k, 0] * 0, mode="lines", line=dict(color=c, width=1), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=grid, y=ice.mean(0), mode="lines", name="PDP (average)", line=dict(color="#212529", width=4)))
pd_sk = partial_dependence(model, Z, [j], grid_resolution=40, kind="average")
fig.add_trace(go.Scatter(x=pd_sk["grid_values"][0], y=pd_sk["average"][0], mode="lines", name="sklearn partial_dependence",
                         line=dict(color=PALETTE["teal"], width=2, dash="dash")))
fig.update_layout(title=f"ICE curves for {feat} (orange: {'x2' if j == 0 else 'x1'} > 0, blue: ≤ 0) + PDP", height=440,
                  xaxis_title=feat, yaxis_title="prediction")
plot(fig)
if feat in ("x1", "x2"):
    intuition("منحنيات ICE تتفرع حسب قيمة الخاصية الأخرى: هذا توقيع **التفاعل** x1·x2. الـPDP (المتوسط) يخفي هذا التباين.")
else:
    intuition("لـx3 منحنيات ICE متوازية: لا تفاعل، والـPDP يلخّص العلاقة جيدًا (شكل جيبي).")
if corr > 0.6:
    st.warning("مع ترابط قوي، PDP يقيّم النموذج عند تركيبات غير واقعية (x1 مرتفع مع x2 منخفض جدًا) — استقراء. البديل: ALE "
               "(Accumulated Local Effects) يعتمد على التوزيع الشرطي.")

st.markdown("## PDP ثنائي")
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402,F401  (documented API)
pd2 = partial_dependence(model, Z, [(0, 1)], grid_resolution=25)
fig = go.Figure(go.Contour(x=pd2["grid_values"][0], y=pd2["grid_values"][1], z=pd2["average"][0].T,
                           colorscale=[[0, "#E7F5FF"], [0.5, "#F3F0FF"], [1, "#FFE8CC"]], contours=dict(showlabels=True)))
fig.update_layout(title="Two-way PDP (x1, x2): curved contours reveal the interaction", xaxis_title="x1", yaxis_title="x2", height=400)
plot(fig)
st.code("""from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X, features=["x1", ("x1", "x2")], kind="both")  # PDP + ICE""", language="python")
why("اعرض ICE مع PDP دائمًا.", "المتوسط قد يكون مسطحًا بينما الأثر الفردي قوي ومتعاكس الاتجاه (تفاعلات تلغي بعضها).")
causal_caution("PDP يصف كيف يتغير **تنبؤ النموذج**، لا ما يحدث في الواقع لو تدخّلنا؛ التفسير السببي يتطلب افتراضات عدم الإرباك.")

if at_least("research"):
    researcher_note(["Goldstein et al. (2015) قدّموا ICE؛ Apley & Zhu (2020) قدّموا ALE لمعالجة الترابط.",
                     "Zhao & Hastie (2021): PDP يطابق أثرًا سببيًا فقط تحت شروط Backdoor على المتغيرات المثبتة."])
    st.markdown(cite("friedman2001", "goldstein2015"))
mistakes(["PDP وحده مع تفاعلات قوية.", "PDP مع خصائص شديدة الترابط.", "قراءة PDP كأثر سببي."])
page_footer("pdp_ice",
            takeaways=["PDP = متوسط منحنيات ICE.", "تباعد منحنيات ICE = تفاعل.", "الترابط يجعل PDP يستقرئ."])
