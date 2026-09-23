import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from components.algorithm_profile import hyperparameter_table
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils import causal as C
from utils.datasets import load_dataset
from utils.plotting import interval_plot, plot

page_header("irm")

formula(r"Y = g_0(D, X) + U,\quad \mathbb E[U\mid X,D]=0;\qquad D = m_0(X) + V,\quad m_0(X) = P(D=1\mid X)",
        title="Interactive regression model (binary treatment)",
        intuition="لا نفترض أثرًا ثابتًا ولا شكلًا جمعيًا: g₀(1, X) وg₀(0, X) دالتان حرتان، والأثر قد يتغير مع X.")
formula(r"\psi_{ATE} = g(1,X) - g(0,X) + \frac{D\,(Y-g(1,X))}{m(X)} - \frac{(1-D)(Y-g(0,X))}{1-m(X)} - \theta",
        title="Doubly robust (AIPW) orthogonal score",
        symbols={"g(d, X)": "نموذج النتيجة لكل مجموعة", "m(X)": "درجة الميل"},
        intuition="الحدان الأولان = تقدير بالانحدار؛ الحدان التاليان يصححان أخطاءه بأوزان مقلوب الميل. الدرجة متسقة إن صح أي من "
                  "النموذجين (Doubly robust) ومتعامدة.")
comparison_table([
    {"المقدِّر": "Regression adjustment", "يعتمد على": "g فقط", "المتانة": "يفشل إن أخطأ g"},
    {"المقدِّر": "IPW", "يعتمد على": "m فقط", "المتانة": "يفشل إن أخطأ m؛ تباين هائل مع ميل قرب 0/1"},
    {"المقدِّر": "AIPW / DML-IRM", "يعتمد على": "g وm", "المتانة": "متسق إن صح أحدهما؛ خطأ من الرتبة الثانية"},
])

st.markdown("## Overlap Lab")
df = load_dataset("causal")
X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
c1, c2 = st.columns(2)
score = c1.segmented_control("المعلمة", ["ATE", "ATT"], default="ATE", key="irm_score", required=True)
trim = c2.select_slider("trimming threshold", [0.0, 0.005, 0.01, 0.025, 0.05, 0.1], value=0.01, key="irm_trim")


@st.cache_data(show_spinner="Cross-fitting لـg وm…", max_entries=24)
def _run(score: str, trim: float):
    g = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=1)
    m = RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1)
    res = C.dml_irm(X, y, d, g, m, n_folds=5, score=score, trim=max(trim, 1e-6))
    ipw = C.ipw_ate(y, d, np.clip(res.extra["m_hat"], max(trim, 1e-6), 1 - max(trim, 1e-6)))
    return res, ipw


res, ipw = _run(score, trim)
truth = df["tau"].mean() if score == "ATE" else df.loc[df.d == 1, "tau"].mean()
naive = y[d == 1].mean() - y[d == 0].mean()
ests = [("Naive difference", naive, np.nan, np.nan), ("OLS with linear X", C.naive_ols(X, y, d).theta, *C.naive_ols(X, y, d).ci)]
if score == "ATE":
    ests.append(("IPW (cross-fitted m̂)", ipw, np.nan, np.nan))
ests.append((f"DML-IRM ({score})", res.theta, *res.ci))
plot(interval_plot([e[0] for e in ests], [e[1] for e in ests], [e[2] for e in ests], [e[3] for e in ests], truth=round(truth, 3),
                   title=f"{score}: estimates (true value from the DGP ≈ {truth:.3f})"))
mh = res.extra["m_hat"]
fig = go.Figure()
fig.add_trace(go.Histogram(x=mh[d == 1], name="treated", nbinsx=40, opacity=0.6, marker_color=PALETTE["coral"]))
fig.add_trace(go.Histogram(x=mh[d == 0], name="control", nbinsx=40, opacity=0.6, marker_color=PALETTE["sky"]))
fig.update_layout(barmode="overlay", title="Cross-fitted propensity scores m̂(X)", xaxis_title="m̂(X)", height=300)
plot(fig)
with st.container(horizontal=True):
    st.metric("θ̂", f"{res.theta:.3f}", border=True)
    st.metric("SE", f"{res.se:.3f}", border=True)
    st.metric("min / max m̂", f"{mh.min():.3f} / {mh.max():.3f}", border=True)
    st.metric("max weight 1/m̂", f"{(1 / np.minimum(mh, 1 - mh)).max():.1f}", border=True)
if trim == 0.0:
    warning("دون قص، درجات ميل قرب 0 أو 1 تعطي أوزانًا ضخمة؛ ملاحظة واحدة قد تسيطر على التقدير.")
why("افحص توزيع درجة الميل قبل الإبلاغ.", "التداخل الضعيف يعني أن الاستنتاج يعتمد على الاستقراء؛ القص يحسّن الاستقرار لكنه يغيّر "
    "المجتمع المستهدف ضمنيًا — صرّح به.")
intuition("ATT يحتاج فقط نموذج النتيجة للضوابط g(0, X) وأوزانًا للضوابط؛ لذلك يتطلب تداخلًا من جهة واحدة (m < 1).")

st.markdown("## DoubleMLIRM")
dml = optional("doubleml")
if dml is None:
    missing_notice("doubleml")


def _code(p):
    return ("data = dml.DoubleMLData(df, y_col='y', d_cols='d', x_cols=['x1', 'x2', 'x3', 'x4', 'x5'])\n"
            f"irm = dml.DoubleMLIRM(data, ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(),\n"
            f"                      score='{p['score']}', trimming_threshold={p['trim']}, n_folds=5)\n"
            "irm.fit(); irm.summary")


def _run_pkg(score, trim):
    data = dml.DoubleMLData(df.drop(columns=["tau", "true_ps"]), y_col="y", d_cols="d", x_cols=[f"x{i}" for i in range(1, 6)])
    irm = dml.DoubleMLIRM(data, ml_g=RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=1),
                          ml_m=RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1),
                          score=score, trimming_threshold=trim, n_folds=5)
    irm.fit()
    return [irm.summary.round(4), f"القيمة الحقيقية في DGP ≈ {df['tau'].mean() if score == 'ATE' else df.loc[df.d == 1, 'tau'].mean():.3f}"]


if dml is not None:
    code_lab("irm_pkg", "DoubleMLIRM on the teaching data", _code, _run_pkg,
             lambda: {"score": st.segmented_control("score", ["ATE", "ATTE"], default="ATE", key="irmp_s", required=True),
                      "trim": st.select_slider("trimming_threshold", [0.01, 0.025, 0.05], value=0.01, key="irmp_t")},
             explanation="في DoubleML اسم ATT هو 'ATTE'. القيم الافتراضية المتحقق منها: trimming_rule='truncate'، "
                         "trimming_threshold=0.01، normalize_ipw=False.")
hyperparameter_table("DoubleMLIRM")

if at_least("advanced"):
    st.markdown("## متقدم: الدرجة لـATT")
    st.latex(r"\psi_{ATT} = \frac{D(Y-g(0,X))}{p} - \frac{m(X)(1-D)(Y-g(0,X))}{p(1-m(X))} - \frac{D}{p}\theta,\quad p = \mathbb E[D]")
    st.markdown("`normalize_ipw=True` يطبّع الأوزان (Hájek) لتقليل التباين في العينات الصغيرة.")
if at_least("research"):
    researcher_note(["AIPW هو Efficient influence function لـATE تحت عدم الإرباك (Robins, Rotnitzky & Zhao, 1994؛ Hahn, 1998).",
                     "مع تداخل محدود فكّر في ATO (overlap weights) أو قصر المجتمع المستهدف وتوثيقه.",
                     "معايرة m̂ مهمة: الأوزان 1/m̂ تضخّم أخطاء الاحتمالات قرب الأطراف."])
    st.markdown(cite("chernozhukov2018", "rosenbaum1983", "belloni2017"))
mistakes(["IPW دون قص مع ميل متطرف.", "تجاهل فحص التداخل.", "استخدام مصنف غير معاير كدرجة ميل.",
          "الخلط بين ATE وATT في التفسير."])
page_footer("irm",
            takeaways=["IRM لمعالجة ثنائية دون افتراض أثر ثابت.", "درجة AIPW مزدوجة المتانة ومتعامدة.",
                       "التداخل والقص جزء من التحليل لا تفصيل تقني."])
