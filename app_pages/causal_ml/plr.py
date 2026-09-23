import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils import causal as C
from utils.datasets import make_plr
from utils.plotting import interval_plot, plot

page_header("plr")

formula(r"Y = D\theta_0 + g_0(X) + \zeta,\qquad D = m_0(X) + V", title="PLR model",
        symbols={r"\theta_0": "أثر D الخطي الثابت", r"g_0": "أثر X على Y (أي شكل)", r"m_0": "كيف يعتمد D على X (الإرباك)"})
comparison_table([
    {"الدرجة": "partialling out (افتراضي في DoubleML)", "الإزعاج": "ℓ₀(X) = E[Y|X]، m₀(X) = E[D|X]",
     "ψ": "(Y − ℓ(X) − θ(D − m(X)))(D − m(X))", "ملاحظة": "كلاهما تنبؤ مباشر؛ الأبسط"},
    {"الدرجة": "IV-type", "الإزعاج": "g₀(X) = E[Y − Dθ|X]، m₀(X)", "ψ": "(Y − Dθ − g(X))(D − m(X))",
     "ملاحظة": "يحتاج ml_g وتقديرًا أوليًا لـθ"},
])

st.markdown("## PLR Residualization Lab")
c1, c2, c3, c4 = st.columns(4)
learner_name = c1.selectbox("متعلم الإزعاج", ["Random Forest", "HistGradientBoosting", "Lasso (polynomial features)"], key="plr_l")
conf = c2.slider("قوة الإرباك", 0.0, 2.0, 1.0, 0.25, key="plr_conf")
nonlin = c3.slider("اللاخطية", 0.0, 1.0, 1.0, 0.25, key="plr_nl")
n = c4.select_slider("n", [500, 1000, 2000], value=1000, key="plr_n")


def _learner(name):
    if name == "Random Forest":
        return RandomForestRegressor(150, max_features=0.5, min_samples_leaf=5, random_state=0, n_jobs=1)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0)
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LassoCV(cv=3, random_state=0, max_iter=5000))


@st.cache_data(show_spinner="Cross-fitting ثم حل الدرجة…", max_entries=24)
def _run(learner_name: str, conf: float, nonlin: float, n: int):
    df = make_plr(n=n, confounding=conf, nonlinearity=nonlin, seed=3)
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    lrn = _learner(learner_name)
    res = C.dml_plr(X, y, d, lrn, lrn, n_folds=5)
    return res, [C.naive_ols(None, y, d), C.naive_ols(X, y, d)], y, d


res, naives, y, d = _run(learner_name, conf, nonlin, n)
u, v = y - res.extra["l_hat"], d - res.extra["m_hat"]
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Scatter(x=d, y=y, mode="markers", marker=dict(color=PALETTE["muted"], opacity=0.35, size=5)))
    b = np.polyfit(d, y, 1)
    xs = np.linspace(d.min(), d.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=PALETTE["coral"], width=3)))
    fig.update_layout(title=f"Raw Y on D: slope {b[0]:.3f} (confounded)", xaxis_title="D", yaxis_title="Y", showlegend=False, height=360)
    plot(fig)
with c2:
    fig = go.Figure(go.Scatter(x=v, y=u, mode="markers", marker=dict(color=PALETTE["purple"], opacity=0.35, size=5)))
    xs = np.linspace(v.min(), v.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=res.theta * xs, mode="lines", line=dict(color=PALETTE["teal"], width=3)))
    fig.update_layout(title=f"Residualised Û on V̂: slope θ̂ = {res.theta:.3f}", xaxis_title="V̂ = D − m̂(X)",
                      yaxis_title="Û = Y − ℓ̂(X)", showlegend=False, height=360)
    plot(fig)
names = ["Naive: Y on D", "OLS with linear X", f"DML-PLR ({learner_name})"]
ests = naives + [res]
plot(interval_plot(names, [e.theta for e in ests], [e.ci[0] for e in ests], [e.ci[1] for e in ests], truth=0.5,
                   title="Estimates of θ₀ with 95% CIs"))
with st.container(horizontal=True):
    st.metric("θ̂ (DML)", f"{res.theta:.4f}", border=True)
    st.metric("SE", f"{res.se:.4f}", border=True)
    st.metric("RMSE ℓ̂ (out-of-fold)", f"{res.extra['rmse_l']:.3f}", border=True)
    st.metric("RMSE m̂ (out-of-fold)", f"{res.extra['rmse_m']:.3f}", border=True)
if st.button("سجّل التقدير", key="plr_log", icon=":material/history:", type="tertiary"):
    log_experiment("PLR Residualization Lab", f"DML-PLR with {learner_name}", {"confounding": conf, "nonlinearity": nonlin, "n": n,
                   "n_folds": 5}, {"theta": res.theta, "se": res.se}, seed=3, dataset="make_plr (θ₀ = 0.5)", split="5-fold cross-fitting")
    st.toast("سُجّل.", icon=":material/check:")
intuition("اليسار: الميل الخام يخلط أثر D مع أثر X (لأن X يحرك الاثنين). اليمين: بعد إزالة كل ما يفسّره X من Y ومن D، الميل "
          "المتبقي هو أثر D وحده.")
why("قيّم دقة الإزعاج خارج الطية لكن لا تختر النموذج بـθ̂.", "RMSE لـℓ̂ وm̂ تشخيص مشروع؛ اختيار المتعلم الذي يعطي θ̂ «أجمل» هو p-hacking.")

st.markdown("## مطابقة مع حزمة DoubleML")
dml = optional("doubleml")


def _code(p):
    return ("import doubleml as dml\n"
            "data = dml.DoubleMLData(df, y_col='y', d_cols='d', x_cols=[f'x{i}' for i in range(1, 11)])\n"
            "ml = RandomForestRegressor(150, max_features=0.5, min_samples_leaf=5)\n"
            f"plr = dml.DoubleMLPLR(data, ml_l=ml, ml_m=ml, n_folds={p['k']}, score='{p['score']}')\n"
            + ("plr = dml.DoubleMLPLR(data, ml_l=ml, ml_m=ml, ml_g=ml, n_folds=%d, score='IV-type')\n" % p["k"] if p["score"] == "IV-type" else "")
            + "plr.fit(); print(plr.summary)")


def _run_pkg(k, score):
    if dml is None:
        return "DoubleML غير مثبتة."
    df = make_plr(n=n, confounding=conf, nonlinearity=nonlin, seed=3)
    data = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=[f"x{i}" for i in range(1, 11)])
    ml = RandomForestRegressor(150, max_features=0.5, min_samples_leaf=5, random_state=0, n_jobs=1)
    kw = dict(ml_g=ml) if score == "IV-type" else {}
    plr = dml.DoubleMLPLR(data, ml_l=ml, ml_m=ml, n_folds=k, score=score, **kw)
    folds = C.fold_indices(len(df), k, seed=7)
    plr.set_sample_splitting([[(tr, te) for tr, te in folds]])
    plr.fit()
    X, yy, dd = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    mine = ""
    if score == "partialling out":
        l_hat = C.cross_fit_predict(ml, X, yy, folds)
        m_hat = C.cross_fit_predict(ml, X, dd, folds)
        r = C.plr_from_nuisance(yy, dd, l_hat, m_hat)
        mine = f"\n\n**تنفيذنا من الصفر على الطيات نفسها:** θ̂ = {r.theta:.6f}، SE = {r.se:.6f}"
    return [plr.summary.round(4), mine or "(درجة IV-type: قارن مع partialling out.)"]


if dml is None:
    missing_notice("doubleml")
code_lab("plr_pkg", "DoubleMLPLR vs our from-scratch estimator (same folds)", _code, _run_pkg,
         lambda: {"k": st.slider("n_folds", 2, 8, 5, key="plrp_k"),
                  "score": st.segmented_control("score", ["partialling out", "IV-type"], default="partialling out", key="plrp_s",
                                                required=True)},
         explanation="مع الطيات نفسها (set_sample_splitting) والمتعلم نفسه يتطابق التقديران حتى دقة الآلة — تحقق مستقل من التنفيذ.")

if at_least("advanced"):
    st.markdown("## متقدم: التباين التقاربي")
    st.latex(r"\hat\sigma^2 = \frac{\frac1n\sum_i \hat V_i^2(\hat U_i - \hat\theta\hat V_i)^2}{\big(\frac1n\sum_i\hat V_i^2\big)^2}")
    st.markdown("هذا تقدير Sandwich (قوي أمام عدم التجانس). مع بيانات مجمّعة استخدم نسخة Cluster-robust (DoubleMLClusterData / "
                "cluster_cols في الإصدار الحالي).")
if at_least("research"):
    researcher_note(["Robinson (1988) قدّم مقدِّر التحييد مع نواة؛ Chernozhukov et al. (2018) عمّماه لأي ML مع Cross-fitting.",
                     "PLR يفترض أثرًا ثابتًا θ₀؛ مع أثر غير متجانس يقدّر متوسطًا مرجحًا بـVar(D|X) لا ATE."])
    st.markdown(cite("robinson1988", "chernozhukov2018", "bach2022"))
mistakes(["اختيار المتعلم الذي يعطي θ̂ المرغوب.", "إدخال متغيرات بعد المعالجة في X.", "تفسير θ̂ كـATE مع أثر غير متجانس.",
          "تجاهل تكرار التقسيم في عينات صغيرة."])
page_footer("plr",
            takeaways=["PLR: انحدر بواقي Y على بواقي D بعد إزالة X بـML خارج الطية.",
                       "تنفيذنا يطابق DoubleMLPLR على الطيات نفسها.", "θ₀ في PLR متوسط موزون بـVar(D|X) إن كان الأثر غير متجانس."])
