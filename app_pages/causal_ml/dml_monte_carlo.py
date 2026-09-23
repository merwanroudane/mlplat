import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from config import MAX_MC_REPS, RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header
from core.state import log_experiment
from core.theme import SEQUENCE
from utils import causal as C
from utils.datasets import make_plr
from utils.plotting import interval_plot, plot

page_header("dml_monte_carlo")

formula(r"Y = \theta_0 D + \kappa\big[(1-\lambda)\,g^{lin}(X) + \lambda\,g^{nl}(X)\big] + \sigma\varepsilon,\qquad "
        r"D = \kappa\big[(1-\lambda)\,m^{lin}(X) + \lambda\,m^{nl}(X)\big] + V",
        title="Simulation design (utils/datasets.make_plr)",
        symbols={r"\kappa": "قوة الإرباك", r"\lambda": "اللاخطية (0 خطي، 1 غير خطي)", r"\sigma": "الضجيج",
                 "X": "p متغيرات بترابط Toeplitz 0.5^|i−j|"},
        intuition="نعرف θ₀ لأننا صنعنا البيانات؛ فنستطيع قياس التحيز والتغطية بالتكرار — ما لا يمكن فعله ببيانات حقيقية.")

st.markdown("## الإعدادات")
c1, c2, c3, c4 = st.columns(4)
n = c1.select_slider("n", [200, 500, 1000, 2000], value=500, key="mc_n")
p = c2.select_slider("p", [5, 10, 20], value=10, key="mc_p")
conf = c3.slider("قوة الإرباك κ", 0.0, 2.0, 1.0, 0.25, key="mc_conf")
nonlin = c4.slider("اللاخطية λ", 0.0, 1.0, 1.0, 0.25, key="mc_nl")
c5, c6, c7, c8 = st.columns(4)
noise = c5.slider("الضجيج σ", 0.5, 3.0, 1.0, 0.25, key="mc_noise")
theta0 = c6.select_slider("θ₀", [0.0, 0.5, 1.0], value=0.5, key="mc_theta")
learner_name = c7.selectbox("المتعلم", ["Random Forest", "HistGradientBoosting", "Lasso + poly(2)"], key="mc_learner")
K = c8.slider("عدد الطيات", 2, 10, 5, key="mc_K")
reps = st.slider("عدد تكرارات Monte Carlo", 10, MAX_MC_REPS, 30, 10, key="mc_reps",
                 help=f"مقيد بـ{MAX_MC_REPS} للنشر العام. كل تكرار يدرّب 2K نموذج إزعاج مرتين.")
st.caption("التداخل (Overlap) في PLR مع معالجة مستمرة يظهر عبر Var(D|X): كلما زاد κ مقارنة بتباين V ضعُف التباين المتبقي في D "
           "وازداد عدم اليقين.")


def _learner(name):
    if name == "Random Forest":
        return RandomForestRegressor(60, max_features=0.5, min_samples_leaf=5, random_state=0, n_jobs=1)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, random_state=0)
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LassoCV(cv=3, max_iter=3000, random_state=0))


@st.cache_data(show_spinner=False, max_entries=16)
def _mc(n, p, conf, nonlin, noise, theta0, learner_name, K, reps):
    lrn = _learner(learner_name)
    rows = []
    for r in range(reps):
        df = make_plr(n=n, p=p, theta=theta0, confounding=conf, nonlinearity=nonlin, noise=noise, seed=RANDOM_SEED + 1000 + r)
        X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
        a = C.naive_ols(None, y, d)
        b = C.naive_ols(X, y, d)
        c = C.plr_no_crossfit(X, y, d, lrn, lrn)
        e = C.dml_plr(X, y, d, lrn, lrn, n_folds=K, seed=r)
        rows.append({"rep": r, "naive": a.theta, "naive_se": a.se, "ols": b.theta, "ols_se": b.se, "nocf": c.theta, "nocf_se": c.se,
                     "dml": e.theta, "dml_se": e.se, "rmse_l": e.extra["rmse_l"], "rmse_m": e.extra["rmse_m"]})
    return pd.DataFrame(rows)


if st.button("شغّل Monte Carlo", key="mc_run", type="primary", icon=":material/play_arrow:"):
    args = (n, p, conf, nonlin, noise, theta0, learner_name, K, reps)
    t0 = time.perf_counter()
    with st.spinner(f"{reps} تكرارًا…"):
        _mc(*args)
    st.session_state["mc_args"] = args
    st.session_state["mc_secs"] = time.perf_counter() - t0
if st.session_state.get("mc_args"):
    args = st.session_state["mc_args"]
    res = _mc(*args)
    th0 = args[5]
    names = {"naive": "Naive: Y on D", "ols": "OLS with linear X", "nocf": "Orthogonal, no cross-fitting", "dml": "DML (cross-fitted)"}
    summ = []
    for k, label in names.items():
        th, se = res[k].to_numpy(), res[f"{k}_se"].to_numpy()
        summ.append({"estimator": label, "mean θ̂": th.mean(), "bias": th.mean() - th0, "SD": th.std(ddof=1), "mean SE": se.mean(),
                     "RMSE": np.sqrt(np.mean((th - th0) ** 2)), "95% coverage": np.mean(np.abs(th - th0) <= 1.96 * se)})
    summ = pd.DataFrame(summ)
    st.dataframe(summ.style.format({c: "{:.4f}" for c in summ.columns if c != "estimator"} | {"95% coverage": "{:.1%}"}),
                 hide_index=True, width="stretch")
    fig = go.Figure()
    for i, (k, label) in enumerate(names.items()):
        fig.add_trace(go.Violin(y=res[k], name=label, box_visible=True, meanline_visible=True, line_color=SEQUENCE[i], points=False))
    fig.add_hline(y=th0, line=dict(color="#212529", dash="dash"), annotation_text=f"θ₀ = {th0}")
    fig.update_layout(title=f"Sampling distributions over {len(res)} replications ({st.session_state.get('mc_secs', 0):.0f} s)",
                      yaxis_title="θ̂", height=420, showlegend=False)
    plot(fig)
    last = res.iloc[-1]
    ests = [(names[k], last[k], last[k] - 1.96 * last[f"{k}_se"], last[k] + 1.96 * last[f"{k}_se"]) for k in names]
    plot(interval_plot([e[0] for e in ests], [e[1] for e in ests], [e[2] for e in ests], [e[3] for e in ests], truth=th0,
                       title="One replication: estimates with 95% CIs"))
    c1, c2 = st.columns(2)
    c1.metric("متوسط RMSE للإزعاج ℓ̂", f"{res['rmse_l'].mean():.3f}")
    c2.metric("متوسط RMSE للإزعاج m̂", f"{res['rmse_m'].mean():.3f}")
    if st.button("سجّل نتائج المحاكاة", key="mc_log", icon=":material/history:", type="tertiary"):
        dml_row = summ[summ.estimator == names["dml"]].iloc[0]
        log_experiment("DML Monte Carlo Lab", f"DML-PLR ({args[6]})", dict(zip(["n", "p", "confounding", "nonlinearity", "noise", "theta0",
                                                                                  "learner", "folds", "reps"], args)),
                       {"bias": dml_row["bias"], "rmse": dml_row["RMSE"], "coverage": dml_row["95% coverage"]}, seed=RANDOM_SEED + 1000,
                       dataset="make_plr", split=f"{args[7]}-fold cross-fitting")
        st.toast("سُجّلت.", icon=":material/check:")
    why("اقرأ التحيز والتغطية معًا.", "مقدِّر بتحيز صغير لكن أخطاء معيارية خاطئة يعطي فترات مضللة؛ التغطية القريبة من 95% هي الاختبار.")
else:
    st.info("اضبط الإعدادات ثم «شغّل Monte Carlo». جرّب: (1) λ = 0 — كل الطرق المعدلة تنجح؛ (2) λ = 1 مع Lasso خطي/OLS — تحيز؛ "
            "(3) κ = 0 — لا إرباك فحتى الساذج غير متحيز؛ (4) n صغير مع متعلم مرن — تغطية أقل من الاسمية.", icon=":material/lightbulb:")
intuition("متى يتعثر DML؟ متعلم لا يلتقط بنية الإزعاج (تحيز من الرتبة الثانية لا يختفي)، أو n صغير جدًا بحيث لا يتقارب الإزعاج، أو "
          "تباين متبقٍ صغير في D (تداخل ضعيف). وDML لا يحل أبدًا مشكلة مربك غير مقاس.")

researcher_note(["تصميم المحاكاة مستوحى من Chernozhukov et al. (2018)؛ الأرقام هنا تعليمية وليست إعادة إنتاج لجداول الورقة.",
                 "للنشر: أبلغ عن التحيز والـRMSE والتغطية لعدة n، ومع ≥ 500 تكرار؛ هنا الحد الأقصى مقيد لأسباب حسابية."])
st.markdown(cite("chernozhukov2018", "bach2022"))
mistakes(["الحكم من تكرار واحد.", "النظر إلى التحيز دون التغطية.", "تعميم نتائج DGP واحد على كل البيانات."])
page_footer("dml_monte_carlo",
            takeaways=["المحاكاة بالحقيقة المعروفة تكشف التحيز والتغطية.", "DML يتفوق حين يكون الإرباك غير خطي ويُلتقط بمتعلم مناسب.",
                       "الإزعاج السيئ وn الصغير يضعفان التغطية."])
