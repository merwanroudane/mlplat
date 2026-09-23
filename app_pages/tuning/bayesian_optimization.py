import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from components.algorithm_profile import hyperparameter_table
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from config import MAX_OPTUNA_TRIALS, RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import plot

page_header("bayesian_optimization")

st.markdown("## الحلقة: Surrogate → Acquisition → Evaluate")
formula(r"\text{EI}(\lambda) = \mathbb E\big[\max(0, f(\lambda) - f^+)\big] = (\mu-f^+)\Phi(z) + \sigma\,\phi(z),\quad z=\frac{\mu-f^+}{\sigma}",
        title="Expected improvement under a Gaussian-process surrogate",
        symbols={r"\mu(\lambda), \sigma(\lambda)": "تنبؤ النموذج البديل وعدم يقينه", "f^+": "أفضل قيمة حتى الآن"},
        intuition="الاستكشاف (σ كبير) مقابل الاستغلال (μ مرتفع): EI كبير حيث يُتوقع تحسن أو حيث لا نعرف.")

st.markdown("## Animation: تحسين بايزي أحادي البعد")
st.caption("دالة هدف مخفية (ضجيجية قليلًا). كل خطوة: نلائم GP على النقاط المقيَّمة، نحسب EI، ونقيّم عند أعلاه.")


def f_true(x):
    return np.sin(3 * x) * (1 - np.tanh(x ** 2)) + 0.4 * np.exp(-((x - 1.2) ** 2) / 0.05)


@st.cache_data(show_spinner=False)
def _bo_trace(n_steps: int = 10):
    rng = np.random.default_rng(1)
    xs = np.linspace(-2, 2, 400)
    X = list(rng.uniform(-2, 2, 3))
    Y = [f_true(x) for x in X]
    frames = []
    for _ in range(n_steps):
        with warnings.catch_warnings():  # the hidden objective has a sharp peak; hitting the length-scale bound is expected
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(Matern(length_scale=0.5, length_scale_bounds=(0.05, 5.0), nu=2.5), alpha=1e-4,
                                          normalize_y=True, random_state=0).fit(np.array(X)[:, None], Y)
        mu, sd = gp.predict(xs[:, None], return_std=True)
        best = max(Y)
        z = (mu - best) / np.maximum(sd, 1e-9)
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        nxt = xs[int(np.argmax(ei))]
        frames.append({"X": list(X), "Y": list(Y), "mu": mu, "sd": sd, "ei": ei, "next": nxt})
        X.append(nxt)
        Y.append(f_true(nxt))
    return xs, frames


xs, frames = _bo_trace()


def _bo_frame(i: int) -> None:
    fr = frames[i]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[fr["mu"] + 2 * fr["sd"], (fr["mu"] - 2 * fr["sd"])[::-1]], fill="toself",
                             fillcolor="rgba(28,126,214,0.12)", line=dict(width=0), name="±2σ"))
    fig.add_trace(go.Scatter(x=xs, y=f_true(xs), name="true objective (hidden)", line=dict(color=PALETTE["muted"], dash="dot")))
    fig.add_trace(go.Scatter(x=xs, y=fr["mu"], name="GP mean", line=dict(color=PALETTE["sky"], width=2.5)))
    fig.add_trace(go.Scatter(x=fr["X"], y=fr["Y"], mode="markers", name="evaluated", marker=dict(color=PALETTE["coral"], size=10)))
    fig.add_trace(go.Scatter(x=xs, y=fr["ei"] / max(fr["ei"].max(), 1e-12) * 0.6 - 1.4, name="EI (scaled)",
                             line=dict(color=PALETTE["purple"], width=2)))
    fig.add_vline(x=fr["next"], line=dict(color=PALETTE["purple"], dash="dash"), annotation_text="next trial")
    fig.update_layout(height=420, title=f"Iteration {i + 1}: {len(fr['X'])} evaluations · best so far {max(fr['Y']):.3f}",
                      yaxis=dict(range=[-1.5, 1.6]))
    plot(fig)


stepper("bo_anim", len(frames), _bo_frame, labels=[f"iteration {i + 1}" for i in range(len(frames))])

st.markdown("## Optuna: المفاهيم (متحقَّق منها في Optuna 5.0.0)")
comparison_table([
    {"المفهوم": "Study", "API": "optuna.create_study(direction='maximize', sampler=..., pruner=...)", "ملاحظة": "Optuna 5.0 يفرض الوسائط بالاسم"},
    {"المفهوم": "Trial", "API": "trial.suggest_float / suggest_int / suggest_categorical", "ملاحظة": "suggest_float(..., log=True) للمقاييس اللوغاريتمية"},
    {"المفهوم": "TPESampler", "API": "optuna.samplers.TPESampler(seed=...)", "ملاحظة": "الافتراضي؛ في 5.0 متعدد المتغيرات وconstant_liar=True افتراضيًا"},
    {"المفهوم": "RandomSampler", "API": "optuna.samplers.RandomSampler", "ملاحظة": "خط أساس"},
    {"المفهوم": "CmaEsSampler", "API": "optuna.samplers.CmaEsSampler", "ملاحظة": "فضاءات مستمرة وميزانية أكبر"},
    {"المفهوم": "GPSampler", "API": "optuna.samplers.GPSampler", "ملاحظة": "تحسين بايزي بـGP؛ مستقر في 5.0 مع qLogEI"},
    {"المفهوم": "QMCSampler", "API": "optuna.samplers.QMCSampler(qmc_type='sobol')", "ملاحظة": "تغطية منتظمة شبه عشوائية"},
    {"المفهوم": "Pruners", "API": "MedianPruner, HyperbandPruner, SuccessiveHalvingPruner, PercentilePruner, WilcoxonPruner",
     "ملاحظة": "trial.report(value, step) + trial.should_prune()"},
    {"المفهوم": "Multi-objective", "API": "create_study(directions=['maximize', 'minimize'])", "ملاحظة": "في 5.0 أصبح TPE الافتراضي بدل NSGA-II"},
])
hyperparameter_table("TPESampler")

st.markdown("## HPO / Optuna Lab")
optuna = optional("optuna")
if optuna is None:
    missing_notice("optuna")
else:
    st.caption(f"optuna {installed_version('optuna')} مثبتة. النموذج: HistGradientBoostingClassifier، الهدف: 3-fold ROC-AUC.")
    X, y = xy("classification")
    Xn, yn = X.to_numpy(), y.to_numpy()
    c1, c2, c3 = st.columns(3)
    sampler = c1.selectbox("Sampler", ["tpe", "random", "cmaes", "gp", "qmc"], key="opt_sampler")
    pruner = c2.selectbox("Pruner", ["none", "median", "hyperband"], key="opt_pruner")
    n_trials = c3.slider("عدد المحاولات", 5, MAX_OPTUNA_TRIALS, 25, 5, key="opt_n")
    space = {"learning_rate": ("log", 0.005, 0.5), "max_leaf_nodes": ("int", 4, 64), "min_samples_leaf": ("int", 5, 100),
             "l2_regularization": ("log", 1e-4, 10.0)}
    with st.expander("فضاء البحث"):
        st.code("\n".join(f"{k}: {v}" for k, v in space.items()), language="text")

    def make_model(p):
        return HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=RANDOM_SEED, **p)

    @st.cache_data(show_spinner="Optuna يبحث…", max_entries=16)
    def _study(sampler: str, pruner: str, n_trials: int):
        from utils.tuning import run_optuna
        return run_optuna(make_model, space, Xn, yn, n_trials, sampler=sampler, pruner=pruner, cv=3, scoring="roc_auc")

    if st.button("شغّل الدراسة", key="opt_run", type="primary", icon=":material/play_arrow:"):
        st.session_state["opt_args"] = (sampler, pruner, n_trials)
    if st.session_state.get("opt_args"):
        df = _study(*st.session_state["opt_args"])
        done = df[df["state"] == "COMPLETE"].copy()
        best = done.loc[done["value"].idxmax()]
        with st.container(horizontal=True):
            st.metric("أفضل ROC-AUC (CV)", f"{best['value']:.4f}", border=True)
            st.metric("محاولات مكتملة", len(done), border=True)
            st.metric("محاولات مُقلَّمة", int((df["state"] == "PRUNED").sum()), border=True)
            st.metric("الزمن الكلي", f"{df.attrs.get('total_seconds', np.nan):.1f} s", border=True)
        hist_best = done.sort_values("trial")["value"].cummax()
        fig = go.Figure(go.Scatter(x=done["trial"], y=done["value"], mode="markers", name="trial value",
                                   marker=dict(color=PALETTE["sky"], size=8)))
        fig.add_trace(go.Scatter(x=done.sort_values("trial")["trial"], y=hist_best, mode="lines", name="best so far",
                                 line=dict(color=PALETTE["coral"], width=3, shape="hv")))
        fig.update_layout(title="Optimisation history", xaxis_title="trial", yaxis_title="CV ROC-AUC", height=340)
        plot(fig)
        c1, c2 = st.columns(2)
        for col, param in ((c1, "learning_rate"), (c2, "max_leaf_nodes")):
            with col:
                f2 = go.Figure(go.Scatter(x=done[param], y=done["value"], mode="markers", marker=dict(color=done["trial"],
                                          colorscale=[[0, "#D0EBFF"], [1, "#7048E8"]], size=9, showscale=True,
                                          colorbar=dict(title="trial"))))
                f2.update_layout(title=f"Slice: {param}", xaxis_title=param, yaxis_title="value", height=300,
                                 xaxis_type="log" if param == "learning_rate" else "linear")
                plot(f2)
        st.markdown("**أفضل معاملات:** `" + ", ".join(f"{k}={best[k]:.4g}" for k in space) + "`")
        st.warning("أفضل قيمة CV متفائلة (اختيرت من بين محاولات كثيرة). قيّم أفضل إعداد على Test معزول أو بـNested CV. كلما زادت "
                   "المحاولات مع بيانات صغيرة، زاد خطر فرط ملاءمة التحقق.", icon=":material/warning:")
        if st.button("سجّل الدراسة", key="opt_log", icon=":material/history:", type="tertiary"):
            log_experiment("HPO / Optuna Lab", "HistGradientBoostingClassifier", {**{k: best[k] for k in space},
                           "sampler": sampler, "pruner": pruner, "n_trials": n_trials}, {"best_cv_auc": best["value"]},
                           seed=RANDOM_SEED, dataset="classification", split="StratifiedKFold(3, shuffle, rs=42)")
            st.toast("سُجّلت.", icon=":material/check:")
st.code('''import optuna
def objective(trial):
    params = {"learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.5, log=True),
              "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 64)}
    scores = []
    for step, (tr, va) in enumerate(folds):
        m = HistGradientBoostingClassifier(**params).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        trial.report(np.mean(scores), step)            # enables pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(scores)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
study.best_params''', language="python")
why("ضع الـPipeline كاملًا داخل objective وقيّم بـCV.", "وإلا تسربت المعالجة أو ضُبطت المعاملات على تقسيم واحد متقلب.")
intuition("TPE يقسم المحاولات إلى «جيدة» و«سيئة» ويقترح نقاطًا تعظّم نسبة كثافتيهما l(x)/g(x)؛ رخيص ويتعامل مع الفضاءات الشرطية.")

if at_least("advanced"):
    st.markdown("## متقدم: متى أي Sampler؟")
    comparison_table([
        {"الموقف": "ميزانية صغيرة (< 100)، فضاء مختلط/شرطي", "الاختيار": "TPE (الافتراضي)"},
        {"الموقف": "فضاء مستمر منخفض الأبعاد ومكلف جدًا", "الاختيار": "GPSampler"},
        {"الموقف": "فضاء مستمر وميزانية كبيرة", "الاختيار": "CMA-ES"},
        {"الموقف": "خط أساس / تغطية منتظمة", "الاختيار": "Random / QMC"},
        {"الموقف": "تدريب تكراري طويل", "الاختيار": "أي Sampler + Hyperband/Median pruner"},
    ])
if at_least("research"):
    researcher_note(["Optuna 5.0 (سبتمبر 2026): TPE متعدد المتغيرات افتراضيًا، TPE بدل NSGA-II لمتعدد الأهداف، وPedAnova للأهمية.",
                     "Akiba et al. (2019) قدّموا مفهوم define-by-run للفضاءات الشرطية الديناميكية.",
                     "في المقارنات المنشورة: ثبّت sampler seed وأبلغ عن عدد المحاولات والتقليم."])
    st.markdown(cite("akiba2019", "bergstra2012", "li2018"))
mistakes(["تقييم كل محاولة على تقسيم واحد.", "مقارنة أفضل قيمة Optuna بأداء نموذج آخر غير مضبوط.",
          "الإبلاغ عن best_value كأداء متوقع.", "فضاءات خطية لمعاملات لوغاريتمية."])
page_footer("bayesian_optimization",
            takeaways=["التحسين البايزي يوازن الاستكشاف والاستغلال بنموذج بديل.", "Optuna: Study/Trial/suggest_*/Sampler/Pruner.",
                       "القيمة الفضلى متفائلة؛ قيّم على بيانات معزولة."])
