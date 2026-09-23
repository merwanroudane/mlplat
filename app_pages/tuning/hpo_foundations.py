import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import loguniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (required import for Halving*SearchCV)
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.animation import stepper
from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import plot
from utils.tuning import grid_vs_random, successive_halving_trace

page_header("hpo_foundations")

st.markdown("## المفردات")
comparison_table([
    {"المصطلح": "Search space", "المعنى": "المجال المسموح لكل معامل وتوزيعه (خطي/لوغاريتمي/فئوي/شرطي)"},
    {"المصطلح": "Trial", "المعنى": "تقييم واحد لتركيبة معاملات"},
    {"المصطلح": "Objective", "المعنى": "الدرجة المراد تعظيمها (متوسط CV) — دالة مكلفة وضجيجية"},
    {"المصطلح": "Budget", "المعنى": "عدد المحاولات، الزمن، أو الموارد (أشجار، Epochs، عينات)"},
    {"المصطلح": "Fidelity", "المعنى": "تقييم رخيص تقريبي (بيانات أقل، أشجار أقل) يُستخدم في Halving/Hyperband"},
])
definition("HPO كمسألة تحسين", "$\\lambda^* = \\arg\\max_{\\lambda\\in\\Lambda} \\widehat{\\text{CV}}(\\lambda)$ — دالة صندوق أسود بلا "
           "تدرج، مكلفة التقييم، وضجيجية؛ لذلك نحتاج استراتيجيات بحث ذكية.")
why("استخدم مقياسًا لوغاريتميًا لمعاملات مثل C وalpha وlearning_rate وgamma.",
    "الفرق بين 0.001 و0.01 مهم بقدر الفرق بين 1 و10؛ شبكة خطية تهدر معظم المحاولات في المنطقة العليا.")

st.markdown("## Grid مقابل Random (Bergstra & Bengio, 2012)")
n_trials = st.select_slider("الميزانية (عدد المحاولات)", [9, 16, 25, 36, 49], value=16, key="hpo_n")
pts = grid_vs_random(None, n_trials)


def f_imp(a):  # objective depends almost only on the first hyperparameter
    return np.exp(-((a - 0.73) ** 2) / 0.004)


c1, c2 = st.columns(2)
for col, name in ((c1, "grid"), (c2, "random")):
    P = pts[name]
    best = f_imp(P[:, 0]).max()
    with col:
        fig = go.Figure(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", marker=dict(size=10, color=PALETTE["sky" if name == "grid" else "coral"])))
        xs = np.linspace(0, 1, 200)
        fig.add_trace(go.Scatter(x=xs, y=-0.12 + 0.1 * f_imp(xs), mode="lines", line=dict(color=PALETTE["purple"], width=2.5)))
        fig.update_layout(title=f"{name.title()} search: {len(np.unique(np.round(P[:, 0], 6)))} distinct values of the important "
                                f"parameter · best objective {best:.2f}", height=360, showlegend=False,
                          xaxis_title="important hyperparameter", yaxis_title="unimportant hyperparameter", yaxis=dict(range=[-0.15, 1.05]))
        plot(fig)
intuition("حين يكون معامل واحد فقط مهمًا (وهو الغالب)، تختبر الشبكة √n قيمة مختلفة فقط له، بينما Random يختبر n قيمة. "
          "لنفس الميزانية، Random يستكشف البعد المهم أفضل بكثير.")

st.markdown("## Successive Halving: Animation")
st.caption("27 إعدادًا بميزانية صغيرة؛ نبقي الثلث الأفضل ونضاعف الميزانية ×3، حتى يبقى الأفضل. التقييمات المبكرة ضجيجية (ميزانية قليلة).")
rungs = successive_halving_trace(27, 3)


def _rung(i: int) -> None:
    r = rungs[i]
    fig = go.Figure()
    order = np.argsort(-r["scores"])
    keep = set(r["configs"][order[: max(1, len(r["configs"]) // 3)]]) if i < len(rungs) - 1 else set(r["configs"])
    fig.add_trace(go.Bar(x=[f"cfg {c}" for c in r["configs"]], y=r["scores"],
                         marker_color=[PALETTE["teal"] if c in keep else "#CED4DA" for c in r["configs"]], name="observed score"))
    fig.add_trace(go.Scatter(x=[f"cfg {c}" for c in r["configs"]], y=r["quality"], mode="markers", name="true quality",
                             marker=dict(symbol="diamond", color=PALETTE["purple"], size=9)))
    fig.update_layout(title=f"Rung {i + 1}: resource = {r['resource']} · {len(r['configs'])} configurations (green = promoted)",
                      height=360, legend=dict(orientation="h", y=1.15))
    plot(fig)


stepper("hpo_sh", len(rungs), _rung, labels=[f"rung {i + 1}" for i in range(len(rungs))])
st.markdown("**Hyperband** يشغّل عدة أقواس Successive halving بمقايضات مختلفة بين عدد الإعدادات والميزانية الأولية، ليحمي من "
            "حالة «الإعداد الجيد يبدأ بطيئًا» (Li et al., 2018).")

st.markdown("## Search Strategy Lab على بيانات حقيقية")
X, y = xy("classification")
budget = st.select_slider("ميزانية (تركيبات)", [9, 16, 25], value=16, key="hpo_budget")


@st.cache_data(show_spinner="يشغّل Grid وRandom وHalving بنفس الميزانية…", max_entries=8)
def _compare(budget: int):
    import time
    pipe = make_pipeline(StandardScaler(), SVC())
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    k = int(np.sqrt(budget))
    rows = []
    t0 = time.perf_counter()
    g = GridSearchCV(pipe, {"svc__C": np.logspace(-2, 3, k), "svc__gamma": np.logspace(-3, 1, k)}, cv=cv).fit(X, y)
    rows.append({"strategy": "GridSearchCV", "best CV": g.best_score_, "trials": len(g.cv_results_["params"]),
                 "seconds": time.perf_counter() - t0, "best params": str({k_: round(v, 4) for k_, v in g.best_params_.items()})})
    dist = {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-3, 1e1)}
    t0 = time.perf_counter()
    r = RandomizedSearchCV(pipe, dist, n_iter=budget, cv=cv, random_state=0).fit(X, y)
    rows.append({"strategy": "RandomizedSearchCV", "best CV": r.best_score_, "trials": budget, "seconds": time.perf_counter() - t0,
                 "best params": str({k_: round(v, 4) for k_, v in r.best_params_.items()})})
    t0 = time.perf_counter()
    h = HalvingRandomSearchCV(pipe, dist, n_candidates=budget * 3, factor=3, resource="n_samples", min_resources=60, cv=cv,
                              random_state=0).fit(X, y)
    rows.append({"strategy": "HalvingRandomSearchCV (3× candidates)", "best CV": h.best_score_, "trials": len(h.cv_results_["params"]),
                 "seconds": time.perf_counter() - t0, "best params": str({k_: round(v, 4) for k_, v in h.best_params_.items()})})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="hpo_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["hpo_done"] = True
if st.session_state.get("hpo_done"):
    res = _compare(budget)
    st.dataframe(res.round(4), hide_index=True, width="stretch")
    st.caption("Halving يجرب مرشحين أكثر بنفس الزمن تقريبًا لأنه يقيّم معظمهم على عينات صغيرة. لاحظ: «best CV» متفائل في الثلاثة "
               "(انظر Nested CV).")
    if st.button("سجّل", key="hpo_log", icon=":material/history:", type="tertiary"):
        for _, r in res.iterrows():
            log_experiment("Search Strategy Lab", f"SVC via {r['strategy']}", {"budget": budget}, {"best_cv": r["best CV"]},
                           seed=0, dataset="classification", split="StratifiedKFold(3)")
        st.toast("سُجّلت.", icon=":material/check:")
st.caption("ملاحظة API: HalvingGridSearchCV/HalvingRandomSearchCV ما زالا تجريبيين ويتطلبان "
           "`from sklearn.experimental import enable_halving_search_cv` (متحقَّق منه في 1.9.1).")

st.markdown("## تسرب HPO وNested CV")
why("قيّم «الإجراء» (البحث + النموذج) بـNested CV أو Test معزول.",
    "كل محاولة إضافية فرصة لاختيار ضجيج التحقق؛ best_score_ يزداد تفاؤلًا مع الميزانية.")
page_link("cross_validation", "Nested CV في وحدة التحقق المتقاطع", ":material/view_week:")
page_link("bayesian_optimization", "التالي: التحسين البايزي وOptuna", ":material/auto_graph:")

if at_least("advanced"):
    st.markdown("## متقدم: فضاءات شرطية ومتعددة الأهداف")
    st.markdown("- **شرطي:** degree يهم فقط إن كان kernel='poly' ⇒ قائمة من القواميس في GridSearchCV، أو `if` داخل Optuna objective.\n"
                "- **متعدد الأهداف:** دقة مقابل زمن/حجم ⇒ جبهة Pareto (Optuna `directions=[...]`).\n"
                "- **ميزانية زمنية:** `timeout` في Optuna أو `n_iter` محدود.")
if at_least("research"):
    researcher_note(["أبلغ عن فضاء البحث والميزانية والاستراتيجية لكل النماذج المقارنة؛ ضبط غير متكافئ يضخم فروقًا وهمية.",
                     "Hyperband (Li et al., 2018) وBOHB يجمعان Multi-fidelity مع النمذجة البديلة."])
    st.markdown(cite("bergstra2012", "li2018", "varma2006"))
mistakes(["شبكة خطية لمعاملات لوغاريتمية.", "ميزانية كبيرة لنموذج وصغيرة لآخر ثم المقارنة.", "الإبلاغ عن best_score_ كأداء."])
page_footer("hpo_foundations",
            takeaways=["Random يتفوق على Grid لنفس الميزانية حين تكون معاملات قليلة مهمة.", "Halving/Hyperband يوزعان الميزانية بذكاء.",
                       "HPO نفسه يسبب تفاؤلًا: Nested CV."])
