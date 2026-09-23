import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import flow
from components.formulas import formula
from config import RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import SEQUENCE
from utils import causal as C
from utils.datasets import make_plr
from utils.plotting import interval_plot, plot

page_header("dml_core")

st.markdown("## المسألة")
formula(r"Y = D\theta_0 + g_0(X) + \zeta,\quad \mathbb E[\zeta\mid D,X]=0;\qquad D = m_0(X) + V,\quad \mathbb E[V\mid X]=0",
        title="Partially linear model (Robinson, 1988; Chernozhukov et al., 2018)",
        symbols={r"\theta_0": "الأثر السببي المستهدف (عدد واحد)", r"g_0, m_0": "دوال إزعاج (Nuisance) غير معروفة وربما معقدة",
                 "X": "مربكات كثيرة الأبعاد"},
        intuition="نريد θ₀ بدقة وفترة ثقة، بينما g₀ وm₀ مجرد «ضجيج بنيوي» نحتاج إزالته. ML ممتاز في تقدير g₀ وm₀ — لكن "
                  "إدخاله بسذاجة يحيّز θ̂.")

st.markdown("## لماذا يفشل «ML ثم OLS» الساذج؟")
comparison_table([
    {"مصدر التحيز": "Regularization bias", "السبب": "ML ينكمش/ينظّم لتحسين التنبؤ ⇒ خطأ ĝ − g₀ من رتبة أبطأ من 1/√n",
     "الأثر على θ̂": "تحيز لا يختفي بمعدل √n", "العلاج": "Neyman-orthogonal score"},
    {"مصدر التحيز": "Overfitting bias", "السبب": "استخدام الملاحظة نفسها لتقدير الإزعاج وللدرجة ⇒ ارتباط أخطائهما",
     "الأثر على θ̂": "تحيز من الإفراط", "العلاج": "Sample splitting / Cross-fitting"},
])
definition("Double/Debiased ML", "(1) قدّر دوال الإزعاج بـML؛ (2) ابنِ درجة متعامدة (غير حساسة لأخطاء الإزعاج من الرتبة الأولى)؛ "
           "(3) استخدم تقسيم العينة؛ (4) Cross-fitting لاستعادة الكفاءة؛ (5) حل معادلة العزم لـθ؛ (6) أخطاء معيارية وفترات ثقة "
           "من تباين الدرجة — صالحة تحت افتراضات معدلات التقارب.")
flow(["1 ML for nuisances ℓ(X)=E[Y|X], m(X)=E[D|X]", "2 Orthogonal score (residualise)", "3 Sample splitting",
      "4 Cross-fitting", "5 Solve for θ̂", "6 SE & CI"], direction="LR")
formula(r"\hat\theta = \frac{\frac1n\sum_i \hat V_i\,\hat U_i}{\frac1n\sum_i \hat V_i^2},\qquad \hat U_i = Y_i - \hat\ell(X_i),\ \hat V_i = D_i - \hat m(X_i)",
        title="DML for PLR (partialling out)",
        intuition="نظّف Y وD من كل ما يفسّره X (بـML وخارج الطية)، ثم انحدر البواقي على البواقي — FWL بـML.")

st.markdown("## Naive vs DML Lab")
st.caption("نكرر التجربة على عينات مستقلة من DGP معروف (θ₀ = 0.5) ونقارن توزيع التقديرات — على نمط الشكل 1 في Chernozhukov et al. (2018).")
c1, c2, c3, c4 = st.columns(4)
reps = c1.select_slider("عدد التكرارات", [20, 50, 100], value=20, key="dc_reps")
n = c2.select_slider("n", [300, 500, 1000], value=500, key="dc_n")
nonlin = c3.slider("اللاخطية", 0.0, 1.0, 1.0, 0.25, key="dc_nl")
leaf = c4.segmented_control("متعلم الإزعاج", ["deep (leaf=1)", "regularised (leaf=5)"], default="deep (leaf=1)", key="dc_leaf",
                            required=True)


@st.cache_data(show_spinner="يشغّل المحاكاة (قد يستغرق عدة ثوانٍ)…", max_entries=12)
def _sim(reps: int, n: int, nonlin: float, leaf: str):
    rf = RandomForestRegressor(n_estimators=60, max_features=0.5, min_samples_leaf=1 if leaf.startswith("deep") else 5,
                               random_state=0, n_jobs=1)
    out = {"Naive ML plug-in": [], "Orthogonal, no cross-fitting": [], "DML (cross-fitted)": []}
    for r in range(reps):
        df = make_plr(n=n, p=10, nonlinearity=nonlin, seed=RANDOM_SEED + r)
        X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
        for name, res in (("Naive ML plug-in", C.naive_plugin(X, y, d, rf, n_iter=3)),
                          ("Orthogonal, no cross-fitting", C.plr_no_crossfit(X, y, d, rf, rf)),
                          ("DML (cross-fitted)", C.dml_plr(X, y, d, rf, rf, n_folds=2, seed=r))):
            out[name].append((res.theta, res.se))
    return {k: np.array(v) for k, v in out.items()}


if st.button("شغّل المحاكاة", key="dc_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["dc_args"] = (reps, n, nonlin, leaf)
if st.session_state.get("dc_args"):
    res = _sim(*st.session_state["dc_args"])
    fig = go.Figure()
    for i, (name, v) in enumerate(res.items()):
        fig.add_trace(go.Histogram(x=v[:, 0], name=name, opacity=0.55, nbinsx=25, marker_color=SEQUENCE[i]))
    fig.add_vline(x=0.5, line=dict(color="#212529", dash="dash", width=2), annotation_text="θ₀ = 0.5")
    fig.update_layout(barmode="overlay", title="Sampling distributions of θ̂", xaxis_title="θ̂", height=380)
    plot(fig)
    rows = []
    for k, v in res.items():
        th, se = v[:, 0], v[:, 1]
        cov = np.mean(np.abs(th - 0.5) <= 1.96 * se) if np.isfinite(se).all() else np.nan
        rows.append({"estimator": k, "mean θ̂": th.mean(), "bias": th.mean() - 0.5, "SD across reps": th.std(ddof=1),
                     "mean reported SE": np.nanmean(se) if np.isfinite(se).any() else np.nan, "95% CI coverage": cov,
                     "RMSE": np.sqrt(np.mean((th - 0.5) ** 2))})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.markdown("اقرأ الجدول بعمودين: **التحيز** و**التغطية**. المقدِّر الساذج (يُدخل ĝ دون تحييد D) متحيز لأن التنظيم يسرّب جزءًا من "
                "أثر D إلى ĝ، ولا خطأ معياري صالح له. الدرجة المتعامدة دون Cross-fitting: مع متعلم عميق يفرط في الملاءمة تصبح "
                "البواقي داخل العينة صغيرة اصطناعيًا، فيظهر تحيز وتختل الأخطاء المعيارية (قارن «mean reported SE» بـ«SD across "
                "reps»)؛ مع متعلم منظَّم قد يكون الأثر صغيرًا — وهذا بالضبط لماذا لا نراهن عليه. DML يتمركز حول θ₀ بتغطية قريبة من "
                "الاسمية. مع 20 تكرارًا فقط التغطية تقريبية؛ زد التكرارات أو استخدم مختبر Monte Carlo.")
else:
    st.info("اضغط «شغّل المحاكاة». كل تكرار يدرّب عدة غابات عشوائية؛ الحدود مقيدة لتناسب النشر العام.", icon=":material/info:")

st.markdown("## تقدير واحد مع فترة ثقة")
df1 = make_plr(n=1000, nonlinearity=nonlin, seed=11)
X1, y1, d1 = df1.filter(like="x").to_numpy(), df1["y"].to_numpy(), df1["d"].to_numpy()


@st.cache_data(show_spinner="يقدّر…", max_entries=8)
def _one(nonlin: float):
    rf = RandomForestRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=5, random_state=0, n_jobs=1)
    ests = [C.naive_ols(None, y1, d1), C.naive_ols(X1, y1, d1), C.dml_plr(X1, y1, d1, rf, rf, n_folds=5)]
    return [(e.name, e.theta, *e.ci) for e in ests]


ests = _one(nonlin)
plot(interval_plot([e[0] for e in ests], [e[1] for e in ests], [e[2] for e in ests], [e[3] for e in ests], truth=0.5,
                   title="95% confidence intervals (n = 1000)"))
why("استخدم DML حين يكون الإرباك غير خطي أو عالي الأبعاد.", "OLS بضوابط خطية متحيز إن كان g₀ غير خطي؛ DML يستخدم ML مرنًا للإزعاج "
    "ويبقي الاستدلال على θ صالحًا.")
intuition("DML لا يحتاج أن تكون ĝ وm̂ مثاليتين؛ يكفي أن يكون خطؤهما صغيرًا بما يكفي (معدل أسرع من n^{-1/4}) لأن الدرجة "
          "المتعامدة تجعل تأثيرهما من الرتبة الثانية (حاصل ضرب الخطأين).")
page_link("neyman_orthogonality", "التالي: تعامد Neyman", ":material/architecture:")

if at_least("advanced"):
    st.markdown("## متقدم: شرط المعدل")
    st.latex(r"\sqrt n(\hat\theta-\theta_0) \Rightarrow N(0,\sigma^2)\quad\text{if}\quad \|\hat m - m_0\|_2\cdot\|\hat\ell-\ell_0\|_2 = o_P(n^{-1/2})")
    st.markdown("يتحقق مثلًا إن تقارب كل منهما بمعدل o(n^{-1/4}) — أبطأ بكثير من √n، ويحققه Lasso/الغابات/التعزيز تحت شروط ندرة/نعومة.")
if at_least("research"):
    researcher_note(["Chernozhukov et al. (2018) هي المرجع الأساسي؛ Robinson (1988) للأصل شبه المعلمي؛ Belloni et al. (2014) "
                     "للحالة الخطية المتفرقة.",
                     "DML ليس خوارزمية تنبؤ؛ إنه إطار استدلال لمعلمات منخفضة الأبعاد مع إزعاج عالي الأبعاد."])
    st.markdown(cite("chernozhukov2018", "robinson1988", "belloni2014"))
mistakes(["«ML ثم OLS» بإدخال تنبؤ ĝ كمتغير ضبط.", "إزعاج بتنبؤات داخل العينة.", "اعتبار DML حلًا للمربكات غير المقاسة.",
          "الحكم على صحة θ̂ بجودة تنبؤ الإزعاج وحدها."])
page_footer("dml_core",
            takeaways=["التحيز الساذج له مصدران: التنظيم والإفراط.", "العلاج: درجة متعامدة + Cross-fitting.",
                       "النتيجة: θ̂ طبيعي تقاربيًا بفترة ثقة صالحة تحت عدم الإرباك وشروط المعدل."])
