import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from components.callouts import causal_caution, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils import causal as C
from utils.datasets import load_dataset
from utils.plotting import interval_plot, plot

page_header("hte")

formula(r"\tau(x) = \mathbb E[Y(1)-Y(0)\mid X=x],\qquad \text{GATE}_g = \mathbb E[\tau(X)\mid G=g],\qquad \text{BLP: } \tau(X)\approx \beta^\top b(X)",
        title="CATE, GATE and the best linear predictor of the CATE",
        symbols={"G": "مجموعة معرّفة مسبقًا (عمر، منطقة...)", "b(X)": "أساسات (ثابت، x1، ...)"},
        intuition="ATE رقم واحد؛ CATE دالة. GATE وBLP ملخصات قابلة للاستدلال الصارم لهذه الدالة.")
comparison_table([
    {"المقدِّر": "S-learner", "الفكرة": "نموذج واحد μ(x, d)؛ τ̂ = μ(x,1) − μ(x,0)", "خطر": "التنظيم قد يتجاهل d فيقلّص τ̂ نحو 0"},
    {"المقدِّر": "T-learner", "الفكرة": "نموذجان منفصلان μ₁، μ₀", "خطر": "فرق خطأين كبيرين؛ ضعيف مع عدم التوازن"},
    {"المقدِّر": "X-learner", "الفكرة": "آثار مقدرة لكل مجموعة ثم ترجيح بالميل", "خطر": "جيد مع عدم التوازن (Künzel et al., 2019)"},
    {"المقدِّر": "R-learner", "الفكرة": "تقليل خسارة Robinson على البواقي", "خطر": "يرث التعامد (Nie & Wager, 2021)"},
    {"المقدِّر": "DR-learner", "الفكرة": "انحدار درجة AIPW (pseudo-outcome) على X", "خطر": "مزدوج المتانة (Kennedy, 2023)"},
    {"المقدِّر": "Causal forest", "الفكرة": "غابة بمعيار تقسيم على تباين الأثر + Honesty", "خطر": "فترات ثقة تقاربية (Wager & Athey, 2018)"},
])

st.markdown("## CATE Lab")
df = load_dataset("causal")
X, y, d, tau = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy(), df["tau"].to_numpy()
st.caption("DGP معروف: τ(x) = 1 + x1 (يزيد الأثر مع x1)، ومعالجة مُربَكة بـx1 وx2 وx3.")


@st.cache_data(show_spinner="DR-learner وT-learner مع Cross-fitting…")
def _meta():
    g = RandomForestRegressor(200, min_samples_leaf=10, random_state=0, n_jobs=1)
    m = RandomForestClassifier(200, min_samples_leaf=10, random_state=0, n_jobs=1)
    final = RandomForestRegressor(200, min_samples_leaf=40, random_state=0, n_jobs=1)
    phi, cate_dr = C.dr_learner_cate(X, y, d, g, m, final, n_folds=5)
    folds = C.fold_indices(len(y), 5, seed=3)
    mu1, mu0 = np.empty(len(y)), np.empty(len(y))
    for tr, te in folds:
        t1, t0 = tr[d[tr] == 1], tr[d[tr] == 0]
        mu1[te] = RandomForestRegressor(200, min_samples_leaf=10, random_state=0, n_jobs=1).fit(X[t1], y[t1]).predict(X[te])
        mu0[te] = RandomForestRegressor(200, min_samples_leaf=10, random_state=0, n_jobs=1).fit(X[t0], y[t0]).predict(X[te])
    return phi, cate_dr, mu1 - mu0


phi, cate_dr, cate_t = _meta()
c1, c2 = st.columns(2)
for col, name, est in ((c1, "DR-learner (out-of-fold)", cate_dr), (c2, "T-learner (out-of-fold)", cate_t)):
    with col:
        fig = go.Figure(go.Scatter(x=tau, y=est, mode="markers", marker=dict(color=PALETTE["purple"], opacity=0.3, size=4)))
        lo, hi = float(tau.min()), float(tau.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PALETTE["teal"], dash="dash")))
        fig.update_layout(title=f"{name}: corr {np.corrcoef(tau, est)[0, 1]:.2f}, RMSE {np.sqrt(np.mean((est - tau) ** 2)):.2f}",
                          xaxis_title="true τ(x)", yaxis_title="estimated τ̂(x)", showlegend=False, height=340)
        plot(fig)

st.markdown("## GATE: آثار المجموعات بفترات ثقة")
bins = st.select_slider("عدد مجموعات x1 (quantiles)", [2, 3, 4, 5], value=4, key="hte_bins")
groups = pd.qcut(df["x1"], bins, labels=[f"x1 Q{i + 1}" for i in range(bins)]).astype(str).to_numpy()
gates = C.gate(phi, groups)
truth_g = [float(tau[groups == g["group"]].mean()) for g in gates]
fig = interval_plot([g["group"] for g in gates], [g["gate"] for g in gates], [g["lo"] for g in gates], [g["hi"] for g in gates],
                    title="GATE by x1 quartile (AIPW pseudo-outcome means) · diamonds = truth")
fig.add_trace(go.Scatter(x=truth_g, y=[g["group"] for g in gates], mode="markers", marker=dict(symbol="diamond-open", size=14,
                                                                                               color="#212529"), name="truth"))
plot(fig)
why("اعرّف المجموعات مسبقًا (قبل رؤية النتائج).", "البحث في مئات المجموعات عن «أقوى أثر» اصطياد؛ GATE بمجموعات مسبقة يعطي "
    "استدلالًا صالحًا بمتوسطات الدرجة المتعامدة.")

st.markdown("## BLP وGATE في DoubleML")
dml = optional("doubleml")
if dml is None:
    missing_notice("doubleml")
else:
    @st.cache_data(show_spinner="DoubleMLIRM + cate + gate…")
    def _dml_blp():
        data = dml.DoubleMLData(df.drop(columns=["tau", "true_ps"]), "y", "d", [f"x{i}" for i in range(1, 6)])
        irm = dml.DoubleMLIRM(data, RandomForestRegressor(150, min_samples_leaf=5, random_state=0, n_jobs=1),
                              RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=1), n_folds=5)
        irm.fit()
        blp = irm.cate(pd.DataFrame({"const": 1.0, "x1": df["x1"]}))
        gate = irm.gate(pd.DataFrame({"x1 < 0": df["x1"] < 0, "x1 ≥ 0": df["x1"] >= 0}))
        return blp.summary, gate.confint()

    blp, gate = _dml_blp()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**BLP: τ(X) ≈ β₀ + β₁·x1** (الحقيقة: 1 + 1·x1)")
        st.dataframe(blp.round(3), width="stretch")
    with c2:
        st.markdown("**GATE** (الحقيقة ≈ 0.2 و1.8)")
        st.dataframe(gate.round(3), width="stretch")
    st.caption("irm.cate(basis) يعيد DoubleMLBLP؛ irm.gate(groups) يقبل DataFrame من مجموعات منطقية — متحقق في DoubleML 0.11.4.")

st.markdown("## Causal Forest (EconML CausalForestDML)")
econml = optional("econml")
if econml is None:
    missing_notice("econml")
else:
    @st.cache_data(show_spinner="CausalForestDML…")
    def _cf():
        from econml.dml import CausalForestDML
        cf = CausalForestDML(model_y=RandomForestRegressor(100, min_samples_leaf=5, random_state=0, n_jobs=1),
                             model_t=RandomForestClassifier(100, min_samples_leaf=5, random_state=0, n_jobs=1),
                             discrete_treatment=True, n_estimators=200, random_state=0, n_jobs=1)
        cf.fit(y, d, X=X)
        grid = np.column_stack([np.linspace(-2.5, 2.5, 50), np.zeros((50, X.shape[1] - 1))])
        lb, ub = cf.effect_interval(grid, alpha=0.05)
        return grid[:, 0], cf.effect(grid), lb, ub, cf.effect(X)

    g1, eff, lb, ub, eff_all = _cf()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[g1, g1[::-1]], y=np.r_[ub, lb[::-1]], fill="toself", fillcolor="rgba(112,72,232,0.15)",
                             line=dict(width=0), name="95% CI"))
    fig.add_trace(go.Scatter(x=g1, y=eff, name="τ̂(x1) causal forest", line=dict(color=PALETTE["purple"], width=3)))
    fig.add_trace(go.Scatter(x=g1, y=1 + g1, name="true τ(x1) = 1 + x1", line=dict(color=PALETTE["teal"], dash="dash")))
    fig.update_layout(title=f"CATE along x1 (other covariates at 0) · corr with truth over sample {np.corrcoef(eff_all, tau)[0, 1]:.2f}",
                      xaxis_title="x1", height=380)
    plot(fig)
    st.caption(f"econml {installed_version('econml')} — CausalForestDML يجمع DML (تحييد Y وT) مع غابة سببية صادقة (Honest=True افتراضيًا).")
causal_caution("CATE تحت عدم الإرباك أيضًا. وتقديرات CATE الفردية صاخبة جدًا؛ لا تتخذ قرارًا فرديًا حاسمًا بناءً على τ̂(xᵢ) واحد دون "
               "تقييم سياسة (Policy evaluation) على بيانات مستقلة.")
intuition("لاحظ أن τ̂ يتسطح عند أطراف x1 (بيانات قليلة + انكماش الغابة): عدم اليقين أكبر حيث تقل البيانات.")

if at_least("research"):
    researcher_note(["Semenova & Chernozhukov (2021): BLP وGATE مع استدلال صالح عبر انحدار الدرجة المتعامدة.",
                     "Kennedy (2023): DR-learner يحقق معدلات Oracle تحت شروط؛ Nie & Wager (2021): R-learner.",
                     "Athey, Tibshirani & Wager (2019): Generalized random forests؛ Wager & Athey (2018): فترات تقاربية."])
    st.markdown(cite("semenova2021", "kennedy2023", "nie2021", "kunzel2019", "wager2018", "athey2019"))
mistakes(["البحث عن مجموعات بعد رؤية النتائج.", "تفسير τ̂(xᵢ) الفردي كحقيقة.", "T-learner مع عدم توازن شديد.",
          "نسيان أن CATE تحت الافتراضات نفسها لـATE."])
page_footer("hte",
            takeaways=["CATE دالة؛ GATE وBLP ملخصاتها القابلة للاستدلال.", "DR-learner وR-learner يرثان التعامد.",
                       "Causal forest يعطي τ̂(x) بفترات؛ المجموعات تُعرَّف مسبقًا."])
