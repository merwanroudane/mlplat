import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import optional
from core.state import at_least
from utils import causal as C
from utils.plotting import interval_plot, plot

page_header("pliv")

st.markdown("## قبل PLIV: الداخلية والأداة")
definition("Endogeneity", "D مرتبط بالخطأ في معادلة Y (مربك غير مقاس U، خطأ قياس، سببية عكسية) ⇒ أي انحدار — بما فيه DML-PLR — متحيز.")
definition("Instrument Z", "متغير (1) **ذو صلة**: يحرك D؛ (2) **مستبعد**: لا يؤثر في Y إلا عبر D؛ (3) **خارجي**: غير مرتبط بـU (بمعلومية X).")
mermaid("""
flowchart LR
  Z[Z instrument] --> D[D treatment]
  U((U unobserved)) --> D
  U --> Y[Y outcome]
  D -->|θ| Y
  X[X controls] --> Z & D & Y
  style U fill:#FFF4E6,stroke:#F76707,stroke-dasharray: 5 5
""")
comparison_table([
    {"الشرط": "Relevance", "قابل للاختبار؟": "نعم (First stage)", "الفشل": "أداة ضعيفة ⇒ تحيز نحو OLS وفترات مضللة"},
    {"الشرط": "Exclusion", "قابل للاختبار؟": "لا (حجة)", "الفشل": "تحيز لا يزول بزيادة n"},
    {"الشرط": "Exogeneity | X", "قابل للاختبار؟": "لا (حجة)", "الفشل": "تحيز"},
])

st.markdown("## PLIV")
formula(r"Y = D\theta_0 + g_0(X) + \zeta,\ \ \mathbb E[\zeta\mid Z,X]=0;\qquad Z = m_0(X) + V",
        title="Partially linear IV model",
        intuition="نحيّد Y وD وZ من X بـML، ثم نستخدم بواقي Z كأداة لبواقي D.")
formula(r"\hat\theta = \frac{\sum_i \tilde Z_i\tilde Y_i}{\sum_i \tilde Z_i\tilde D_i},\quad \tilde Y = Y-\hat\ell(X),\ \tilde D = D-\hat r(X),\ \tilde Z = Z-\hat m(X)",
        title="PLIV partialling-out estimator (DoubleMLPLIV: ml_l, ml_m, ml_r)")

st.markdown("## Weak Instrument Lab")
c1, c2 = st.columns(2)
strength = c1.select_slider("قوة الأداة (أثر Z على D)", [0.02, 0.05, 0.1, 0.25, 0.5, 1.0], value=1.0, key="pliv_s")
n = c2.select_slider("n", [500, 1000, 2000, 4000], value=2000, key="pliv_n")


@st.cache_data(show_spinner="Cross-fitting لثلاثة إزعاجات…", max_entries=24)
def _run(strength: float, n: int):
    X, y, d, z = C.make_pliv(n=n, strength=strength, seed=2)
    rf = RandomForestRegressor(100, min_samples_leaf=5, max_features=0.6, random_state=0, n_jobs=1)
    pliv = C.dml_pliv(X, y, d, z, rf, rf, rf)
    plr = C.dml_plr(X, y, d, rf, rf)
    ols = C.naive_ols(X, y, d)
    return [ols, plr, pliv]


ests = _run(strength, n)
names = ["OLS with X", "DML-PLR (ignores endogeneity)", "DML-PLIV (uses Z)"]
plot(interval_plot(names, [e.theta for e in ests], [e.ci[0] for e in ests], [e.ci[1] for e in ests], truth=0.5,
                   title="Endogenous D: only IV targets θ₀ = 0.5"))
fs = ests[2].extra["first_stage_corr"]
c1, c2, c3 = st.columns(3)
c1.metric("θ̂ PLIV", f"{ests[2].theta:.3f}")
c2.metric("SE PLIV", f"{ests[2].se:.3f}")
c3.metric("corr(Z̃, D̃) first stage", f"{fs:.3f}")
if abs(fs) < 0.1:
    warning("أداة ضعيفة: الارتباط بين بواقي Z وD صغير جدًا. التقدير متقلب والتوزيع الطبيعي التقاربي غير موثوق؛ الفترات المعتادة "
            "مضللة. استخدم فترات قوية أمام الضعف (Anderson–Rubin) أو ابحث عن أداة أقوى.")
intuition("مع أداة قوية يتمركز PLIV حول 0.5 بينما يبقى PLR وOLS متحيزين للأعلى (U يرفع D وY معًا). مع أداة ضعيفة ينفجر "
          "تباين PLIV — لا مجال لـ«أداة قليلة القوة» مجانية.")
why("تحقق من قوة المرحلة الأولى وقدّم حجة للاستبعاد.", "الاستبعاد غير قابل للاختبار؛ PLIV يحيّد X بمرونة لكنه لا يصلح أداة غير صالحة.")

dml = optional("doubleml")
st.code("""data = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols, z_cols="z")
pliv = dml.DoubleMLPLIV(data, ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor(), ml_r=RandomForestRegressor())
pliv.fit(); pliv.summary""", language="python")
st.caption("متحقَّق من التوقيع في DoubleML 0.11.4: DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, ml_g=None, n_folds=5, n_rep=1, "
           "score='partialling out').")

if at_least("advanced"):
    st.markdown("## متقدم: الأثر المحلي")
    st.markdown("مع أثر غير متجانس، IV يقدّر أثرًا لمن تحرّكهم الأداة (Compliers) — انظر IIVM وLATE. PLIV يفترض θ₀ ثابتًا.")
if at_least("research"):
    researcher_note(["Chernozhukov et al. (2018) يقدمون PLIV كمثال أساسي؛ ومع أدوات متعددة/عالية الأبعاد يمكن اختيارها بـLasso.",
                     "Angrist, Imbens & Rubin (1996) للإطار السببي لـIV."])
    st.markdown(cite("chernozhukov2018", "angrist1996"))
mistakes(["أداة تؤثر في Y مباشرة.", "تجاهل ضعف المرحلة الأولى.", "استخدام PLR مع داخلية واضحة."])
page_footer("pliv",
            takeaways=["الداخلية تحيّز كل انحدار؛ الأداة الصالحة تحلها.", "PLIV يحيّد Y وD وZ من X بـML ثم يستخدم Z̃ أداة.",
                       "الأداة الضعيفة = تقدير غير موثوق."])
