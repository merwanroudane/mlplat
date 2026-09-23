import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils import causal as C
from utils.plotting import bars, interval_plot, plot

page_header("iivm")

definition("المجموعات الأربع (Angrist, Imbens & Rubin, 1996)", "مع أداة ثنائية Z (مثل «دعوة للبرنامج») ومعالجة ثنائية D: "
           "**Compliers** يعالَجون فقط إن دُعوا؛ **Always-takers** دائمًا؛ **Never-takers** أبدًا؛ **Defiers** عكس الدعوة.")
comparison_table([
    {"النوع": "Complier", "D(Z=0)": 0, "D(Z=1)": 1, "دور": "الوحيدون الذين تكشف الأداة أثرهم"},
    {"النوع": "Always-taker", "D(Z=0)": 1, "D(Z=1)": 1, "دور": "لا تغيّرهم الأداة"},
    {"النوع": "Never-taker", "D(Z=0)": 0, "D(Z=1)": 0, "دور": "لا تغيّرهم الأداة"},
    {"النوع": "Defier", "D(Z=0)": 1, "D(Z=1)": 0, "دور": "يُستبعدون بافتراض الرتابة"},
])
formula(r"\text{LATE} = \frac{\mathbb E[Y\mid Z=1]-\mathbb E[Y\mid Z=0]}{\mathbb E[D\mid Z=1]-\mathbb E[D\mid Z=0]} = \mathbb E[Y(1)-Y(0)\mid \text{complier}]",
        title="Wald / LATE (Imbens & Angrist, 1994)",
        intuition="أثر الدعوة على النتيجة (ITT) مقسومًا على أثرها على المشاركة: الأثر على من غيّرت الدعوة سلوكهم فقط.")
comparison_table([
    {"الافتراض": "Independence (| X)", "المعنى": "Z عشوائي كأنه (بمعلومية X)"},
    {"الافتراض": "Exclusion", "المعنى": "Z يؤثر في Y عبر D فقط"},
    {"الافتراض": "Monotonicity", "المعنى": "لا Defiers"},
    {"الافتراض": "Relevance", "المعنى": "P(complier) > 0"},
])
formula(r"\theta = \frac{\mathbb E\big[g(1,X)-g(0,X)+\frac{Z(Y-g(1,X))}{m(X)}-\frac{(1-Z)(Y-g(0,X))}{1-m(X)}\big]}"
        r"{\mathbb E\big[r(1,X)-r(0,X)+\frac{Z(D-r(1,X))}{m(X)}-\frac{(1-Z)(D-r(0,X))}{1-m(X)}\big]}",
        title="IIVM: ratio of two doubly robust ITT scores (DoubleMLIIVM: ml_g, ml_m, ml_r)",
        symbols={"g(z, X)": "E[Y | Z=z, X]", "r(z, X)": "E[D | Z=z, X]", "m(X)": "P(Z=1 | X)"})

st.markdown("## LATE Lab")
c1, c2 = st.columns(2)
share = c1.slider("نسبة Compliers", 0.1, 0.9, 0.6, 0.05, key="iivm_share")
n = c2.select_slider("n", [1000, 2000, 4000], value=2000, key="iivm_n")


@st.cache_data(show_spinner="يقدّر…", max_entries=16)
def _run(share: float, n: int):
    X, y, d, z, typ = C.make_iivm(n=n, late=1.0, share_compliers=share, seed=4)
    g = RandomForestRegressor(150, min_samples_leaf=10, random_state=0, n_jobs=1)
    clf = RandomForestClassifier(150, min_samples_leaf=10, random_state=0, n_jobs=1)
    late = C.dml_iivm(X, y, d, z, g, clf, clf)
    naive = y[d == 1].mean() - y[d == 0].mean()
    itt = y[z == 1].mean() - y[z == 0].mean()
    fs = d[z == 1].mean() - d[z == 0].mean()
    irm = C.dml_irm(X, y, d, g, clf, n_folds=5)
    counts = pd.Series(typ).value_counts()
    return late, naive, itt, fs, irm, counts


late, naive, itt, fs, irm, counts = _run(share, n)
c1, c2 = st.columns([1.5, 1])
with c1:
    plot(interval_plot(["Naive treated vs untreated", "DML-IRM (assumes D unconfounded)", "Wald (no covariates)", "DML-IIVM (LATE)"],
                       [naive, irm.theta, itt / fs, late.theta], [np.nan, irm.ci[0], np.nan, late.ci[0]],
                       [np.nan, irm.ci[1], np.nan, late.ci[1]], truth=1.0, title="True complier effect = 1.0"), height=340)
with c2:
    plot(bars(counts.index, counts.values, title="Latent types (unobservable in practice)", text_fmt=".0f"), height=340)
st.markdown(f"ITT = {itt:.3f}، المرحلة الأولى (نسبة Compliers المقدرة) = {fs:.3f} ⇒ Wald = {itt / fs:.3f}. "
            f"DML-IIVM = **{late.theta:.3f}** (SE {late.se:.3f}).")
intuition("المقارنة الساذجة ومقدِّر IRM متحيزان لأن Always-takers لديهم نتائج أعلى أصلًا (اختيار ذاتي). الأداة العشوائية تتجاوز ذلك "
          "— لكنها تكشف أثر Compliers فقط، لا أثر Always-takers (الذي هو 2.0 في هذا DGP).")
why("صرّح بأن LATE يخص Compliers.", "قد يختلف أثرهم عن أثر من يُعالَج دائمًا أو لا يُعالَج أبدًا؛ التعميم يحتاج افتراضات إضافية.")
st.code("""data = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols, z_cols="z")
iivm = dml.DoubleMLIIVM(data, ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(), ml_r=RandomForestClassifier(),
                        score="LATE", subgroups={"always_takers": True, "never_takers": True})
iivm.fit(); iivm.summary""", language="python")
st.caption("متحقَّق من التوقيع في DoubleML 0.11.4: score الافتراضي 'LATE'، و`subgroups` يسمح بافتراض غياب Always/Never-takers.")

if at_least("research"):
    researcher_note(["Imbens & Angrist (1994) عرّفا LATE؛ Angrist, Imbens & Rubin (1996) ربطاه بالنتائج الكامنة.",
                     "مع مقام صغير (Compliers قليلون) تتصرف IIVM كأداة ضعيفة: فترات عريضة وتقريب طبيعي ضعيف."])
    st.markdown(cite("imbens1994", "angrist1996", "chernozhukov2018"))
mistakes(["تعميم LATE على كل المجتمع.", "تجاهل الرتابة.", "أداة تؤثر في Y مباشرة (مثل رسالة تحفيزية تغيّر السلوك دون المشاركة)."])
page_footer("iivm",
            takeaways=["أداة ثنائية تكشف أثر Compliers: LATE.", "LATE = ITT / المرحلة الأولى؛ IIVM نسخته مزدوجة المتانة.",
                       "الرتابة والاستبعاد افتراضات حاسمة."])
