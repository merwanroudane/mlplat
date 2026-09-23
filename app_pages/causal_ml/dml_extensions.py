import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset
from utils.plotting import plot

page_header("dml_extensions")

st.markdown("## خريطة الامتدادات (مدعومة ومتحقَّق منها في DoubleML 0.11.4)")
comparison_table([
    {"الامتداد": "CATE / GATE / BLP", "الأداة": "irm.cate(basis)، irm.gate(groups)، DoubleMLPolicyTree", "المرجع": "Semenova & Chernozhukov (2021)",
     "الوحدة": "HTE"},
    {"الامتداد": "Cluster-robust inference", "الأداة": "DoubleMLData(..., cluster_cols=...) / DoubleMLClusterData", "المرجع": "Chiang et al. (2022)",
     "الوحدة": "هنا"},
    {"الامتداد": "Sensitivity analysis (omitted confounders)", "الأداة": "sensitivity_analysis(cf_y, cf_d, rho)", "المرجع": "Chernozhukov et al. (2022, Long story short)",
     "الوحدة": "هنا"},
    {"الامتداد": "Multiple treatments", "الأداة": "d_cols=[...] + bootstrap() + confint(joint=True)", "المرجع": "Chernozhukov et al. (2018)",
     "الوحدة": "هنا"},
    {"الامتداد": "Continuous treatment", "الأداة": "PLR (θ خطي)؛ Multi-valued: DoubleMLAPOS", "المرجع": "Chernozhukov et al. (2018)",
     "الوحدة": "PLR"},
    {"الامتداد": "Quantile / potential quantiles", "الأداة": "DoubleMLPQ، DoubleMLQTE، DoubleMLLPQ، DoubleMLCVAR", "المرجع": "Belloni et al. (2017)",
     "الوحدة": "هنا"},
    {"الامتداد": "Sample selection", "الأداة": "DoubleMLSSM", "المرجع": "توثيق DoubleML", "الوحدة": "هنا"},
    {"الامتداد": "Panel / DiD", "الأداة": "DoubleMLDIDMulti، DoubleMLPLPR", "المرجع": "Callaway & Sant'Anna (2021)؛ Chang (2020)", "الوحدة": "Panel DiD"},
    {"الامتداد": "RDD", "الأداة": "doubleml.rdd (DoubleMLRDDData)", "المرجع": "توثيق DoubleML", "الوحدة": "إشارة"},
])
warning("لا نستخدم عبارة «Dynamic DML» دون مرجع صريح. الأثر عبر الزمن في هذه المنصة يُقدَّم عبر إطار موثق: ATT(g, t) وتجميع "
        "Event study (Callaway & Sant'Anna, 2021) في وحدة Panel DiD.")

st.markdown("## تحليل الحساسية للمربكات غير المقاسة")
formula(r"|\hat\theta - \theta| \lesssim \rho\,\sqrt{\frac{C_Y^2\,C_D^2}{1-C_D^2}}\cdot S",
        title="Omitted-variable bias bound (Chernozhukov et al., 2022 — schematic form)",
        symbols={"C_Y (cf_y)": "نسبة التباين المتبقي في Y التي يفسرها مربك غير مقاس", "C_D (cf_d)": "القوة المقابلة في معادلة المعالجة",
                 r"\rho": "الارتباط بين أثريه (1 = أسوأ حالة)", "S": "مقياس مستخرج من البيانات"},
        intuition="«كم يجب أن يكون المربك المحذوف قويًا ليلغي النتيجة؟» — Robustness value (RV) يجيب: أقل قوة مشتركة تجعل الحد "
                  "يشمل الصفر.")
dml = optional("doubleml")
if dml is None:
    missing_notice("doubleml")
else:
    c1, c2, c3 = st.columns(3)
    cf_y = c1.slider("cf_y", 0.0, 0.3, 0.03, 0.01, key="ext_cfy")
    cf_d = c2.slider("cf_d", 0.0, 0.3, 0.03, 0.01, key="ext_cfd")
    rho = c3.slider("rho", 0.0, 1.0, 1.0, 0.1, key="ext_rho")

    @st.cache_resource(show_spinner="DoubleMLIRM…")
    def _irm():
        df = load_dataset("causal").drop(columns=["tau", "true_ps"])
        data = dml.DoubleMLData(df, "y", "d", [f"x{i}" for i in range(1, 6)])
        irm = dml.DoubleMLIRM(data, RandomForestRegressor(150, min_samples_leaf=5, random_state=0, n_jobs=1),
                              RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=1), n_folds=5)
        irm.fit()
        return irm

    irm = _irm()
    irm.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho)
    sp = irm.sensitivity_params
    th = float(np.ravel(irm.coef)[0])
    lo, hi = float(np.ravel(sp["theta"]["lower"])[0]), float(np.ravel(sp["theta"]["upper"])[0])
    cil, ciu = float(np.ravel(sp["ci"]["lower"])[0]), float(np.ravel(sp["ci"]["upper"])[0])
    rv, rva = float(np.ravel(sp["rv"])[0]), float(np.ravel(sp["rva"])[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[cil, ciu], y=["CI incl. confounding"] * 2, mode="lines", line=dict(color=PALETTE["coral"], width=6)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=["θ bounds"] * 2, mode="lines", line=dict(color=PALETTE["purple"], width=10)))
    fig.add_trace(go.Scatter(x=[th], y=["θ bounds"], mode="markers", marker=dict(size=14, color="#212529")))
    fig.add_vline(x=0, line=dict(color=PALETTE["muted"], dash="dot"))
    fig.update_layout(showlegend=False, height=230, title=f"ATE θ̂ = {th:.3f} · bounds [{lo:.3f}, {hi:.3f}] · RV = {rv:.1%}, RVa = {rva:.1%}")
    plot(fig)
    st.markdown(f"**التفسير:** مربك غير مقاس يفسّر {cf_y:.0%} من التباين المتبقي في Y و{cf_d:.0%} في المعالجة يحرك التقدير إلى المجال "
                f"[{lo:.3f}, {hi:.3f}]. قيمة المتانة RV = **{rv:.1%}**: مربك بهذه القوة في المعادلتين معًا يكفي لجعل الحد الأدنى صفرًا "
                f"(وRVa = {rva:.1%} لإلغاء الدلالة الإحصائية).")
    st.caption(f"DoubleML {installed_version('doubleml')}: sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0) "
               "— القيم الافتراضية متحقَّق منها؛ `sensitivity_benchmark` يستخدم متغيرًا مقاسًا كمرجع لقوة معقولة.")
why("قارن RV بقوة متغيرات مقاسة (Benchmarking).", "«مربك بقوة التعليم» أوضح من «cf_y = 0.07» لأي قارئ.")

st.markdown("## Cluster-robust inference")
st.markdown("حين تكون الملاحظات مجمّعة (طلاب في مدارس، معاملات لعملاء)، يجب أن تحترم الطيات المجموعات وأن تُحسب الأخطاء المعيارية "
            "مجمّعة؛ وإلا تكون الفترات ضيقة زيفًا. في DoubleML: تمرير `cluster_cols` في DoubleMLData (متعدد الاتجاهات مدعوم، Chiang et al., 2022).")
if dml is not None:
    @st.cache_data(show_spinner="PLR بأخطاء عادية مقابل مجمّعة…")
    def _cluster():
        rng = np.random.default_rng(3)
        G, per = 60, 15
        cl = np.repeat(np.arange(G), per)
        a = rng.normal(size=G)[cl]
        X = rng.normal(size=(G * per, 5)) + a[:, None] * 0.5
        dd = 0.7 * X[:, 0] + a + rng.normal(size=G * per)
        yy = 0.5 * dd + np.sin(X[:, 0]) + 2 * a + rng.normal(size=G * per)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, 6)])
        df["d"], df["y"], df["cluster"] = dd, yy, cl
        rf = RandomForestRegressor(100, min_samples_leaf=5, random_state=0, n_jobs=1)
        plain = dml.DoubleMLPLR(dml.DoubleMLData(df, "y", "d", [f"x{i}" for i in range(1, 6)]), rf, rf, n_folds=5)
        plain.fit()
        clus = dml.DoubleMLPLR(dml.DoubleMLData(df, "y", "d", [f"x{i}" for i in range(1, 6)], cluster_cols="cluster"), rf, rf, n_folds=5)
        clus.fit()
        return float(plain.coef[0]), float(plain.se[0]), float(clus.coef[0]), float(clus.se[0])

    try:
        pc, ps, cc, cs = _cluster()
        c1, c2 = st.columns(2)
        c1.metric("SE عادي (i.i.d.)", f"{ps:.4f}", f"θ̂ = {pc:.3f}", delta_color="off")
        c2.metric("SE مجمّع (cluster_cols)", f"{cs:.4f}", f"θ̂ = {cc:.3f}", delta_color="off")
        st.caption("60 مجموعة × 15 ملاحظة مع أثر مجموعة مشترك: الخطأ المعياري المجمّع أكبر — وهو الصحيح.")
    except Exception as exc:  # the cluster API is version-sensitive: show the verified reference instead of failing
        st.info(f"واجهة التجميع في هذا الإصدار لم تقبل الإعداد المبسط ({type(exc).__name__}). راجع توثيق DoubleML لـcluster_cols.")

st.markdown("## معالجات متعددة وآثار كمّية")
st.code('''# multiple treatments with simultaneous (joint) confidence bands
data = dml.DoubleMLData(df, y_col="y", d_cols=["d1", "d2", "d3"], x_cols=x_cols)
plr = dml.DoubleMLPLR(data, ml_l, ml_m).fit()
plr.bootstrap(n_rep_boot=1000); plr.confint(joint=True)

# quantile treatment effects for a binary treatment
qte = dml.DoubleMLQTE(data_binary, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier(),
                      quantiles=[0.25, 0.5, 0.75]).fit()
qte.summary''', language="python")
intuition("مع عدة معالجات، فترات منفصلة لكل منها لا تضبط احتمال خطأ واحد على الأقل؛ الفترات المشتركة (Multiplier bootstrap) تفعل. "
          "والآثار الكمية تسأل: هل يرفع البرنامج أسفل التوزيع أم أعلاه؟")

if at_least("research"):
    researcher_note(["Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022) لإطار الحساسية المستخدم في DoubleML.",
                     "Chiang et al. (2022) للتجميع متعدد الاتجاهات؛ Belloni et al. (2017) للآثار الكمية وتقييم البرامج.",
                     "أي امتداد «ديناميكي» يجب أن يستند إلى مرجع محدد وافتراضات تعريف صريحة."])
    st.markdown(cite("chernozhukov2022long", "chiang2022", "belloni2017", "semenova2021"))
mistakes(["الإبلاغ دون تحليل حساسية في بيانات رصدية.", "أخطاء معيارية i.i.d. لبيانات مجمّعة.", "فترات منفصلة لمعالجات متعددة.",
          "استخدام مصطلحات امتدادات غير موثقة."])
page_footer("dml_extensions",
            takeaways=["تحليل الحساسية يحدد قوة المربك اللازمة لإلغاء النتيجة (RV).", "البيانات المجمّعة تحتاج طيات وأخطاء مجمّعة.",
                       "الامتدادات المدعومة: CATE/GATE، معالجات متعددة ومستمرة، كمّيات، اختيار العينة، DiD، RDD."])
