import inspect

import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least, log_experiment
from utils.datasets import load_dataset, make_plr

page_header("doubleml_package")

dml = optional("doubleml")
st.markdown("## الفئات المتاحة في DoubleML (مقروءة من الحزمة المثبتة)")
comparison_table([
    {"الفئة": "DoubleMLPLR", "النموذج": "Partially linear regression", "الإزعاج": "ml_l, ml_m (ml_g لـIV-type)", "الوحدة": "PLR"},
    {"الفئة": "DoubleMLPLIV", "النموذج": "Partially linear IV", "الإزعاج": "ml_l, ml_m, ml_r", "الوحدة": "PLIV"},
    {"الفئة": "DoubleMLIRM", "النموذج": "Interactive regression (binary D)", "الإزعاج": "ml_g, ml_m", "الوحدة": "IRM"},
    {"الفئة": "DoubleMLIIVM", "النموذج": "Interactive IV (binary Z, D)", "الإزعاج": "ml_g, ml_m, ml_r", "الوحدة": "IIVM"},
    {"الفئة": "DoubleMLAPO / DoubleMLAPOS", "النموذج": "Average potential outcomes (multi-valued D)", "الإزعاج": "ml_g, ml_m", "الوحدة": "Extensions"},
    {"الفئة": "DoubleMLDID / DIDCS / DIDMulti", "النموذج": "DiD (2 فترات، مقاطع متكررة، متعدد الفترات)", "الإزعاج": "ml_g, ml_m", "الوحدة": "Panel DiD"},
    {"الفئة": "DoubleMLPLPR", "النموذج": "Partially linear panel regression", "الإزعاج": "ml_l, ml_m", "الوحدة": "Panel DiD"},
    {"الفئة": "DoubleMLLPLR", "النموذج": "Logistic partially linear (binary Y)", "الإزعاج": "ml_M, ml_t, ml_m", "الوحدة": "Extensions"},
    {"الفئة": "DoubleMLPQ / QTE / LPQ / CVAR", "النموذج": "Potential quantiles, QTE, local PQ, CVaR", "الإزعاج": "ml_g, ml_m", "الوحدة": "Extensions"},
    {"الفئة": "DoubleMLSSM", "النموذج": "Sample selection", "الإزعاج": "ml_g, ml_m, ml_pi", "الوحدة": "Extensions"},
    {"الفئة": "DoubleMLPolicyTree / DoubleMLBLP", "النموذج": "سياسات وCATE", "الإزعاج": "—", "الوحدة": "HTE"},
])
if dml is not None:
    found = sorted(n for n in dir(dml) if n.startswith("DoubleML"))
    st.caption(f"DoubleML {installed_version('doubleml')} مثبتة. الأسماء المكتشفة الآن: " + ", ".join(f"`{n}`" for n in found))
else:
    missing_notice("doubleml")

st.markdown("## سير العمل")
st.code('''import doubleml as dml
from sklearn.ensemble import RandomForestRegressor

data = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols)      # roles of the columns
plr = dml.DoubleMLPLR(data, ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor(),
                      n_folds=5, n_rep=3)                               # cross-fitting design
plr.fit(store_predictions=True)                                         # nuisances + orthogonal score
plr.summary                                                             # θ̂, SE, t, p, CI
plr.confint(level=0.95)
plr.bootstrap(method="normal", n_rep_boot=500); plr.confint(joint=True)  # multiplier bootstrap
plr.evaluate_learners()                                                 # out-of-fold nuisance losses
plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03); plr.sensitivity_summary''', language="python")
if dml is not None:
    st.caption("التوقيعات الحالية (inspect.signature):")
    st.code("\n".join(f"{c}{inspect.signature(getattr(dml, c).__init__)}".replace("(self, ", "(") for c in
                      ("DoubleMLData", "DoubleMLPLR", "DoubleMLIRM")), language="python")

st.markdown("## DoubleML Runner")
if dml is not None:
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("النموذج", ["PLR (make_plr, θ₀ = 0.5)", "IRM ATE (causal data)"], key="dmlr_model")
    learner = c2.selectbox("المتعلم", ["Random Forest", "Lasso / Logistic (CV)"], key="dmlr_learner")
    n_rep = c3.slider("n_rep", 1, 5, 1, key="dmlr_rep")

    @st.cache_data(show_spinner="DoubleML fit…", max_entries=16)
    def _run(model: str, learner: str, n_rep: int):
        if model.startswith("PLR"):
            df = make_plr(n=1000, seed=8)
            data = dml.DoubleMLData(df, "y", "d", [f"x{i}" for i in range(1, 11)])
            reg = RandomForestRegressor(150, min_samples_leaf=5, random_state=0, n_jobs=1) if learner.startswith("Random") \
                else LassoCV(cv=3, random_state=0)
            obj = dml.DoubleMLPLR(data, reg, reg, n_folds=5, n_rep=n_rep)
            truth = 0.5
        else:
            df = load_dataset("causal").drop(columns=["tau", "true_ps"])
            data = dml.DoubleMLData(df, "y", "d", [f"x{i}" for i in range(1, 6)])
            reg = RandomForestRegressor(150, min_samples_leaf=5, random_state=0, n_jobs=1) if learner.startswith("Random") \
                else LassoCV(cv=3, random_state=0)
            clf = RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=1) if learner.startswith("Random") \
                else LogisticRegressionCV(cv=3, max_iter=2000, l1_ratios=(0.0,), use_legacy_attributes=False)
            obj = dml.DoubleMLIRM(data, reg, clf, n_folds=5, n_rep=n_rep)
            truth = float(load_dataset("causal")["tau"].mean())
        obj.fit()
        losses = obj.evaluate_learners()
        return obj.summary, {k: float(np.mean(v)) for k, v in losses.items()}, truth

    if st.button("fit()", key="dmlr_run", type="primary", icon=":material/play_arrow:"):
        st.session_state["dmlr_args"] = (model, learner, n_rep)
    if st.session_state.get("dmlr_args"):
        summ, losses, truth = _run(*st.session_state["dmlr_args"])
        st.dataframe(summ.round(4), width="stretch")
        st.markdown(f"**القيمة الحقيقية في DGP:** {truth:.3f} · **خسائر الإزعاج (RMSE خارج الطية):** "
                    + ", ".join(f"`{k}` = {v:.3f}" for k, v in losses.items()))
        if st.button("سجّل", key="dmlr_log", icon=":material/history:", type="tertiary"):
            log_experiment("DoubleML Runner", st.session_state["dmlr_args"][0], {"learner": st.session_state["dmlr_args"][1],
                           "n_rep": st.session_state["dmlr_args"][2], "n_folds": 5},
                           {"coef": float(summ["coef"].iloc[0]), "se": float(summ["std err"].iloc[0])}, seed=None,
                           dataset=st.session_state["dmlr_args"][0])
            st.toast("سُجّل.", icon=":material/check:")
    st.caption("ملاحظة إصدار scikit-learn 1.8+: في LogisticRegressionCV نمرر l1_ratios=(0.0,) صراحة بدل penalty المهجور، و"
               "use_legacy_attributes=False لتفادي تحذير تغيّر الافتراضي في 1.10.")

st.markdown("## مقارنة المكتبات")
comparison_table([
    {"": "DoubleML", "التركيز": "استدلال على معلمات منخفضة الأبعاد (ATE، θ في PLR/IV، DiD، كمّيات)", "القوة": "مطابقة للأوراق، Cross-fitting، Bootstrap، حساسية",
     "الإصدار المتحقق": "0.11.4"},
    {"": "EconML (PyWhy)", "التركيز": "آثار غير متجانسة وسياسات (DML، DR، Causal forests، Meta-learners)", "القوة": "CATE وفترات وتفسير",
     "الإصدار المتحقق": "0.17.0"},
    {"": "DoWhy (PyWhy)", "التركيز": "النمذجة السببية الكاملة: DAG ← تعريف ← تقدير ← دحض", "القوة": "Refutation tests وتكامل مع EconML",
     "الإصدار المتحقق": "0.14 (غير مثبتة هنا؛ اختيارية)"},
])
why("استخدم DoWhy لصياغة الافتراضات واختبارات الدحض، وDoubleML/EconML للتقدير.",
    "المكتبات متكاملة: DoWhy يمكنه استدعاء مقدِّرات EconML، وDoubleML يعطي استدلالًا بمعايير الأوراق الأصلية.")
intuition("كل المكتبات تنفذ الفكرة نفسها (تعامد + Cross-fitting)؛ الفرق في الواجهة والمعلمات المستهدفة والأدوات المحيطة.")

if at_least("research"):
    researcher_note(["Bach et al. (2022) يصفون تصميم DoubleML الكائني؛ الإصدار الحالي أضاف DiD متعدد الفترات، RDD، وPanel PLR.",
                     "للنشر: اذكر الإصدار، المتعلمين وإعداداتهم، n_folds، n_rep، والبذور."])
    st.markdown(cite("bach2022", "chernozhukov2018"))
mistakes(["DoubleMLData بأعمدة بعد المعالجة في x_cols.", "نسيان n_rep في عينات صغيرة.", "الخلط بين 'ATTE' (DoubleML) و'ATT'."])
page_footer("doubleml_package",
            takeaways=["DoubleMLData يحدد الأدوار؛ كل نموذج فئة؛ fit/summary/confint/bootstrap/sensitivity.",
                       "EconML للآثار غير المتجانسة، DoWhy للافتراضات والدحض.", "وثّق الإصدار والإعدادات."])
