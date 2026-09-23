import streamlit as st

from content.algorithm_metadata import ALGORITHMS
from content.glossary import GLOSSARY
from content.hyperparameters import HYPERPARAMETERS
from core.curriculum import MODULES, get_module
from core.page import footer, page_header

page_header("search")

FORMULA_INDEX = [
    ("Empirical risk / ERM", "statistical_learning"), ("Bias–variance decomposition", "bias_variance"), ("MSE, MAE, Huber, pinball", "loss_functions"),
    ("Log loss, hinge, exponential", "loss_functions"), ("Gradient descent update", "gradient_descent"), ("Newton step", "advanced_optimizers"),
    ("Soft-thresholding (Lasso)", "advanced_optimizers"), ("Normal equations / hat matrix", "linear_regression"), ("Elastic Net objective", "regularization"),
    ("Sigmoid / log-odds", "logistic_regression"), ("Bayes rule / Naive Bayes", "naive_bayes"), ("LDA discriminant", "lda_qda"),
    ("SVM margin / kernel", "svm"), ("Gini / entropy / information gain", "decision_trees"), ("Gradient boosting pseudo-residuals", "boosting"),
    ("XGBoost gain and leaf weight", "xgboost"), ("Precision / recall / F-beta", "classification_metrics"), ("Brier / ECE", "calibration"),
    ("Cost-optimal threshold", "threshold_tuning"), ("SMOTE", "imbalanced"), ("Split conformal quantile", "conformal_prediction"),
    ("Expected improvement", "bayesian_optimization"), ("K-Means objective / silhouette", "kmeans"), ("EM responsibilities / BIC", "gmm"),
    ("PCA via SVD", "pca"), ("t-SNE objective", "manifold"), ("Isolation score", "anomaly_detection"), ("Permutation importance", "model_inspection"),
    ("Partial dependence", "pdp_ice"), ("Shapley value", "shap"), ("Potential outcomes / ATE / ATT / CATE", "causal_foundations"),
    ("PLR model / DML estimator", "dml_core"), ("Neyman orthogonality", "neyman_orthogonality"), ("DML1 / DML2", "cross_fitting"),
    ("PLIV estimator", "pliv"), ("AIPW score (ATE/ATT)", "irm"), ("LATE / Wald", "iivm"), ("GATE / BLP", "hte"), ("ATT(g, t)", "panel_did"),
    ("Omitted-variable bias bound / RV", "dml_extensions"), ("PSI", "drift"), ("Demographic parity / equal opportunity", "fairness"),
    ("Q-learning update", "rl_bridge"), ("Backpropagation", "dl_bridge"), ("TF-IDF", "text_ml"),
]

q = st.text_input("ابحث", key="search_q", placeholder="مثال: cross-fitting، gamma، معايرة، leakage، SHAP")
if q:
    ql = q.lower()
    mods = [m for m in MODULES if ql in " ".join([m.title_ar, m.title_en, m.description, *m.keywords, *m.objectives]).lower()]
    algos = [a for a in ALGORITHMS if ql in f"{a.name} {a.problem} {a.intuition} {a.estimator}".lower()]
    hps = [h for h in HYPERPARAMETERS if ql in f"{h.parameter} {h.algo} {h.meaning}".lower()]
    forms = [(f, mid) for f, mid in FORMULA_INDEX if ql in f.lower()]
    gl = [g for g in GLOSSARY if ql in " ".join(g).lower()]
    labs = [m for m in MODULES if m.lab and ql in f"{m.lab} {m.title_en}".lower()]
    total = len(mods) + len(algos) + len(hps) + len(forms) + len(gl) + len(labs)
    st.caption(f"{total} نتيجة")
    tabs = st.tabs([f"الدروس ({len(mods)})", f"الخوارزميات ({len(algos)})", f"المعاملات ({len(hps)})", f"الصيغ ({len(forms)})",
                    f"المسرد ({len(gl)})", f"المختبرات ({len(labs)})"])
    with tabs[0]:
        for m in mods:
            st.page_link(m.file, label=f"{m.title_ar} · {m.title_en}", icon=m.icon)
            st.caption(m.description)
    with tabs[1]:
        for a in algos:
            st.page_link(get_module(a.module).file, label=f"{a.name} — {a.problem}", icon=":material/model_training:")
    with tabs[2]:
        for h in hps:
            st.markdown(f"- `{h.algo}.{h.parameter}` = `{h.default!r}`{' → ' + h.resolved if h.resolved else ''} — {h.meaning}")
        if hps:
            st.page_link(get_module("hyperparameter_encyclopedia").file, label="افتح الموسوعة", icon=":material/menu_book:")
    with tabs[3]:
        for f, mid in forms:
            st.page_link(get_module(mid).file, label=f"{f} — {get_module(mid).title_ar}", icon=":material/functions:")
    with tabs[4]:
        for en, ar, definition, mid in gl:
            st.markdown(f"**{en} · {ar}** — {definition}")
            st.page_link(get_module(mid).file, label=f"الوحدة: {get_module(mid).title_ar}", icon=":material/arrow_back:")
    with tabs[5]:
        for m in labs:
            st.page_link(m.file, label=f"{m.lab} — {m.title_ar}", icon=":material/science:")
else:
    st.info("اكتب كلمة بالعربية أو الإنجليزية. يبحث في الدروس والخوارزميات والمعاملات الفائقة والصيغ والمسرد والمختبرات.",
            icon=":material/search:")
footer()
