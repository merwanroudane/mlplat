import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from config import MAX_TREES_LAB
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import lines, plot

page_header("xgboost")
algorithm_profile("xgboost")

st.markdown("## الهدف المنظَّم والتقريب من الرتبة الثانية")
formula(r"\mathcal L^{(t)} = \sum_i \ell\big(y_i, \hat y_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t),\qquad "
        r"\Omega(f) = \gamma T + \tfrac12\lambda\sum_{j=1}^T w_j^2 + \alpha\sum_j|w_j|",
        title="XGBoost objective (Chen & Guestrin, 2016)",
        symbols={"T": "عدد الأوراق", "w_j": "وزن (تنبؤ) الورقة j", r"\gamma": "gamma: ثمن كل ورقة",
                 r"\lambda, \alpha": "reg_lambda (L2) وreg_alpha (L1) على أوزان الأوراق"})
formula(r"g_i = \partial_{\hat y}\ell(y_i,\hat y_i),\ h_i = \partial^2_{\hat y}\ell(y_i,\hat y_i);\quad "
        r"w_j^* = -\frac{G_j}{H_j+\lambda},\quad \text{Gain} = \tfrac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma",
        title="Gradients, hessians, optimal leaf weight and split gain",
        symbols={"G_j, H_j": "مجموع g وh للعينات في الورقة j"},
        intuition="تقريب تايلور التربيعي يعطي حلًا مغلقًا لوزن كل ورقة وصيغة كسب لكل تقسيم. التقسيم يُقبل فقط إن تجاوز كسبه γ. "
                  "λ في المقام يقلص الأوزان؛ min_child_weight = الحد الأدنى لـH في الابن.",
        example="Log loss: g = p − y، h = p(1 − p). ورقة فيها G = −12، H = 5، λ = 1 ⇒ w* = 12/6 = 2 (log-odds).")
comparison_table_rows = [
    {"مجموعة": "الانكماش", "المعاملات": "learning_rate (eta), n_estimators", "الغرض": "خطوات صغيرة + جولات كافية"},
    {"مجموعة": "تعقيد الشجرة", "المعاملات": "max_depth, min_child_weight, gamma, max_leaves", "الغرض": "التحكم في السعة"},
    {"مجموعة": "العشوائية", "المعاملات": "subsample, colsample_bytree/bylevel/bynode", "الغرض": "تنوع وتنظيم وسرعة"},
    {"مجموعة": "التنظيم", "المعاملات": "reg_lambda, reg_alpha", "الغرض": "انكماش أوزان الأوراق"},
    {"مجموعة": "الحساب", "المعاملات": "tree_method='hist', device, max_bin", "الغرض": "السرعة و GPU"},
]
st.dataframe(pd.DataFrame(comparison_table_rows), hide_index=True, width="stretch")
hyperparameter_table("XGBClassifier")

st.markdown("## مختبر XGBoost مع التوقف المبكر")
xgb = optional("xgboost")
if xgb is None:
    missing_notice("xgboost")
else:
    st.caption(f"xgboost {installed_version('xgboost')} مثبتة.")
    X, y = xy("classification")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)
    c1, c2, c3, c4 = st.columns(4)
    lr = c1.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.3, 0.5], value=0.1, key="xgb_lr")
    depth = c2.slider("max_depth", 1, 10, 6, key="xgb_depth")
    mcw = c3.select_slider("min_child_weight", [0.1, 1, 3, 10, 30], value=1, key="xgb_mcw")
    gamma = c4.select_slider("gamma", [0.0, 0.1, 0.5, 1.0, 5.0], value=0.0, key="xgb_gamma")
    c5, c6, c7, c8 = st.columns(4)
    sub = c5.slider("subsample", 0.3, 1.0, 1.0, 0.1, key="xgb_sub")
    col = c6.slider("colsample_bytree", 0.3, 1.0, 1.0, 0.1, key="xgb_col")
    lam = c7.select_slider("reg_lambda", [0.0, 1.0, 5.0, 20.0], value=1.0, key="xgb_lam")
    es = c8.toggle("early stopping (20 rounds)", value=True, key="xgb_es")

    @st.cache_data(show_spinner="يدرّب XGBoost…", max_entries=32)
    def _fit(lr, depth, mcw, gamma, sub, col, lam, es):
        m = xgb.XGBClassifier(n_estimators=MAX_TREES_LAB, learning_rate=lr, max_depth=depth, min_child_weight=mcw, gamma=gamma,
                              subsample=sub, colsample_bytree=col, reg_lambda=lam, tree_method="hist", eval_metric="logloss",
                              early_stopping_rounds=20 if es else None, random_state=0, n_jobs=1)
        m.fit(Xa, ya, eval_set=[(Xa, ya), (Xv, yv)], verbose=False)
        ev = m.evals_result()
        from sklearn.metrics import roc_auc_score
        best = getattr(m, "best_iteration", None)
        return ev["validation_0"]["logloss"], ev["validation_1"]["logloss"], best, roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    tr, va, best, auc = _fit(lr, depth, mcw, gamma, sub, col, lam, es)
    fig = lines(np.arange(1, len(tr) + 1), {"train log loss": tr, "validation log loss": va}, title="XGBoost learning curves",
                xaxis="boosting round", yaxis="log loss")
    if best is not None:
        fig.add_vline(x=best + 1, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text=f"best iteration {best + 1}")
    plot(fig, height=340)
    with st.container(horizontal=True):
        st.metric("جولات مدرّبة", len(tr), border=True)
        st.metric("أفضل جولة", best + 1 if best is not None else "—", border=True)
        st.metric("Test ROC-AUC", f"{auc:.4f}", border=True)
    if st.button("سجّل التجربة", key="xgb_log", icon=":material/history:", type="tertiary"):
        log_experiment("XGBoost Lab", "XGBClassifier", {"learning_rate": lr, "max_depth": depth, "min_child_weight": mcw,
                       "gamma": gamma, "subsample": sub, "colsample_bytree": col, "reg_lambda": lam}, {"test_auc": auc},
                       seed=0, dataset="classification", split="train/valid/test 60/15/25 stratified")
        st.toast("سُجّلت.", icon=":material/check:")
    why("مجموعة التوقف المبكر يجب أن تكون منفصلة عن Test.", "عدد الجولات يُختار عليها؛ الإبلاغ عليها تفاؤل.")

st.code("""model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50,
                          eval_metric="logloss", enable_categorical=True)      # pandas 'category' supported
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
model.best_iteration""", language="python")
intuition("في XGBoost ≥ 2.0 يُمرَّر early_stopping_rounds وeval_metric إلى المُنشئ لا إلى fit، وtree_method الافتراضي "
          "الفعلي هو 'hist' (متحقَّق من config الـbooster في الإصدار المثبت).")

if at_least("advanced"):
    st.markdown("## متقدم: القيم المفقودة وSHAP الأصلي")
    st.markdown("- عند كل تقسيم يتعلم XGBoost **اتجاهًا افتراضيًا** للقيم المفقودة (Sparsity-aware split finding).\n"
                "- `model.get_booster().predict(DMatrix(X), pred_contribs=True)` يعطي قيم TreeSHAP أصليًا.\n"
                "- الأهمية: `importance_type` بين weight/gain/cover — كلها متحيزة كـMDI؛ استخدم Permutation أو SHAP.")
if at_least("research"):
    researcher_note(["Chen & Guestrin (2016): Weighted quantile sketch وSparsity-aware splits وCache-aware access.",
                     "الإبلاغ: اذكر الإصدار وtree_method وearly stopping وبيانات التوقف؛ النتائج تتغير بينها."])
    st.markdown(cite("chen2016"))
template_checklist({3: "الهدف المنظَّم", 6: "Gain وأوزان الأوراق", 7: "تقريب الرتبة الثانية", 9: "جدول المعاملات الفائقة (قيم محلولة)",
                    22: "XGBClassifier", 23: "مختبر XGBoost", 24: "منزلقات المختبر"})
mistakes(["early stopping على Test.", "ضبط عشرات المعاملات بشبكة كاملة.", "الاعتماد على feature_importances_ الافتراضية."])
page_footer("xgboost",
            takeaways=["XGBoost = تعزيز بتقريب تربيعي وتنظيم صريح على الأوراق.", "gamma وmin_child_weight وlambda تتحكم في التقسيم.",
                       "استخدم hist + early stopping على تحقق منفصل."])
