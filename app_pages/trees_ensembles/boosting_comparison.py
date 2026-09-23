import time
import tracemalloc

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from components.callouts import mistakes, researcher_note, why
from components.cards import comparison_table
from config import RANDOM_SEED
from core.page import page_footer, page_header
from core.registry import available, installed_version
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import bars, plot
from utils.preprocessing import ordinal_categoricals

page_header("boosting_comparison")

st.markdown("## المقارنة المفاهيمية")
comparison_table([
    {"": "HistGradientBoosting", "النمو": "Best-first (max_leaf_nodes)", "الفئات": "أصلي (category dtype)",
     "القيم المفقودة": "أصلي", "المعاملات الرئيسية": "learning_rate, max_iter, max_leaf_nodes, l2_regularization",
     "مكتبة إضافية": "لا"},
    {"": "XGBoost", "النمو": "Depth-wise (افتراضي) أو lossguide", "الفئات": "enable_categorical=True",
     "القيم المفقودة": "أصلي (اتجاه متعلَّم)", "المعاملات الرئيسية": "eta, max_depth, min_child_weight, gamma, subsample, colsample_*",
     "مكتبة إضافية": "نعم"},
    {"": "LightGBM", "النمو": "Leaf-wise", "الفئات": "أصلي", "القيم المفقودة": "أصلي",
     "المعاملات الرئيسية": "num_leaves, min_child_samples, feature_fraction, bagging_*", "مكتبة إضافية": "نعم"},
    {"": "CatBoost", "النمو": "Symmetric (oblivious)", "الفئات": "أصلي + Ordered TS (الأقوى)", "القيم المفقودة": "أصلي للعددية",
     "المعاملات الرئيسية": "depth, l2_leaf_reg, iterations, learning_rate", "مكتبة إضافية": "نعم"},
])

st.markdown("## مختبر المقارنة على نفس البيانات ونفس الطيات")
st.caption("بيانات القروض المختلطة (فئات + قيم مفقودة). كل مكتبة بإعداداتها المعتدلة المعقولة ونفس ميزانية الأشجار، "
           "3 طيات طبقية مشتركة. الذاكرة = ذروة تخصيص Python أثناء fit (تقدير تقريبي؛ لا يشمل كل ذاكرة C++).")
libs = ["HistGradientBoosting"] + [n for n, mod in (("XGBoost", "xgboost"), ("LightGBM", "lightgbm"), ("CatBoost", "catboost"))
                                   if available(mod)]
missing = [n for n, mod in (("XGBoost", "xgboost"), ("LightGBM", "lightgbm"), ("CatBoost", "catboost")) if not available(mod)]
if missing:
    st.info(f"غير مثبتة هنا (ستُتخطى): {', '.join(missing)}.", icon=":material/extension_off:")
n_trees = st.select_slider("عدد الأشجار لكل مكتبة", [50, 100, 200, 300], value=200, key="bc_n")
chosen = st.pills("المكتبات", libs, selection_mode="multi", default=libs, key="bc_libs")
X, y = xy("mixed")
cats = ["employment", "region", "city"]


def _make(name, n):
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.1, early_stopping=False, random_state=RANDOM_SEED), "category"
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, learning_rate=0.1, max_depth=6, tree_method="hist", enable_categorical=True,
                                 random_state=RANDOM_SEED, n_jobs=1), "category"
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.1, num_leaves=31, random_state=RANDOM_SEED, n_jobs=1,
                                  verbose=-1), "category"
    import catboost as cb
    return cb.CatBoostClassifier(iterations=n, learning_rate=0.1, depth=6, verbose=0, random_seed=RANDOM_SEED, thread_count=1,
                                 allow_writing_files=False, cat_features=cats), "str"


@st.cache_data(show_spinner="يدرّب المكتبات على 3 طيات…", max_entries=8)
def _benchmark(names: tuple, n: int):
    rows = []
    folds = list(StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED).split(X, y))
    for name in names:
        aucs, lls, secs, mems = [], [], [], []
        for tr, te in folds:
            model, cat_mode = _make(name, n)
            Xc = ordinal_categoricals(X) if cat_mode == "category" else X.astype({c: str for c in cats})
            tracemalloc.start()
            t0 = time.perf_counter()
            model.fit(Xc.iloc[tr], y.iloc[tr])
            secs.append(time.perf_counter() - t0)
            mems.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
            p = model.predict_proba(Xc.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te], p))
            lls.append(log_loss(y.iloc[te], p))
        rows.append({"library": name, "ROC-AUC": np.mean(aucs), "AUC SD": np.std(aucs), "log loss": np.mean(lls),
                     "fit seconds / fold": np.mean(secs), "peak Python MB": np.mean(mems)})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="bc_run", type="primary", icon=":material/play_arrow:") and chosen:
    st.session_state["bc_args"] = (tuple(chosen), n_trees)
if st.session_state.get("bc_args"):
    res = _benchmark(*st.session_state["bc_args"])
    st.dataframe(res.style.format({c: "{:.4f}" for c in ("ROC-AUC", "AUC SD", "log loss")} |
                                  {"fit seconds / fold": "{:.2f}", "peak Python MB": "{:.1f}"}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        plot(bars(res["library"], res["ROC-AUC"], errors=res["AUC SD"], title="ROC-AUC (mean ± SD over folds)",
                  color=PALETTE["sky"]), height=320)
    with c2:
        plot(bars(res["library"], res["fit seconds / fold"], title="Fit time per fold (s)", color=PALETTE["coral"], text_fmt=".2f"),
             height=320)
    spread = res["ROC-AUC"].max() - res["ROC-AUC"].min()
    st.markdown(f"الفرق بين الأعلى والأدنى في AUC = **{spread:.4f}**، مقابل انحراف معياري عبر الطيات ≈ {res['AUC SD'].mean():.4f}. "
                + ("الفروق ضمن الضجيج: **لا فائز**." if spread < 2 * res["AUC SD"].mean() else
                   "الفرق أكبر من ضجيج الطيات هنا، لكنه قد يتغير مع الضبط وبيانات أخرى."))
    if st.button("سجّل النتائج", key="bc_log", icon=":material/history:", type="tertiary"):
        for _, r in res.iterrows():
            log_experiment("Boosting Comparison Lab", r["library"], {"n_trees": st.session_state["bc_args"][1],
                           "learning_rate": 0.1}, {"roc_auc": r["ROC-AUC"], "fit_s": r["fit seconds / fold"]}, seed=RANDOM_SEED,
                           dataset="mixed", split="StratifiedKFold(3)")
        st.toast("سُجّلت.", icon=":material/check:")
st.caption("الإصدارات المثبتة: " + ", ".join(f"{m} {installed_version(m) or '—'}" for m in ("xgboost", "lightgbm", "catboost")))
why("قارن بعد ضبط كل مكتبة بميزانية متساوية، وعلى عدة مجموعات بيانات.",
    "الإعدادات الافتراضية تختلف كثيرًا (CatBoost 1000 تكرار بمعدل آلي، XGBoost 100 جولة بـeta 0.3)؛ مقارنة الافتراضيات تقارن "
    "الإعدادات لا الخوارزميات.")

st.markdown("## التفسير المشترك")
comparison_table([
    {"الأداة": "Permutation importance", "HGB": "✓", "XGBoost": "✓", "LightGBM": "✓", "CatBoost": "✓"},
    {"الأداة": "SHAP (TreeSHAP)", "HGB": "عبر shap.TreeExplainer", "XGBoost": "pred_contribs أصلي", "LightGBM": "pred_contrib أصلي",
     "CatBoost": "ShapValues أصلي"},
    {"الأداة": "Built-in importance", "HGB": "—", "XGBoost": "gain/weight/cover", "LightGBM": "split/gain", "CatBoost": "PredictionValuesChange"},
])
if at_least("research"):
    researcher_note(["في المعايير المنشورة (Benchmarks) تتقارب المكتبات الثلاث بعد الضبط؛ الفروق تعتمد على البيانات.",
                     "وثّق: الإصدارات، البذور، عدد الخيوط، الميزانية، استراتيجية الضبط، والطيات المشتركة."])
mistakes(["إعلان فائز من تشغيل واحد بالإعدادات الافتراضية.", "مقارنة على طيات مختلفة.", "تجاهل زمن التدريب والتنبؤ والذاكرة."])
page_footer("boosting_comparison",
            takeaways=["المكتبات الأربع متقاربة في الدقة غالبًا بعد الضبط.", "تختلف في معالجة الفئات والنمو والسرعة.",
                       "قارن بإنصاف: نفس الطيات، ميزانية متساوية، وعدم يقين."])
