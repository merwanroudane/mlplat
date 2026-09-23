import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least
from utils.datasets import xy
from utils.plotting import bars, plot

page_header("catboost")
algorithm_profile("catboost")

st.markdown("## مشكلة ترميز الهدف الساذج")
formula(r"\hat x_i^{k} = \frac{\sum_{j\in\mathcal D_i}\mathbb 1[x_j^k = x_i^k]\,y_j + a\,P}{\sum_{j\in\mathcal D_i}\mathbb 1[x_j^k = x_i^k] + a}",
        title="Ordered target statistic (Prokhorenkova et al., 2018)",
        symbols={r"\mathcal D_i": "الأمثلة التي تسبق i في تبديل عشوائي (Permutation)", "P": "قيمة مسبقة (Prior)", "a": "وزن المسبق"},
        intuition="ترميز الفئة بمتوسط الهدف على **كل** البيانات يتضمن تسمية الصف نفسه ⇒ تسرب (رأيناه في وحدة Leakage). CatBoost "
                  "يستخدم فقط الصفوف «السابقة» في ترتيب عشوائي — كأن البيانات تصل تدريجيًا — فلا يرى الصف تسميته.",
        example="فئة نادرة ظهرت مرة واحدة: الترميز الساذج = تسميتها بالضبط (تسرب تام)؛ المرتّب = المسبق P فقط.")
st.markdown("## Ordered boosting")
intuition("المشكلة نفسها تظهر في التدرجات: البواقي المحسوبة بنموذج رأى الصف متفائلة (Prediction shift). Ordered boosting "
          "يحسب بواقي كل صف بنماذج دُرّبت على الصفوف السابقة فقط. والأشجار **متناظرة** (Oblivious): نفس التقسيم في كل مستوى "
          "⇒ تنبؤ سريع جدًا وتنظيم ضمني.")
why("لا تطبّق One-hot أعمى على كل الفئات قبل CatBoost.",
    "مرّر cat_features مباشرة؛ One-hot المسبق يلغي Target statistics والتوليفات الفئوية التي يبنيها CatBoost، ويضخم الأبعاد مع "
    "الفئات عالية الكاردينالية. CatBoost يستخدم One-hot داخليًا فقط للفئات بعدد قيم ≤ one_hot_max_size.")
hyperparameter_table("CatBoostClassifier")

st.markdown("## مختبر CatBoost: فئات أصلية مقابل One-hot")
cb = optional("catboost")
if cb is None:
    missing_notice("catboost")
else:
    st.caption(f"catboost {installed_version('catboost')} مثبتة.")
    X, y = xy("mixed")
    cats = ["employment", "region", "city"]
    X = X.copy()
    X[cats] = X[cats].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    c1, c2, c3 = st.columns(3)
    depth = c1.slider("depth", 2, 10, 6, key="cb_depth")
    l2 = c2.select_slider("l2_leaf_reg", [0.1, 1, 3, 10, 30], value=3, key="cb_l2")
    iters = c3.slider("iterations", 50, 500, 200, 50, key="cb_it")

    @st.cache_data(show_spinner="يدرّب CatBoost مرتين…", max_entries=16)
    def _fit(depth, l2, iters):
        native = cb.CatBoostClassifier(iterations=iters, depth=depth, l2_leaf_reg=l2, learning_rate=0.08, verbose=0,
                                       random_seed=0, thread_count=1, allow_writing_files=False)
        native.fit(Xtr, ytr, cat_features=cats)
        oh_tr = pd.get_dummies(Xtr, columns=cats)
        oh_te = pd.get_dummies(Xte, columns=cats).reindex(columns=oh_tr.columns, fill_value=0)
        onehot = cb.CatBoostClassifier(iterations=iters, depth=depth, l2_leaf_reg=l2, learning_rate=0.08, verbose=0,
                                       random_seed=0, thread_count=1, allow_writing_files=False).fit(oh_tr, ytr)
        imp = native.get_feature_importance(type="PredictionValuesChange")
        return (roc_auc_score(yte, native.predict_proba(Xte)[:, 1]), roc_auc_score(yte, onehot.predict_proba(oh_te)[:, 1]),
                oh_tr.shape[1], list(Xtr.columns), imp)

    auc_n, auc_o, p_oh, cols, imp = _fit(depth, l2, iters)
    with st.container(horizontal=True):
        st.metric("Native cat_features — Test AUC", f"{auc_n:.4f}", border=True)
        st.metric("Blind one-hot — Test AUC", f"{auc_o:.4f}", f"{auc_o - auc_n:+.4f}", delta_color="off", border=True)
        st.metric("أعمدة بعد One-hot", p_oh, border=True)
    o = np.argsort(-imp)
    plot(bars(np.array(cols)[o], imp[o], title="CatBoost feature importance (PredictionValuesChange)", horizontal=True), height=340)
    st.caption("على هذه البيانات الصغيرة قد يكون الفرق صغيرًا؛ الميزة تتضح مع فئات عالية الكاردينالية وتفاعلات بينها. "
               "لا تعلن فائزًا من تشغيل واحد.")
st.code("""model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                           eval_metric="AUC", early_stopping_rounds=100, verbose=0)
model.fit(X_train, y_train, cat_features=["employment", "region", "city"], eval_set=(X_val, y_val),
          use_best_model=True)""", language="python")
st.caption("القيم الافتراضية المحلولة (iterations=1000، depth=6، l2_leaf_reg=3، border_count=254 على CPU) مقروءة من "
           "get_all_params() في catboost 1.2.10؛ learning_rate يُختار آليًا إن لم يُحدَّد.")

if at_least("research"):
    researcher_note(["Prokhorenkova et al. (2018) أثبتوا أن Target statistics الساذجة والبواقي المحسوبة داخل العينة تسببان "
                     "Prediction shift، واقترحوا الحل المرتّب.",
                     "الفكرة قريبة جدًا من Cross-fitting في DML: لا تستخدم الملاحظة نفسها في بناء ما يُقيِّمها."])
    st.markdown(cite("prokhorenkova2018"))
template_checklist({3: "Ordered target statistics", 7: "Ordered boosting + أشجار متناظرة", 9: "جدول المعاملات (قيم محلولة)",
                    13: "cat_features", 22: "CatBoostClassifier", 23: "مختبر الفئات"})
mistakes(["One-hot أعمى قبل CatBoost.", "ترك الفئات أرقامًا دون إعلانها cat_features.", "ترك كتابة ملفات catboost_info في الإنتاج."])
page_footer("catboost",
            takeaways=["Ordered target statistics تمنع تسرب الهدف في ترميز الفئات.", "مرّر cat_features مباشرة.",
                       "إعدادات افتراضية قوية؛ التدريب أبطأ والتنبؤ سريع."])
