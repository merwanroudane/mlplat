import streamlit as st

from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import footer, page_header

page_header("cheat_sheets")
sheet = st.segmented_control("الورقة", ["Algorithm selection", "Scaling", "Metrics", "CV splitters", "Tree parameters",
                                         "Boosting parameters", "HPO", "DML workflow"], default="Algorithm selection", key="cs_sheet",
                             required=True)

if sheet == "Algorithm selection":
    mermaid("""
flowchart TD
  A{Labels?} -->|no| U{Goal?}
  U -->|groups| C[K-Means / GMM / DBSCAN / HDBSCAN]
  U -->|compress/visualise| P[PCA / t-SNE / UMAP]
  U -->|rare points| N[Isolation Forest / LOF]
  A -->|yes| T{Tabular?}
  T -->|text| X[TF-IDF + Logistic / NB / Linear SVM]
  T -->|yes| I{Interpretability critical?}
  I -->|yes| L[Linear / Logistic / shallow tree / GAM]
  I -->|no| B[HistGradientBoosting / XGBoost / LightGBM / CatBoost + RF baseline]
  A -->|causal question| D[Causal design + DML / EconML]
""")
    st.caption("دائمًا: Baseline + CV مناسب للبنية + مقارنة على طيات مشتركة.")
elif sheet == "Scaling":
    comparison_table([
        {"الخوارزمية": "Linear/Logistic (منظَّم)، Ridge/Lasso/EN", "القياس": "إلزامي (العقوبة حساسة للمقياس)"},
        {"الخوارزمية": "kNN، K-Means، DBSCAN، SVM، PCA، MLP، GD/SGD", "القياس": "إلزامي (مسافات/تدرجات)"},
        {"الخوارزمية": "OLS غير منظَّم", "القياس": "غير لازم للتنبؤ (مفيد للتفسير/التكييف)"},
        {"الخوارزمية": "الأشجار، RF، التعزيز، Naive Bayes، LDA", "القياس": "غير لازم"},
    ])
    st.code("make_pipeline(StandardScaler(), model)   # fit on train folds only", language="python")
elif sheet == "Metrics":
    comparison_table([
        {"الموقف": "تصنيف متوازن، تكاليف متساوية", "المقياس": "Accuracy + ROC-AUC"},
        {"الموقف": "فئة موجبة نادرة", "المقياس": "PR-AUC، Recall@Precision، Balanced accuracy، MCC"},
        {"الموقف": "احتمالات مطلوبة", "المقياس": "Log loss، Brier، Reliability diagram"},
        {"الموقف": "تكاليف غير متماثلة", "المقياس": "التكلفة المتوقعة مع عتبة مضبوطة"},
        {"الموقف": "انحدار، أخطاء كبيرة مكلفة", "المقياس": "RMSE"},
        {"الموقف": "انحدار متين/وسيط", "المقياس": "MAE"},
        {"الموقف": "مئين/مخزون", "المقياس": "Pinball loss"},
        {"الموقف": "عدّ", "المقياس": "Poisson deviance / D²"},
    ])
elif sheet == "CV splitters":
    comparison_table([
        {"البنية": "i.i.d. انحدار", "المُقسِّم": "KFold(shuffle=True)"},
        {"البنية": "تصنيف", "المُقسِّم": "StratifiedKFold"},
        {"البنية": "بيانات قليلة", "المُقسِّم": "RepeatedStratifiedKFold"},
        {"البنية": "كيانات متكررة", "المُقسِّم": "GroupKFold / StratifiedGroupKFold"},
        {"البنية": "زمن", "المُقسِّم": "TimeSeriesSplit(gap=h−1, test_size=…)"},
        {"البنية": "ضبط + تقييم", "المُقسِّم": "Nested CV"},
        {"البنية": "DML", "المُقسِّم": "Cross-fitting (n_folds, n_rep) — ليس CV للتقييم"},
    ])
elif sheet == "Tree parameters":
    comparison_table([
        {"المعامل": "max_depth", "لتقليل Overfitting": "↓", "ملاحظة": "رتبة التفاعلات"},
        {"المعامل": "min_samples_leaf", "لتقليل Overfitting": "↑", "ملاحظة": "الأكثر فاعلية غالبًا"},
        {"المعامل": "max_features", "لتقليل Overfitting": "↓ (في RF)", "ملاحظة": "يقلل ارتباط الأشجار"},
        {"المعامل": "ccp_alpha", "لتقليل Overfitting": "↑", "ملاحظة": "تقليم بعد النمو"},
        {"المعامل": "n_estimators (RF)", "لتقليل Overfitting": "لا يسبب Overfitting", "ملاحظة": "حتى الاستقرار"},
    ])
elif sheet == "Boosting parameters":
    comparison_table([
        {"الوظيفة": "الانكماش", "sklearn HGB": "learning_rate", "XGBoost": "learning_rate (eta=0.3)", "LightGBM": "learning_rate (0.1)", "CatBoost": "learning_rate (auto)"},
        {"الوظيفة": "عدد الأشجار", "sklearn HGB": "max_iter (100)", "XGBoost": "n_estimators (100)", "LightGBM": "n_estimators (100)", "CatBoost": "iterations (1000)"},
        {"الوظيفة": "التعقيد", "sklearn HGB": "max_leaf_nodes (31)", "XGBoost": "max_depth (6)", "LightGBM": "num_leaves (31)", "CatBoost": "depth (6)"},
        {"الوظيفة": "حجم الورقة", "sklearn HGB": "min_samples_leaf (20)", "XGBoost": "min_child_weight (1)", "LightGBM": "min_child_samples (20)", "CatBoost": "min_data_in_leaf"},
        {"الوظيفة": "L2", "sklearn HGB": "l2_regularization (0)", "XGBoost": "reg_lambda (1)", "LightGBM": "reg_lambda (0)", "CatBoost": "l2_leaf_reg (3)"},
        {"الوظيفة": "عينات الصفوف", "sklearn HGB": "—", "XGBoost": "subsample (1)", "LightGBM": "subsample + subsample_freq", "CatBoost": "bootstrap_type / subsample"},
        {"الوظيفة": "عينات الأعمدة", "sklearn HGB": "max_features (1.0)", "XGBoost": "colsample_by* (1)", "LightGBM": "colsample_bytree (1)", "CatBoost": "rsm"},
    ])
    st.caption("القيم بين الأقواس = الافتراضيات المتحقق منها (موسوعة المعاملات)؛ لـCatBoost قيم محلولة.")
elif sheet == "HPO":
    st.markdown("1. عرّف فضاء البحث (لوغاريتمي لـlearning_rate/C/alpha/gamma).\n2. الهدف = متوسط CV مناسب للبنية داخل Pipeline.\n"
                "3. الميزانية: ابدأ بـRandom (أو TPE) بـ30–100 محاولة؛ Halving/Pruning للنماذج المكلفة.\n"
                "4. ثبّت البذور وسجّل كل محاولة.\n5. قيّم الأفضل على Test معزول أو بـNested CV.\n6. قارن بنماذج مضبوطة بالميزانية نفسها.")
else:
    mermaid("""
flowchart LR
  Q[Causal question + DAG] --> A[Assumptions: unconfoundedness, overlap, SUTVA]
  A --> M[Choose model: PLR / PLIV / IRM / IIVM / DiD]
  M --> L[Nuisance learners chosen a priori]
  L --> CF[Cross-fitting K folds, n_rep]
  CF --> E[θ̂, SE, CI]
  E --> S[Sensitivity analysis RV]
  S --> R[Report: design, learners, losses, versions]
""")
    st.code("""data = dml.DoubleMLData(df, y_col, d_cols, x_cols)
model = dml.DoubleMLIRM(data, ml_g, ml_m, n_folds=5, n_rep=5, score="ATE")
model.fit(); model.summary; model.evaluate_learners()
model.sensitivity_analysis(cf_y=0.03, cf_d=0.03); model.sensitivity_summary""", language="python")
footer()
