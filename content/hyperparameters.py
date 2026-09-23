"""Central hyperparameter metadata (sections 17 and 87).

Defaults are written ONCE here. They were read from the installed libraries
on the date in ``CHECKED`` (see docs/research_notes.md) and are re-checked
automatically by tests/test_hyperparameters.py against ``inspect.signature``
(scikit-learn, LightGBM, DoubleML, Optuna) or the resolved booster config
(XGBoost, CatBoost). A version upgrade that changes a default makes that
test fail, which forces a re-verification instead of silently wrong teaching.

``default`` is the Python literal as it appears in the signature;
``resolved`` (optional) is the effective value when the signature says None.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED = "2026-09-23"

VERIFIED_VERSIONS = {
    "scikit-learn": "1.9.1",
    "xgboost": "3.2.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "optuna": "5.0.0",
    "DoubleML": "0.11.4",
    "econml": "0.17.0",
    "imbalanced-learn": "0.14.2",
}

SK = "https://scikit-learn.org/stable/modules/generated/"
XGB = "https://xgboost.readthedocs.io/en/stable/parameter.html"
LGB = "https://lightgbm.readthedocs.io/en/stable/Parameters.html"
CB = "https://catboost.ai/docs/en/references/training-parameters/"
DML = "https://docs.doubleml.org/stable/api/api.html"
OPT = "https://optuna.readthedocs.io/en/stable/reference/samplers/index.html"
IMB = "https://imbalanced-learn.org/stable/references/index.html"


@dataclass(frozen=True)
class HP:
    library: str
    estimator: str          # import path used by the test, e.g. "sklearn.svm.SVC"
    parameter: str
    default: object         # value in the signature (checked by tests)
    allowed: str
    meaning: str
    increase: str
    decrease: str
    interactions: str = ""
    runtime: str = ""
    risk: str = ""          # which direction risks over/under-fitting
    resolved: str = ""      # effective value when default is None / 'auto'
    note: str = ""
    url: str = ""

    @property
    def algo(self) -> str:
        return self.estimator.rsplit(".", 1)[-1]

    @property
    def verified_version(self) -> str:
        return VERIFIED_VERSIONS.get(self.library, "")


_LR = "sklearn.linear_model.LogisticRegression"
_SVC = "sklearn.svm.SVC"
_KNN = "sklearn.neighbors.KNeighborsClassifier"
_DT = "sklearn.tree.DecisionTreeClassifier"
_RF = "sklearn.ensemble.RandomForestClassifier"
_GB = "sklearn.ensemble.GradientBoostingClassifier"
_HGB = "sklearn.ensemble.HistGradientBoostingClassifier"
_XGB = "xgboost.XGBClassifier"
_LGB = "lightgbm.LGBMClassifier"
_CB = "catboost.CatBoostClassifier"

HYPERPARAMETERS: tuple[HP, ...] = (
    # ------------------------------------------------ linear models
    HP("scikit-learn", _LR, "C", 1.0, "float > 0 (مقياس لوغاريتمي عادةً 1e-4…1e4)",
       "مقلوب قوة التنظيم: الهدف هو C·Σloss + penalty.",
       "تنظيم أضعف، معاملات أكبر، حدّ قرار أكثر مرونة.", "تنظيم أقوى، انكماش المعاملات نحو الصفر.",
       "يتفاعل مع l1_ratio (نوع العقوبة) ومع قياس الخصائص (التنظيم حساس للمقياس).",
       "قيم C الكبيرة قد تحتاج max_iter أكبر للتقارب.", "C كبير ⇒ Overfitting؛ C صغير ⇒ Underfitting.",
       url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "l1_ratio", 0.0, "float في [0, 1]",
       "مزج العقوبة: 0 = L2، 1 = L1، بينهما Elastic Net. منذ 1.8 هو الطريقة المعتمدة لاختيار نوع العقوبة.",
       "ندرة أكبر (معاملات صفرية أكثر).", "انكماش سلس دون إصفار.",
       "قيم > 0 تتطلب solver='saga' (أو 'liblinear' لـ L1 الثنائي). C=np.inf يعني بلا تنظيم.",
       "saga أبطأ ويستفيد من القياس.", "",
       note="penalty مهجور منذ 1.8 ويُزال في 1.10: استخدم l1_ratio=0 بدل penalty='l2' وl1_ratio=1 بدل 'l1' وC=np.inf بدل None.",
       url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "penalty", "deprecated", "مهجور (deprecated) في 1.8، يُزال في 1.10",
       "كان يحدد نوع العقوبة. اتركه على قيمته الافتراضية واستخدم l1_ratio وC.",
       "—", "—", "استخدامه مع l1_ratio يعطي تحذير عدم اتساق.", "", "",
       note="تحقق مباشر: رسالة DeprecationWarning في scikit-learn 1.9.1.",
       url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "solver", "lbfgs", "'lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'",
       "خوارزمية التحسين المستخدمة (Solver option، ليس Hyperparameter إحصائيًا).",
       "—", "—",
       "lbfgs/newton-cg/newton-cholesky/sag: L2 أو بلا عقوبة؛ saga: كل العقوبات؛ liblinear: L1/L2 ثنائي وOvR فقط.",
       "newton-cholesky سريع حين n ≫ p؛ sag/saga للبيانات الكبيرة المقاسة.", "",
       url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "class_weight", None, "None | 'balanced' | dict",
       "وزن خسارة كل فئة؛ 'balanced' يعطي n/(k·n_c).", "—", "—",
       "يغيّر الاحتمالات (يُفقد المعايرة) — بديل: ضبط العتبة.", "", "",
       url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "max_iter", 100, "int > 0", "حد أقصى لتكرارات الـsolver.",
       "يسمح بالتقارب؛ لا يغير الحل إن تقارب.", "قد يوقف قبل التقارب (ConvergenceWarning).", "",
       "زمن أطول.", "", url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "tol", 0.0001, "float > 0", "معيار التوقف.", "توقف أبكر وحل أقل دقة.", "دقة أعلى وزمن أطول.",
       "", "", "", url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", _LR, "fit_intercept", True, "bool", "هل يُقدَّر الحد الثابت؟", "—", "—",
       "الحد الثابت لا يُعاقَب في lbfgs.", "", "", url=SK + "sklearn.linear_model.LogisticRegression.html"),
    HP("scikit-learn", "sklearn.linear_model.Ridge", "alpha", 1.0, "float ≥ 0",
       "قوة عقوبة L2: ‖y − Xw‖² + α‖w‖².", "انكماش أكبر وتباين أقل.", "يقترب من OLS.",
       "يعتمد على مقياس X؛ مع الخصائص المترابطة يوزّع الأثر بينها.", "", "α صغير ⇒ Overfitting؛ كبير ⇒ Underfitting.",
       url=SK + "sklearn.linear_model.Ridge.html"),
    HP("scikit-learn", "sklearn.linear_model.Lasso", "alpha", 1.0, "float ≥ 0",
       "قوة عقوبة L1: (1/2n)‖y − Xw‖² + α‖w‖₁.", "معاملات صفرية أكثر.", "خصائص أكثر تدخل النموذج.",
       "مع الخصائص المترابطة يختار واحدة عشوائيًا تقريبًا.", "Coordinate descent؛ مسار alphas مع warm start سريع.",
       "", url=SK + "sklearn.linear_model.Lasso.html"),
    HP("scikit-learn", "sklearn.linear_model.ElasticNet", "l1_ratio", 0.5, "float في [0, 1]",
       "نسبة L1 في العقوبة α·(l1_ratio‖w‖₁ + ½(1 − l1_ratio)‖w‖²).", "سلوك أقرب لـLasso.", "سلوك أقرب لـRidge.",
       "يتفاعل مع alpha؛ اضبطهما معًا (ElasticNetCV).", "", "", url=SK + "sklearn.linear_model.ElasticNet.html"),
    # ------------------------------------------------ kNN
    HP("scikit-learn", _KNN, "n_neighbors", 5, "int ≥ 1", "عدد الجيران المصوِّتين.",
       "حدود أنعم، تحيز أعلى.", "حدود متعرجة، تباين أعلى (k=1 يحفظ التدريب).",
       "مع weights='distance' يقل أثر k الكبير.", "تكلفة التنبؤ تزداد قليلًا.", "k صغير ⇒ Overfitting.",
       url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    HP("scikit-learn", _KNN, "weights", "uniform", "'uniform' | 'distance' | callable", "تصويت متساوٍ أم مرجّح بمقلوب المسافة.",
       "—", "—", "'distance' يجعل train accuracy = 1 دائمًا.", "", "", url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    HP("scikit-learn", _KNN, "metric", "minkowski", "'minkowski', 'cosine', 'manhattan', ...", "دالة المسافة.",
       "—", "—", "مع 'minkowski' يحدد p نوع المسافة.", "بعض المقاييس تمنع KD/Ball tree.", "",
       url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    HP("scikit-learn", _KNN, "p", 2, "float ≥ 1 (مع minkowski)", "أس مسافة Minkowski: 1 = Manhattan، 2 = Euclidean.",
       "يعطي وزنًا أكبر لأكبر فرق إحداثي.", "أكثر متانة أمام فرق كبير في بُعد واحد.", "", "", "",
       url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    HP("scikit-learn", _KNN, "algorithm", "auto", "'auto','ball_tree','kd_tree','brute'", "بنية البحث عن الجيران (لا تغيّر النتيجة).",
       "—", "—", "KD-tree يضعف في الأبعاد العالية.", "يؤثر في السرعة فقط.", "",
       url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    HP("scikit-learn", _KNN, "leaf_size", 30, "int ≥ 1", "حجم الورقة في KD/Ball tree (سرعة وذاكرة فقط).", "—", "—", "", "",
       "", url=SK + "sklearn.neighbors.KNeighborsClassifier.html"),
    # ------------------------------------------------ SVM
    HP("scikit-learn", _SVC, "C", 1.0, "float > 0", "كلفة انتهاك الهامش في SVM اللين.",
       "هامش أضيق، متجهات داعمة أقل، ملاءمة أدق للتدريب.", "هامش أوسع وتنظيم أقوى.",
       "مع RBF يتفاعل بقوة مع gamma: ابحث على شبكة لوغاريتمية مشتركة.", "C كبير يبطئ التقارب.",
       "C كبير ⇒ Overfitting.", url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "kernel", "rbf", "'linear','poly','rbf','sigmoid','precomputed' | callable",
       "دالة النواة التي تعرّف التشابه (فضاء الخصائص الضمني).", "—", "—", "gamma/degree/coef0 تخص نوى بعينها.",
       "التدريب بين O(n²) وO(n³) — لا يصلح لمئات الآلاف.", "", url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "gamma", "scale", "'scale' | 'auto' | float > 0",
       "مدى تأثير النقطة في RBF: K = exp(−γ‖x − x'‖²). 'scale' = 1/(p·Var(X)).",
       "تأثير محلي جدًا، حدود متعرجة.", "حدود أنعم، أقرب للخطي.", "مع C.", "", "γ كبير ⇒ Overfitting.",
       url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "degree", 3, "int ≥ 0", "درجة نواة poly فقط.", "مرونة أكبر.", "أبسط.", "يُتجاهل لغير poly.",
       "", "", url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "coef0", 0.0, "float", "الحد الثابت في نواتي poly وsigmoid.", "—", "—", "", "", "",
       url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "class_weight", None, "None | 'balanced' | dict", "يضرب C لكل فئة.", "—", "—", "", "", "",
       url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "probability", "deprecated", "مهجور في 1.9، يُزال في 1.11",
       "كان يفعّل Platt scaling داخليًا بـ5-fold CV.", "—", "—",
       "البديل الموصى به: CalibratedClassifierCV(SVC(), ensemble=False).", "", "",
       note="تحقق مباشر من رسالة التحذير في scikit-learn 1.9.1.", url=SK + "sklearn.svm.SVC.html"),
    HP("scikit-learn", _SVC, "tol", 0.001, "float > 0", "معيار التوقف.", "—", "—", "", "", "", url=SK + "sklearn.svm.SVC.html"),
    # ------------------------------------------------ trees
    HP("scikit-learn", _DT, "criterion", "gini", "'gini','entropy','log_loss'",
       "مقياس الشوائب لاختيار التقسيم ('entropy' و'log_loss' متكافئان).", "—", "—",
       "للانحدار: 'squared_error','absolute_error','poisson' ('friedman_mse' مهجور في 1.9).", "", "",
       url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "splitter", "best", "'best' | 'random'", "أفضل تقسيم أم أفضل تقسيم عشوائي.", "—", "—",
       "'random' يشبه Extra Trees لشجرة واحدة.", "'random' أسرع.", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "max_depth", None, "None | int ≥ 1", "أقصى عمق للشجرة.", "تفاعلات أعمق وتباين أعلى.",
       "شجرة أبسط وتحيز أعلى.", "مع min_samples_leaf وmax_leaf_nodes.", "", "عمق كبير ⇒ Overfitting.",
       url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "min_samples_split", 2, "int ≥ 2 | float (نسبة)", "أقل عدد عينات للسماح بتقسيم عقدة.",
       "شجرة أصغر.", "تقسيم أكثر.", "", "", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "min_samples_leaf", 1, "int ≥ 1 | float (نسبة)", "أقل عدد عينات في كل ورقة.",
       "أوراق أكبر وتنبؤات أنعم.", "أوراق نقية صغيرة (حفظ).", "من أكثر المعاملات فاعلية ضد Overfitting.", "",
       "1 ⇒ خطر Overfitting.", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "max_features", None, "None | 'sqrt' | 'log2' | int | float",
       "عدد الخصائص المرشحة عند كل تقسيم.", "أشجار أقوى وأكثر ترابطًا.", "عشوائية أكبر.", "جوهري في Random Forest.",
       "أقل ⇒ أسرع.", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "max_leaf_nodes", None, "None | int ≥ 2", "حد لعدد الأوراق (نمو أفضل-أولًا).",
       "مرونة أكبر.", "أبسط.", "إن حُدد يُنمّى أفضل-أولًا.", "", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "min_impurity_decrease", 0.0, "float ≥ 0", "أقل انخفاض موزون في الشوائب لقبول التقسيم.",
       "تقسيمات أقل.", "—", "", "", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "class_weight", None, "None | 'balanced' | dict", "أوزان الفئات في حساب الشوائب.", "—", "—",
       "", "", "", url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "ccp_alpha", 0.0, "float ≥ 0", "معامل تقليم Minimal cost-complexity.",
       "تقليم أكثر وشجرة أصغر.", "شجرة كاملة.", "اختره بـcost_complexity_pruning_path + CV.", "", "",
       url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _DT, "monotonic_cst", None, "None | array من {-1, 0, 1}",
       "قيود رتابة لكل خاصية (1 متزايد، −1 متناقص).", "—", "—",
       "لا يدعم التصنيف متعدد الفئات؛ مدعوم مع القيم المفقودة منذ 1.9.", "", "",
       url=SK + "sklearn.tree.DecisionTreeClassifier.html"),
    HP("scikit-learn", _RF, "n_estimators", 100, "int ≥ 1", "عدد الأشجار.", "تقدير أثبت (لا Overfitting إضافي).",
       "تقلب أكبر.", "العائد يتناقص بعد عدد معين.", "خطي في عدد الأشجار.", "زيادته لا تسبب Overfitting.",
       url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    HP("scikit-learn", _RF, "max_features", "sqrt", "'sqrt' | 'log2' | None | int | float",
       "الخصائص المرشحة عند كل تقسيم — مصدر عدم ترابط الأشجار.", "أشجار أقوى لكن مترابطة.", "أشجار أضعف وأقل ترابطًا.",
       "للانحدار الافتراضي 1.0 (كل الخصائص).", "", "", url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    HP("scikit-learn", _RF, "bootstrap", True, "bool", "عينات Bootstrap لكل شجرة.", "—", "—", "مطلوب لـoob_score.",
       "", "", url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    HP("scikit-learn", _RF, "oob_score", False, "bool | callable", "تقدير الأداء على العينات خارج الحقيبة.", "—", "—",
       "بديل رخيص لـCV لكنه ليس بديلًا عن Test.", "تكلفة صغيرة إضافية.", "",
       url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    HP("scikit-learn", _RF, "max_samples", None, "None | int | float في (0, 1]", "حجم عينة Bootstrap.",
       "—", "أشجار أكثر تنوعًا وأسرع.", "", "", "", url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    # ------------------------------------------------ boosting (sklearn)
    HP("scikit-learn", _GB, "learning_rate", 0.1, "float ≥ 0", "الانكماش: وزن مساهمة كل شجرة.",
       "تعلّم أسرع وخطر Overfitting.", "يحتاج أشجارًا أكثر لكنه يعمّم أفضل غالبًا.",
       "مقايضة قوية مع n_estimators.", "", "", url=SK + "sklearn.ensemble.GradientBoostingClassifier.html"),
    HP("scikit-learn", _GB, "n_estimators", 100, "int ≥ 1", "عدد مراحل التعزيز.", "ملاءمة أدق ثم Overfitting.",
       "Underfitting.", "استخدم n_iter_no_change للتوقف المبكر.", "خطي في المراحل.", "كثير ⇒ Overfitting (عكس RF).",
       url=SK + "sklearn.ensemble.GradientBoostingClassifier.html"),
    HP("scikit-learn", _GB, "subsample", 1.0, "float في (0, 1]", "نسبة العينات لكل شجرة (Stochastic GB).", "—",
       "عشوائية وتنظيم أكثر.", "", "أقل ⇒ أسرع.", "", url=SK + "sklearn.ensemble.GradientBoostingClassifier.html"),
    HP("scikit-learn", _GB, "max_depth", 3, "None | int", "عمق كل شجرة (رتبة التفاعلات).", "تفاعلات أعلى.", "أبسط.",
       "", "", "", url=SK + "sklearn.ensemble.GradientBoostingClassifier.html"),
    HP("scikit-learn", _GB, "criterion", "deprecated", "مهجور في 1.9", "كان يختار بين friedman_mse وsquared_error (متطابقان).",
       "—", "—", "", "", "", url=SK + "sklearn.ensemble.GradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "learning_rate", 0.1, "float > 0", "الانكماش.", "أسرع وأخطر.", "أبطأ وأكثر أمانًا.",
       "مع max_iter والتوقف المبكر.", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "max_iter", 100, "int ≥ 1", "عدد أشجار التعزيز (للتصنيف الثنائي).", "—", "—",
       "early_stopping='auto' يفعَّل تلقائيًا حين n > 10000.", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "max_leaf_nodes", 31, "None | int ≥ 2", "أقصى أوراق لكل شجرة.", "تعقيد أكبر.", "أبسط.",
       "مع max_depth وmin_samples_leaf.", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "min_samples_leaf", 20, "int ≥ 1", "أقل عينات في الورقة.", "تنظيم أقوى.", "أدق وأخطر.",
       "", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "l2_regularization", 0.0, "float ≥ 0", "عقوبة L2 على قيم الأوراق.", "تنظيم أقوى.", "—", "",
       "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "max_bins", 255, "int في [2, 255]", "عدد الـbins للقيم غير المفقودة (+1 للمفقودة).",
       "دقة أعلى في مواقع التقسيم.", "تنظيم وسرعة.", "", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "early_stopping", "auto", "'auto' | bool", "التوقف المبكر على validation_fraction.", "—",
       "—", "'auto' ⇒ مفعّل إن كان n > 10000.", "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", _HGB, "categorical_features", "from_dtype", "'from_dtype' | mask | أسماء | None",
       "الخصائص الفئوية المعالجة أصليًا (أعمدة pandas category افتراضيًا).", "—", "—", "حتى max_bins فئة لكل خاصية.",
       "", "", url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    HP("scikit-learn", "sklearn.ensemble.AdaBoostClassifier", "n_estimators", 50, "int ≥ 1", "أقصى عدد للمتعلمين الضعفاء.",
       "—", "—", "مع learning_rate.", "", "", url=SK + "sklearn.ensemble.AdaBoostClassifier.html"),
    HP("scikit-learn", "sklearn.ensemble.AdaBoostClassifier", "learning_rate", 1.0, "float > 0", "وزن كل متعلم.",
       "—", "—", "", "", "", url=SK + "sklearn.ensemble.AdaBoostClassifier.html"),
    # ------------------------------------------------ XGBoost (resolved from booster config)
    HP("xgboost", _XGB, "learning_rate", None, "float في [0, 1] (alias eta)", "الانكماش لكل جولة.",
       "أسرع وأخطر.", "يحتاج جولات أكثر.", "مع n_estimators والتوقف المبكر.", "", "", resolved="0.3",
       note="الـwrapper يمرر None ⇒ القيمة الفعلية من config الـbooster هي eta=0.3.", url=XGB),
    HP("xgboost", _XGB, "n_estimators", None, "int ≥ 1", "عدد جولات التعزيز.", "—", "—", "", "", "", resolved="100", url=XGB),
    HP("xgboost", _XGB, "max_depth", None, "int ≥ 0 (0 = بلا حد مع lossguide)", "أقصى عمق.", "تفاعلات أعمق.", "أبسط.",
       "مع grow_policy='lossguide' استخدم max_leaves.", "", "", resolved="6", url=XGB),
    HP("xgboost", _XGB, "min_child_weight", None, "float ≥ 0", "أقل مجموع هيسيان في الابن.", "تنظيم أقوى.", "—",
       "للانحدار بالمربعات = عدد العينات.", "", "", resolved="1", url=XGB),
    HP("xgboost", _XGB, "gamma", None, "float ≥ 0 (alias min_split_loss)", "أقل انخفاض في الخسارة لقبول التقسيم.",
       "أشجار أصغر.", "—", "", "", "", resolved="0", url=XGB),
    HP("xgboost", _XGB, "subsample", None, "float في (0, 1]", "نسبة الصفوف لكل شجرة.", "—", "تنظيم عشوائي.", "", "",
       "", resolved="1", url=XGB),
    HP("xgboost", _XGB, "colsample_bytree", None, "float في (0, 1]", "نسبة الأعمدة لكل شجرة.", "—", "تنوع وتنظيم.",
       "تتضاعف مع bylevel وbynode.", "", "", resolved="1", url=XGB),
    HP("xgboost", _XGB, "colsample_bylevel", None, "float في (0, 1]", "نسبة الأعمدة لكل مستوى عمق.", "—", "—", "", "", "",
       resolved="1", url=XGB),
    HP("xgboost", _XGB, "colsample_bynode", None, "float في (0, 1]", "نسبة الأعمدة لكل عقدة (تشبه max_features في RF).",
       "—", "—", "", "", "", resolved="1", url=XGB),
    HP("xgboost", _XGB, "reg_alpha", None, "float ≥ 0 (alias alpha)", "عقوبة L1 على أوزان الأوراق.", "أوراق صفرية أكثر.",
       "—", "", "", "", resolved="0", url=XGB),
    HP("xgboost", _XGB, "reg_lambda", None, "float ≥ 0 (alias lambda)", "عقوبة L2 على أوزان الأوراق (تظهر في مقام Gain).",
       "تنظيم أقوى.", "—", "", "", "", resolved="1", url=XGB),
    HP("xgboost", _XGB, "tree_method", None, "'auto','exact','approx','hist'", "خوارزمية بناء الأشجار.", "—", "—",
       "", "hist الأسرع ويدعم GPU عبر device='cuda'.", "", resolved="hist (grow_quantile_histmaker)", url=XGB),
    HP("xgboost", _XGB, "early_stopping_rounds", None, "None | int", "أوقف إن لم تتحسن مجموعة eval_set خلال k جولة.",
       "—", "—", "يُمرَّر للـconstructor منذ 2.0 مع eval_set في fit().", "", "", url=XGB),
    # ------------------------------------------------ LightGBM (sklearn API names, native aliases in note)
    HP("lightgbm", _LGB, "num_leaves", 31, "int في [2, 131072]", "أقصى أوراق للشجرة (المتحكم الأساسي في التعقيد مع Leaf-wise).",
       "تعقيد أعلى بسرعة.", "أبسط.", "اجعله < 2^max_depth إن حددت max_depth.", "", "كبير ⇒ Overfitting.", url=LGB),
    HP("lightgbm", _LGB, "max_depth", -1, "int (≤ 0 = بلا حد)", "أقصى عمق.", "—", "حماية من الأشجار العميقة جدًا.", "",
       "", "", url=LGB),
    HP("lightgbm", _LGB, "learning_rate", 0.1, "float > 0", "الانكماش.", "—", "—", "مع n_estimators.", "", "", url=LGB),
    HP("lightgbm", _LGB, "n_estimators", 100, "int ≥ 0 (native: num_iterations)", "عدد جولات التعزيز.", "—", "—",
       "", "", "", url=LGB),
    HP("lightgbm", _LGB, "min_child_samples", 20, "int ≥ 0 (native: min_data_in_leaf)", "أقل عينات في الورقة.",
       "تنظيم أقوى.", "—", "قيمة كبيرة تمنع أوراقًا صغيرة جدًا مع num_leaves كبير.", "", "", url=LGB),
    HP("lightgbm", _LGB, "colsample_bytree", 1.0, "float في (0, 1] (native: feature_fraction)", "نسبة الخصائص لكل شجرة.",
       "—", "تنوع وسرعة.", "", "", "", url=LGB),
    HP("lightgbm", _LGB, "subsample", 1.0, "float في (0, 1] (native: bagging_fraction)", "نسبة الصفوف (Bagging).",
       "—", "تنظيم وسرعة.", "لا يعمل إلا مع subsample_freq > 0.", "", "", url=LGB),
    HP("lightgbm", _LGB, "subsample_freq", 0, "int ≥ 0 (native: bagging_freq)", "إعادة أخذ العينة كل k جولة (0 = معطّل).",
       "—", "—", "", "", "", url=LGB),
    HP("lightgbm", _LGB, "reg_alpha", 0.0, "float ≥ 0 (native: lambda_l1)", "L1 على أوزان الأوراق.", "—", "—", "", "", "",
       url=LGB),
    HP("lightgbm", _LGB, "reg_lambda", 0.0, "float ≥ 0 (native: lambda_l2)", "L2 على أوزان الأوراق.", "—", "—", "", "", "",
       url=LGB),
    HP("lightgbm", _LGB, "min_split_gain", 0.0, "float ≥ 0 (native: min_gain_to_split)", "أقل Gain لقبول التقسيم.",
       "—", "—", "", "", "", url=LGB),
    # ------------------------------------------------ CatBoost (resolved from get_all_params)
    HP("catboost", _CB, "iterations", None, "int ≥ 1", "عدد الأشجار.", "—", "—", "", "", "", resolved="1000", url=CB),
    HP("catboost", _CB, "learning_rate", None, "float > 0", "الانكماش.", "—", "—",
       "إن لم يُحدَّد يُختار آليًا حسب حجم البيانات وiterations.", "", "", resolved="auto (data-dependent)", url=CB),
    HP("catboost", _CB, "depth", None, "int في [1, 16]", "عمق الأشجار المتناظرة (Oblivious).", "تفاعلات أعمق.", "أبسط.",
       "", "الزمن ~ 2^depth.", "", resolved="6", url=CB),
    HP("catboost", _CB, "l2_leaf_reg", None, "float ≥ 0", "عقوبة L2 على قيم الأوراق.", "تنظيم أقوى.", "—", "", "", "",
       resolved="3", url=CB),
    HP("catboost", _CB, "random_strength", None, "float ≥ 0", "ضجيج يضاف لدرجات التقسيم لمنع Overfitting.", "—", "—", "", "",
       "", resolved="1", url=CB),
    HP("catboost", _CB, "bagging_temperature", None, "float ≥ 0", "شدة Bayesian bootstrap (0 = بلا عشوائية).", "—", "—",
       "فعّال فقط مع bootstrap_type='Bayesian' (الافتراضي على CPU هو MVS).", "", "", resolved="1 (Bayesian only)", url=CB),
    HP("catboost", _CB, "border_count", None, "int في [1, 65535]", "عدد حدود التقطيع للخصائص العددية.", "دقة أعلى.",
       "أسرع.", "", "", "", resolved="254 (CPU)", url=CB),
    HP("catboost", _CB, "one_hot_max_size", None, "int ≥ 1", "الفئات بعدد قيم ≤ هذا تُرمَّز One-hot، والباقي بـTarget statistics.",
       "One-hot لفئات أكثر.", "Target statistics أكثر.", "", "", "", resolved="2 (CPU، يعتمد على الإعدادات)", url=CB),
    HP("catboost", _CB, "early_stopping_rounds", None, "None | int", "أوقف إن لم يتحسن eval_set خلال k جولة (يضبط od_type='Iter').",
       "—", "—", "", "", "", url=CB),
    # ------------------------------------------------ unsupervised
    HP("scikit-learn", "sklearn.cluster.KMeans", "n_clusters", 8, "int ≥ 1", "عدد العناقيد k.", "Inertia أقل دائمًا.",
       "—", "لا تختره بـInertia وحدها: استخدم Silhouette ومعرفة المجال.", "", "", url=SK + "sklearn.cluster.KMeans.html"),
    HP("scikit-learn", "sklearn.cluster.KMeans", "init", "k-means++", "'k-means++' | 'random' | array", "طريقة البدء.",
       "—", "—", "k-means++ يقلل البدايات السيئة.", "", "", url=SK + "sklearn.cluster.KMeans.html"),
    HP("scikit-learn", "sklearn.cluster.KMeans", "n_init", "auto", "'auto' | int", "عدد مرات التشغيل من بدايات مختلفة.",
       "حل أثبت.", "—", "'auto' = 1 مع k-means++ و10 مع random.", "خطي.", "", url=SK + "sklearn.cluster.KMeans.html"),
    HP("scikit-learn", "sklearn.cluster.DBSCAN", "eps", 0.5, "float > 0", "نصف قطر الجوار.", "عناقيد أقل وأكبر، ضجيج أقل.",
       "ضجيج أكثر.", "اختره بمنحنى k-distance مع k = min_samples.", "", "", url=SK + "sklearn.cluster.DBSCAN.html"),
    HP("scikit-learn", "sklearn.cluster.DBSCAN", "min_samples", 5, "int ≥ 1", "أقل عدد جيران (مع النقطة) لنقطة مركزية.",
       "تعريف أشد للكثافة.", "—", "", "", "", url=SK + "sklearn.cluster.DBSCAN.html"),
    HP("scikit-learn", "sklearn.cluster.HDBSCAN", "min_cluster_size", 5, "int ≥ 2", "أصغر حجم لعنقود.", "عناقيد أقل.", "—",
       "المعامل الأهم في HDBSCAN.", "", "", url=SK + "sklearn.cluster.HDBSCAN.html"),
    HP("scikit-learn", "sklearn.cluster.HDBSCAN", "min_samples", None, "None | int", "يتحكم في تحفّظ تعريف الكثافة.",
       "ضجيج أكثر.", "—", "None ⇒ يساوي min_cluster_size.", "", "", url=SK + "sklearn.cluster.HDBSCAN.html"),
    HP("scikit-learn", "sklearn.manifold.TSNE", "perplexity", 30.0, "float > 0 (< n)", "عدد الجيران الفعّال تقريبًا.",
       "بنية أكثر شمولًا.", "بنية محلية جدًا.", "جرّب عدة قيم؛ الشكل يتغير.", "", "", url=SK + "sklearn.manifold.TSNE.html"),
    HP("scikit-learn", "sklearn.manifold.TSNE", "learning_rate", "auto", "'auto' | float", "خطوة التحسين.", "—", "—",
       "'auto' = max(n/early_exaggeration/4, 50).", "", "", url=SK + "sklearn.manifold.TSNE.html"),
    HP("scikit-learn", "sklearn.naive_bayes.GaussianNB", "var_smoothing", 1e-09, "float ≥ 0",
       "جزء من أكبر تباين يُضاف لكل التباينات للاستقرار.", "حدود أنعم.", "—", "", "", "",
       url=SK + "sklearn.naive_bayes.GaussianNB.html"),
    HP("scikit-learn", "sklearn.naive_bayes.MultinomialNB", "alpha", 1.0, "float ≥ 0",
       "تنعيم Laplace/Lidstone للعدّ.", "تنعيم أقوى.", "ثقة أكبر في العدّ الملاحَظ.", "", "", "",
       url=SK + "sklearn.naive_bayes.MultinomialNB.html"),
    HP("scikit-learn", "sklearn.discriminant_analysis.LinearDiscriminantAnalysis", "shrinkage", None,
       "None | 'auto' | float في [0, 1]", "انكماش مصفوفة التغاير (Ledoit-Wolf مع 'auto').", "—", "—",
       "يعمل مع solver='lsqr' أو 'eigen' لا 'svd'.", "", "",
       url=SK + "sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html"),
    HP("scikit-learn", "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis", "reg_param", 0.0,
       "float في [0, 1]", "تنظيم تغاير كل فئة نحو الهوية.", "أقرب لـNB.", "—",
       "منذ 1.8 أضيفت solver وshrinkage لـQDA.", "", "",
       url=SK + "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html"),
    HP("scikit-learn", "sklearn.ensemble.IsolationForest", "contamination", "auto", "'auto' | float في (0, 0.5]",
       "النسبة المتوقعة للشذوذ (تحدد عتبة predict فقط).", "—", "—", "لا تغيّر score_samples.", "", "",
       url=SK + "sklearn.ensemble.IsolationForest.html"),
    HP("scikit-learn", "sklearn.ensemble.IsolationForest", "max_samples", "auto", "'auto' | int | float",
       "عينات كل شجرة ('auto' = min(256, n)).", "—", "—", "", "", "", url=SK + "sklearn.ensemble.IsolationForest.html"),
    # ------------------------------------------------ evaluation meta-estimators
    HP("scikit-learn", "sklearn.calibration.CalibratedClassifierCV", "method", "sigmoid", "'sigmoid','isotonic','temperature'",
       "طريقة المعايرة: Platt، أو رتيبة غير معلمية، أو Temperature scaling (أضيفت في 1.8).", "—", "—",
       "isotonic يحتاج بيانات أكثر (> ~1000) وإلا يفرط في الملاءمة.", "", "",
       url=SK + "sklearn.calibration.CalibratedClassifierCV.html"),
    HP("scikit-learn", "sklearn.calibration.CalibratedClassifierCV", "ensemble", "auto", "'auto' | bool",
       "متوسط مصنفات الطيات أم مصنّف واحد على كل البيانات.", "—", "—", "ensemble=False هو بديل SVC(probability=True).",
       "", "", url=SK + "sklearn.calibration.CalibratedClassifierCV.html"),
    HP("scikit-learn", "sklearn.model_selection.TunedThresholdClassifierCV", "scoring", "balanced_accuracy",
       "str | callable (scorer)", "المقياس الذي تُعظّمه العتبة.", "—", "—", "مرّر make_scorer لتكلفة أعمال.", "", "",
       url=SK + "sklearn.model_selection.TunedThresholdClassifierCV.html"),
    HP("scikit-learn", "sklearn.model_selection.TunedThresholdClassifierCV", "thresholds", 100, "int | array",
       "عدد العتبات المرشحة أو قائمتها.", "دقة أعلى.", "—", "", "", "",
       url=SK + "sklearn.model_selection.TunedThresholdClassifierCV.html"),
    # ------------------------------------------------ DoubleML
    HP("DoubleML", "doubleml.DoubleMLPLR", "n_folds", 5, "int ≥ 2", "عدد طيات Cross-fitting.", "إزعاج مدرَّب على بيانات أكثر.",
       "—", "مع n_rep.", "خطي في الطيات.", "", url=DML),
    HP("DoubleML", "doubleml.DoubleMLPLR", "n_rep", 1, "int ≥ 1", "تكرار التقسيم العشوائي؛ التجميع بالوسيط.",
       "نتيجة أقل اعتمادًا على تقسيم واحد.", "—", "", "خطي.", "", url=DML),
    HP("DoubleML", "doubleml.DoubleMLPLR", "score", "partialling out", "'partialling out' | 'IV-type' | callable",
       "الدرجة المتعامدة.", "—", "—", "'IV-type' يتطلب ml_g.", "", "", url=DML),
    HP("DoubleML", "doubleml.DoubleMLIRM", "score", "ATE", "'ATE' | 'ATTE' | callable", "المعلمة المستهدفة.", "—", "—", "",
       "", "", url=DML),
    HP("DoubleML", "doubleml.DoubleMLIRM", "trimming_threshold", 0.01, "float في [0, 0.5)", "قص درجات الميل إلى [t, 1 − t].",
       "استقرار أكبر وتحيز محتمل.", "تباين أكبر مع درجات متطرفة.", "trimming_rule='truncate'.", "", "", url=DML),
    # ------------------------------------------------ scikit-learn: class weights in ensembles
    HP("scikit-learn", _RF, "class_weight", None, "None | 'balanced' | 'balanced_subsample' | dict",
       "أوزان الفئات في معيار التقسيم وفي أوراق الأشجار.", "—", "—",
       "'balanced_subsample' يعيد حساب الأوزان على عينة Bootstrap لكل شجرة.", "", "يرفع الاحتمالات للفئة النادرة (يفقد المعايرة).",
       url=SK + "sklearn.ensemble.RandomForestClassifier.html"),
    HP("scikit-learn", _HGB, "class_weight", None, "None | 'balanced' | dict", "يضرب خسارة كل مثال في وزن فئته.",
       "—", "—", "يتفاعل مع early_stopping (الوزن يدخل في درجة التحقق الداخلية).", "", "يغيّر الاحتمالات.",
       url=SK + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    # ------------------------------------------------ imbalanced-learn: samplers
    HP("imbalanced-learn", "imblearn.over_sampling.SMOTE", "sampling_strategy", "auto",
       "'auto' | float (نسبة الأقلية/الأغلبية بعد العيّنة، ثنائي فقط) | dict | 'minority' | 'not majority' | 'all'",
       "الهدف بعد إعادة العيّنة؛ 'auto' = 'not majority' ⇒ موازنة كاملة.", "float أكبر ⇒ نقاط اصطناعية أكثر.",
       "float أصغر ⇒ موازنة جزئية (غالبًا أفضل من 1:1).", "يتفاعل مع العتبة: موازنة أكبر ⇒ احتمالات أعلى للأقلية.",
       "الزمن ∝ عدد النقاط المولَّدة.", "موازنة كاملة مع تداخل كبير ⇒ نقاط في منطقة الأغلبية.", url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.SMOTE", "k_neighbors", 5, "int ≥ 1 أو كائن NearestNeighbors",
       "عدد جيران الأقلية الذين يُختار منهم شريك الاستيفاء.", "استيفاء أبعد ⇒ تنوع أكبر وخطر اختراق منطقة الأغلبية.",
       "استيفاء محلي جدًا ⇒ قريب من التكرار.", "يجب أن يكون أقل من عدد أمثلة الأقلية في طية التدريب.", "", "",
       url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.BorderlineSMOTE", "kind", "borderline-1",
       "'borderline-1' | 'borderline-2'", "1: الشريك من الأقلية فقط؛ 2: قد يكون من أي فئة (بخطوة أقصر).", "—", "—",
       "يولّد فقط من أمثلة «DANGER» (أكثر من نصف جيرانها من الأغلبية).", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.BorderlineSMOTE", "m_neighbors", 10, "int ≥ 1",
       "عدد الجيران (من كل الفئات) لتصنيف المثال: SAFE أو DANGER أو NOISE.", "تصنيف أكثر سلاسة.", "تصنيف أكثر محلية.",
       "", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.SVMSMOTE", "out_step", 0.5, "float",
       "حجم خطوة الاستقراء إلى الخارج حين يكون مثال الدعم محاطًا بالأقلية.", "استقراء أبعد.", "استقراء أقرب.",
       "يستخدم متجهات الدعم من SVC لتحديد منطقة الحدود.", "أبطأ بسبب تدريب SVM.", "", url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.ADASYN", "n_neighbors", 5, "int ≥ 1",
       "الجيران المستخدمون لقياس «صعوبة» كل مثال أقلية (نسبة الأغلبية بين جيرانه).", "تقدير أنعم للصعوبة.",
       "تقدير أكثر ضجيجًا.", "يولّد نقاطًا أكثر حول الأمثلة الصعبة ⇒ حساس لضجيج التسميات.", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.over_sampling.RandomOverSampler", "shrinkage", None, "None | float ≥ 0 | dict",
       "None = تكرار حرفي؛ قيمة موجبة = Smoothed bootstrap (إضافة ضجيج Gaussian مقياسه ∝ shrinkage).",
       "نقاط أكثر تشتتًا.", "أقرب إلى التكرار.", "", "", "التكرار الحرفي قد يدفع نحو حفظ الأمثلة.", url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.RandomUnderSampler", "replacement", False, "bool",
       "السحب من الأغلبية مع الإرجاع أو بدونه.", "—", "—", "", "أسرع خيار على الإطلاق.",
       "يرمي معلومات الأغلبية ⇒ تباين أعلى.", url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.NearMiss", "version", 1, "1 | 2 | 3",
       "1: أقرب متوسط مسافة لأقرب أمثلة أقلية؛ 2: لأبعدها؛ 3: جيران كل مثال أقلية.", "—", "—",
       "النسخ 1 و2 حساسة جدًا للضجيج والقيم الشاذة.", "", "قد يحذف بنية الأغلبية بعيدًا عن الحد.", url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.NearMiss", "n_neighbors", 3, "int ≥ 1",
       "عدد جيران الأقلية في حساب متوسط المسافة.", "اختيار أنعم.", "اختيار أكثر محلية.", "", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.EditedNearestNeighbours", "n_neighbors", 3, "int ≥ 1",
       "حجم الجوار المستخدم لتحديد الأمثلة «المخالفة» لجيرانها.", "تنظيف أكثر عدوانية.", "تنظيف أقل.",
       "", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.EditedNearestNeighbours", "kind_sel", "all", "'all' | 'mode'",
       "'all': يُحذف المثال إن خالفه أي جار؛ 'mode': إن خالفته الأغلبية.", "—", "—", "'all' أكثر حذفًا.", "", "",
       url=IMB),
    HP("imbalanced-learn", "imblearn.under_sampling.NeighbourhoodCleaningRule", "threshold_cleaning", 0.5,
       "float في [0, 1]", "عتبة نسبة الفئة التي تُنظَّف في المرحلة الثانية (Laurikkala).", "تنظيف فئات أقل.",
       "تنظيف فئات أكثر.", "", "", "", url=IMB),
    # ------------------------------------------------ imbalanced-learn: ensembles
    HP("imbalanced-learn", "imblearn.ensemble.BalancedRandomForestClassifier", "sampling_strategy", "all",
       "'all' | 'auto' | float | dict", "كيف يُعاد توازن عينة كل شجرة؛ 'all' = كل الفئات تُسحب (سلوك Chen et al., 2004).",
       "—", "—", "القيمة الافتراضية أصبحت 'all' في 0.13.", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.BalancedRandomForestClassifier", "replacement", True, "bool",
       "سحب مع الإرجاع (Bootstrap متوازن لكل شجرة).", "—", "—", "الافتراضي True منذ 0.13.", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.BalancedRandomForestClassifier", "bootstrap", False, "bool",
       "Bootstrap إضافي من scikit-learn فوق العيّنة المتوازنة.", "—", "—",
       "False افتراضيًا منذ 0.13 لأن replacement=True يؤدي دور Bootstrap.", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.BalancedRandomForestClassifier", "n_estimators", 100, "int ≥ 1",
       "عدد الأشجار.", "تباين أقل وزمن أطول.", "تباين أكبر.", "", "خطي في n_estimators.", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.BalancedBaggingClassifier", "n_estimators", 10, "int ≥ 1",
       "عدد الحقائب؛ كل حقيبة تُعاد موازنتها بـsampler.", "تباين أقل.", "تباين أكبر.", "", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.BalancedBaggingClassifier", "sampler", None, "None | sampler",
       "None = RandomUnderSampler؛ أي Sampler آخر (مثل SMOTE) يعطي «SMOTEBagging».", "—", "—", "", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.EasyEnsembleClassifier", "n_estimators", 10, "int ≥ 1",
       "عدد العيّنات المتوازنة؛ على كل منها AdaBoost كامل.", "تغطية أكبر للأغلبية.", "تغطية أقل.",
       "estimator=None ⇒ AdaBoostClassifier افتراضي.", "مكلف: n_estimators × مراحل AdaBoost.", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.RUSBoostClassifier", "n_estimators", 50, "int ≥ 1",
       "الحد الأقصى لمراحل Boosting (مع RandomUnderSampler قبل كل مرحلة).", "مراحل أكثر.", "مراحل أقل.", "", "", "",
       url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.RUSBoostClassifier", "learning_rate", 1.0, "float > 0",
       "الانكماش لكل مرحلة.", "تعلم أسرع وأقل استقرارًا.", "استقرار أكبر يحتاج مراحل أكثر.",
       "مع الجذوع الافتراضية (depth 1) قد يرفع خطأ «worse than random» على بيانات متداخلة.", "", "", url=IMB),
    HP("imbalanced-learn", "imblearn.ensemble.RUSBoostClassifier", "estimator", None, "None | classifier",
       "None ⇒ DecisionTreeClassifier(max_depth=1).", "—", "—", "شجرة أعمق (3) مع learning_rate أصغر أكثر استقرارًا.",
       "", "", url=IMB),
    # ------------------------------------------------ Optuna
    HP("optuna", "optuna.samplers.TPESampler", "n_startup_trials", 10, "int ≥ 0", "محاولات عشوائية قبل تفعيل TPE.",
       "استكشاف أولي أكثر.", "—", "", "", "", note="في Optuna 5.0 أصبح TPE متعدد المتغيرات مع constant liar افتراضيًا.",
       url=OPT),
    HP("optuna", "optuna.samplers.TPESampler", "constant_liar", True, "bool", "يمنع المحاولات المتوازية من تكرار نفس المنطقة.",
       "—", "—", "", "", "", note="القيمة الافتراضية True في 5.0.0 (تحقق من التوقيع).", url=OPT),
)


def by_algorithm() -> dict[str, list[HP]]:
    out: dict[str, list[HP]] = {}
    for h in HYPERPARAMETERS:
        out.setdefault(h.algo, []).append(h)
    return out


def for_estimator(algo: str) -> list[HP]:
    return [h for h in HYPERPARAMETERS if h.algo == algo]
