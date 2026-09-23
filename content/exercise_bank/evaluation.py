"""Exercises: validation and evaluation."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "cross_validation": {
        "beginner": _ex("cross_val_score", "احسب 5-fold ROC-AUC لـLogisticRegression مع StratifiedKFold.", "cv=StratifiedKFold(5, shuffle=True).",
                        "```python\ncross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y,\n"
                        "                cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')\n```"),
        "intermediate": _ex("Nested CV", "اكتب Nested CV لضبط C وgamma في SVC.", "GridSearchCV داخل cross_val_score.",
                            "```python\ninner = GridSearchCV(pipe, grid, cv=3)\nouter_scores = cross_val_score(inner, X, y, cv=5)\n```"),
        "research": _ex("تباين CV", "لماذا SD عبر الطيات يستهين بعدم اليقين؟", "تداخل التدريب.",
                        "الطيات تتشارك (k−2)/(k−1) من بيانات التدريب ⇒ الدرجات مترابطة؛ لا يوجد مقدِّر غير متحيز لتباين CV "
                        "(Bengio & Grandvalet, 2004)؛ تصحيح Nadeau–Bengio يضخّم التباين تقريبيًا."),
    },
    "classification_metrics": {
        "beginner": _ex("احسب", "TP=30, FP=20, FN=10, TN=940: Accuracy وPrecision وRecall وF1.", "من المصفوفة.",
                        "Accuracy = 0.97، Precision = 0.6، Recall = 0.75، F1 = 0.667."),
        "intermediate": _ex("Precision@k", "اكتب دالة Precision@k لأعلى k احتمالات.", "argsort.",
                            "```python\ndef precision_at_k(y, p, k):\n    top = np.argsort(-p)[:k]\n    return y[top].mean()\n```"),
        "research": _ex("AUC كإحصائية Mann–Whitney", "بيّن أن AUC = U/(n₁n₀).", "أزواج موجب-سالب.",
                        "AUC = نسبة الأزواج (موجب، سالب) التي يكون فيها score الموجب أكبر (+½ للتعادل) = U/(n₁n₀)."),
    },
    "regression_metrics": {
        "beginner": _ex("MAE وRMSE", "أخطاء (2, −2, 4): احسب MAE وRMSE.", "مطلق ومربع.", "MAE = 2.67، RMSE = √(24/3) = 2.83."),
        "intermediate": _ex("Pinball", "احسب pinball loss لـτ = 0.9 على تنبؤات مئين.", "mean_pinball_loss.",
                            "```python\nfrom sklearn.metrics import mean_pinball_loss\nmean_pinball_loss(y, q90, alpha=0.9)\n```"),
        "research": _ex("D²", "عرّف D² وعلاقته بـR².", "Deviance.",
                        "D² = 1 − Dev(y, ŷ)/Dev(y, ŷ_null)؛ مع Deviance التربيعي يساوي R². متاح d2_absolute_error_score وd2_pinball_score."),
    },
    "threshold_tuning": {
        "beginner": _ex("عتبة نظرية", "C_FP = 5، C_FN = 20. العتبة المثلى؟", "C_FP/(C_FP + C_FN).", "5/25 = 0.2."),
        "intermediate": _ex("Recall مقيَّد", "اختر أعلى عتبة تحقق Recall ≥ 0.9 على التحقق.", "precision_recall_curve.",
                            "```python\np, r, t = precision_recall_curve(y_val, proba)\nt[np.where(r[:-1] >= 0.9)[0].max()]\n```"),
        "research": _ex("سياسة", "لماذا العتبة في سياق عام (منح/رفض) قرار معياري؟", "من يتحمل الكلفة؟",
                        "لأن توزيع FP وFN على الأفراد والمجموعات ذو آثار أخلاقية وقانونية؛ يجب أن يُوثَّق ويُراجَع."),
    },
    "calibration": {
        "beginner": _ex("Reliability", "ارسم calibration_curve لـGaussianNB.", "n_bins=10.", "المنحنى على شكل S معكوس: ثقة مفرطة."),
        "intermediate": _ex("Temperature", "عاير RandomForest بـmethod='temperature' وقارن Brier.", "منذ 1.8.",
                            "```python\nCalibratedClassifierCV(RandomForestClassifier(), method='temperature', cv=5)\n```"),
        "research": _ex("ECE", "لماذا ECE يعتمد على عدد الصناديق؟", "تحيز التقدير.", "صناديق كثيرة ⇒ تباين؛ قليلة ⇒ تحيز يخفي سوء المعايرة؛ "
                        "استخدم صناديق Quantile وأبلغ عن العدد."),
    },
    "imbalanced": {
        "beginner": _ex("stratify", "قسّم بيانات 2% موجبة إلى تدريب/اختبار مع الحفاظ على النسبة.", "stratify=y.",
                        "train_test_split(X, y, stratify=y, test_size=0.3)."),
        "intermediate": _ex("imblearn Pipeline", "ابنِ Pipeline بـSMOTE وLogisticRegression وقيّمه بـCV.", "imblearn.pipeline.",
                            "```python\nfrom imblearn.pipeline import make_pipeline\ncross_val_score(make_pipeline(StandardScaler(), SMOTE(), LogisticRegression()), X, y, scoring='average_precision')\n```"),
        "research": _ex("Prior shift", "صحّح احتمالات نموذج دُرّب على 50/50 إلى انتشار 2%.", "صيغة تصحيح الأولوية.",
                        "p = p_s·(0.02/0.5) / [p_s·(0.02/0.5) + (1 − p_s)·(0.98/0.5)]."),
    },
    "feature_selection": {
        "beginner": _ex("SelectKBest", "اختر أفضل 10 خصائص بـf_regression داخل Pipeline.", "make_pipeline.",
                        "make_pipeline(SelectKBest(f_regression, k=10), LinearRegression())."),
        "intermediate": _ex("RFECV", "شغّل RFECV مع Ridge واطبع n_features_.", "RFECV(Ridge(), cv=5).", "عادة قريب من 10–15 على high_dim."),
        "research": _ex("Stability selection", "صف الإجراء.", "عينات فرعية.",
                        "كرر Lasso على 100 عينة نصفية بـα عشوائي ضمن مدى؛ احتفظ بالخصائص المختارة في > π_thr من المرات؛ يتحكم في "
                        "الاكتشافات الخاطئة (Meinshausen & Bühlmann, 2010)."),
    },
    "conformal_prediction": {
        "beginner": _ex("q̂", "درجات معايرة مفرزة (0.1, 0.3, 0.5, 0.9, 1.2, 2.0, 2.5, 3.1, 4.0) وα = 0.2. ما q̂؟", "⌈(n+1)(1−α)⌉.",
                        "n = 9 ⇒ ⌈10·0.8⌉ = 8 ⇒ q̂ = 3.1."),
        "intermediate": _ex("من الصفر", "اكتب دالة split_conformal(model, X_cal, y_cal, X_new, alpha).", "np.sort + ceil.",
                            "```python\ns = np.sort(np.abs(y_cal - model.predict(X_cal)))\nk = int(np.ceil((len(s) + 1) * (1 - alpha)))\n"
                            "q = s[min(k, len(s)) - 1]\np = model.predict(X_new); return p - q, p + q\n```"),
        "research": _ex("CQR", "كيف يجعل Conformalized Quantile Regression الفترات متكيفة؟", "انحدار مئيني.",
                        "درّب نموذجي مئين τ = α/2 و1 − α/2؛ الدرجة = max(q̂_lo − y, y − q̂_hi)؛ صحّح الحدين بمئين هذه الدرجات: عرض "
                        "متغير مع X وتغطية هامشية مضمونة."),
    },
}
