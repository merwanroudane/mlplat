"""Exercises: imbalanced classification track (the overview module 'imbalanced' lives in evaluation.py)."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "imb_nature": {
        "beginner": _ex("IR والانتشار", "10,000 مريض منهم 80 حالة. احسب π وIR وAccuracy لمصنف «سليم دائمًا».",
                        "IR = الأغلبية/الأقلية.", "π = 0.008، IR = 9920/80 = 124، Accuracy = 0.992."),
        "intermediate": _ex("عتبة بايز", "مع π = 0.05 وكثافتين Gaussian بتباين 1 ومتوسطين 0 و2، أين يقع الحد الأمثل لـAccuracy؟",
                            "f₁/f₀ = exp(2x − 2) ≥ 19.", "2x − 2 ≥ ln 19 ⇒ x ≥ 1 + ln(19)/2 ≈ 2.47 (بدل 1 في الحالة المتوازنة)."),
        "research": _ex("الجزر الصغيرة", "في المختبر ثبّت النسبة 5% وغيّر عدد العناقيد من 1 إلى 4 مع Logistic ثم kNN. فسّر الفرق.",
                        "نموذج خطي = حد واحد.",
                        "Logistic يرسم حدًا واحدًا فيفقد العناقيد الموزعة حول الأصل؛ kNN محلي فيلتقط الجزر إن كانت كثيفة بما "
                        "يكفي. الصعوبة من البنية لا من النسبة."),
    },
    "imb_metrics": {
        "beginner": _ex("MCC", "TP = 40، FN = 10، FP = 60، TN = 890: احسب Precision وRecall وMCC وAccuracy.", "طبّق الصيغ.",
                        "Precision = 0.40، Recall = 0.80، Accuracy = 0.93، MCC = (40·890 − 60·10)/√(100·50·950·900) ≈ 0.54."),
        "intermediate": _ex("Bootstrap طبقي", "اكتب دالة تعطي فترة 95% لـaverage_precision بإعادة سحب الموجبات والسالبة كلًا على حدة.",
                            "utils.imbalance.bootstrap_ci.",
                            "```python\nlo, hi = bootstrap_ci(y_test, p_test, average_precision_score, n_boot=1000)\n```"),
        "research": _ex("Precision عبر البيئات", "نموذج بـTPR = 0.8 وFPR = 0.05. احسب Precision عند π = 0.1 و0.01 و0.001.",
                        "Precision = TPR·π / (TPR·π + FPR·(1 − π)).", "0.64، 0.139، 0.0158 — النموذج نفسه، تجربة مستخدم مختلفة جذريًا."),
    },
    "imb_cost_sensitive": {
        "beginner": _ex("t*", "C_FP = 5 دولارات (مكالمة)، C_FN = 400 دولار (احتيال فائت). ما العتبة؟", "C_FP/(C_FP + C_FN).",
                        "5/405 ≈ 0.0123."),
        "intermediate": _ex("عتبة بتكلفة مخصصة", "اضبط العتبة بـTunedThresholdClassifierCV باستخدام make_scorer لدالة −التكلفة.",
                            "greater_is_better=True لدالة تُرجع سالب التكلفة.",
                            "```python\nscorer = make_scorer(neg_cost)\nTunedThresholdClassifierCV(pipe, scoring=scorer, cv=5).fit(X, y)\n```"),
        "research": _ex("إعادة الوزن = تغيير الأولوية", "أثبت أن تعظيم الأرجحية الموزونة لنموذج لوجستي صحيح المواصفة يعطي "
                        "odds_w = (w₁/w₀)·odds، وبالتالي إزاحة الحد الثابت بـlog(w₁/w₀).",
                        "الحل الأمثل للخسارة الموزونة في مجتمع لا نهائي: P_w(y=1|x) ∝ w₁ P(y=1|x).",
                        "الخسارة المتوقعة الموزونة تُعظَّم عند p_w(x) = w₁p(x)/(w₁p(x) + w₀(1 − p(x))) ⇒ logit p_w = logit p + "
                        "log(w₁/w₀). في النموذج الخطي ينتقل ذلك كله إلى الحد الثابت؛ مع التنظيم أو المواصفة الخاطئة تتغير "
                        "المعاملات الأخرى قليلًا."),
    },
    "imb_oversampling": {
        "beginner": _ex("Pipeline صحيح", "ابنِ Pipeline بـStandardScaler وSMOTE وLogisticRegression وقيّمه بـaverage_precision.",
                        "from imblearn.pipeline import make_pipeline.",
                        "```python\ncross_val_score(make_pipeline(StandardScaler(), SMOTE(random_state=0), LogisticRegression()),\n"
                        "                X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='average_precision')\n```"),
        "intermediate": _ex("sampling_strategy كمعامل فائق", "اضبط smote__sampling_strategy ∈ {0.1, 0.25, 0.5, 1.0} وk_neighbors "
                            "∈ {3, 5, 7} بـGridSearchCV.", "أسماء الخطوات: smote__k_neighbors.",
                            "```python\ngrid = {'smote__sampling_strategy': [0.1, 0.25, 0.5, 1.0], 'smote__k_neighbors': [3, 5, 7]}\n"
                            "GridSearchCV(pipe, grid, scoring='average_precision', cv=5).fit(X, y)\n```"),
        "research": _ex("SMOTE قبل CV", "قارن CV لـSMOTE مطبّق على كل البيانات قبل التقسيم مقابل داخل الـPipeline. فسّر الفرق.",
                        "النقاط المستوفاة تحمل معلومات أمثلة التحقق.",
                        "التطبيق المسبق يعطي درجات أعلى زيفًا لأن نقاطًا اصطناعية مشتقة من أمثلة التحقق تظهر في التدريب، "
                        "والتحقق نفسه يصبح متوازنًا فلا يعكس الانتشار الحقيقي."),
    },
    "imb_undersampling": {
        "beginner": _ex("Tomek", "طبّق TomekLinks واحسب كم مثالًا حُذف من الأغلبية.", "fit_resample ثم قارن np.bincount.",
                        "عادة عدد قليل — طريقة تنظيف لا موازنة."),
        "intermediate": _ex("NearMiss ضد العشوائي", "قارن RandomUnderSampler وNearMiss(version=1) وNearMiss(version=3) داخل CV "
                            "بـPR-AUC.", "استخدم utils.imbalance.make_sampler.",
                            "NearMiss-1 غالبًا الأسوأ في الترتيب لأنه يرمي بنية الأغلبية ويتأثر بالشواذ."),
        "research": _ex("تصحيح الأولوية بعد NearMiss", "لماذا لا يعيد تصحيح الأولوية البسيط المعايرةَ بعد NearMiss كما يفعل "
                        "بعد RandomUnderSampler؟", "هل P(x|y=0) باقٍ كما هو؟",
                        "NearMiss يختار أمثلة الأغلبية القريبة من الأقلية فيغيّر P(x|y=0)، بينما الصيغة تفترض تغيير الأولوية "
                        "فقط. الحل: معايرة تجريبية على بيانات بالانتشار الحقيقي."),
    },
    "imb_ensembles": {
        "beginner": _ex("BRF", "درّب BalancedRandomForestClassifier(n_estimators=200) واحسب mean(predict_proba[:, 1]) مقارنة بالانتشار.",
                        "random_state ثابت.", "المتوسط أعلى بكثير من الانتشار ⇒ الاحتمالات ليست مخاطر حقيقية."),
        "intermediate": _ex("SMOTEBagging", "ابنِ BalancedBaggingClassifier(sampler=SMOTE()) وقارنه بالنسخة الافتراضية.",
                            "sampler معامل في المُنشئ.",
                            "```python\nBalancedBaggingClassifier(sampler=SMOTE(random_state=0), n_estimators=50, random_state=0)\n```"),
        "research": _ex("مقارنة منصفة", "صمم تجربة تقارن BRF بـRandomForest + TunedThresholdClassifierCV بالتكلفة نفسها وعلى "
                        "الطيات نفسها. ما المقياس الحاسم؟",
                        "القرار النهائي هو التكلفة عند نقطة التشغيل.",
                        "Repeated stratified CV بطيات ثابتة؛ لكل نموذج اختر العتبة داخليًا بأقل تكلفة؛ قارن التكلفة المتوقعة "
                        "والفروق المقترنة عبر الطيات، إضافة إلى PR-AUC والمعايرة."),
    },
    "imb_probabilities": {
        "beginner": _ex("تصحيح", "p_s = 0.8 من نموذج 50/50 والانتشار الحقيقي 2%. احسب p.", "الصيغة مع π_s = 0.5.",
                        "p = 0.8·0.04 / (0.8·0.04 + 0.2·1.96) = 0.032/0.424 ≈ 0.075."),
        "intermediate": _ex("معايرة بعد SMOTE", "لف Pipeline بـSMOTE في CalibratedClassifierCV(method='sigmoid', cv=5) وقارن Brier.",
                            "CalibratedClassifierCV يقبل Pipeline كاملًا.",
                            "Brier يقترب من النموذج العادي، ويعود متوسط الاحتمالات قريبًا من الانتشار."),
        "research": _ex("EM", "برهن أن نقطة التقارب في خوارزمية Saerens تحقق π̂ = متوسط الاحتمالات المصححة، وناقش شرط الهوية.",
                        "نقطة ثابتة للتكرار.",
                        "عند التقارب π̂ = (1/N)Σ p_i(π̂) — شرط اتساق: متوسط الاحتمالات المصححة يساوي الانتشار المفترض. يتطلب "
                        "Label shift واحتمالات تدريب معايرة؛ مع نموذج ضعيف التمييز يصبح التقدير غير مستقر."),
    },
    "imb_multiclass_extreme": {
        "beginner": _ex("Macro مقابل Weighted", "احسب f1_score بـaverage='macro' و'weighted' لنموذج متعدد الفئات وفسّر الفرق.",
                        "f1_score(y, pred, average=...).", "Weighted قريب من أداء الفئة الكبيرة؛ Macro يكشف ضعف الفئات النادرة."),
        "intermediate": _ex("قاموس sampling_strategy", "استخدم RandomUnderSampler(sampling_strategy={0: 800}) ثم SMOTE(sampling_strategy="
                            "{2: 400}) في Pipeline واحد.", "imblearn.pipeline يقبل عدة samplers.",
                            "```python\nmake_pipeline(RandomUnderSampler(sampling_strategy={0: 800}), SMOTE(sampling_strategy={2: 400}),\n"
                            "              LogisticRegression(max_iter=3000))\n```"),
        "research": _ex("هجين", "أضف درجة IsolationForest (score_samples) كخاصية إلى مصنف موجّه داخل Pipeline دون تسرب.",
                        "محوّل مخصص يدرّب IsolationForest على طية التدريب فقط.",
                        "اكتب Transformer يدرّب IsolationForest في fit ويضيف −score_samples في transform؛ ضعه قبل المصنف "
                        "في Pipeline ليُدرَّب داخل كل طية."),
    },
    "imb_workflow": {
        "beginner": _ex("خط الأساس", "في مختبر المقارنة شغّل LogisticRegression مع «none» و«tuned threshold». ما الذي يتغير؟",
                        "PR-AUC مقابل Recall/Precision.", "PR-AUC ثابت (النموذج نفسه)؛ تتغير نقطة التشغيل فقط."),
        "intermediate": _ex("تقرير", "نزّل تقرير المختبر وأضف إليه مصفوفة التكلفة والعتبة المختارة وعدد الإنذارات لكل 1000 حالة.",
                            "الإنذارات = (TP + FP)/n × 1000.", "تقرير يمكن لصاحب القرار قراءته دون معرفة ML."),
        "research": _ex("Nested CV", "اكتب إجراء Nested CV يختار الاستراتيجية (none/class_weight/SMOTE) والعتبة في الحلقة الداخلية.",
                        "GridSearchCV على Pipeline فيه خطوة sampler قابلة للاستبدال ('passthrough').",
                        "```python\npipe = Pipeline([('scale', StandardScaler()), ('sampler', 'passthrough'), ('clf', LogisticRegression())])\n"
                        "grid = {'sampler': ['passthrough', SMOTE(), RandomUnderSampler()], 'clf__class_weight': [None, 'balanced']}\n"
                        "inner = TunedThresholdClassifierCV(GridSearchCV(pipe, grid, scoring='average_precision', cv=3), cv=3)\n"
                        "cross_val_score(inner, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))\n```\n"
                        "(Pipeline هنا من imblearn.pipeline)."),
    },
}
