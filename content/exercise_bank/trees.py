"""Exercises: trees and ensembles."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "decision_trees": {
        "beginner": _ex("كسب المعلومات", "أب 10 (5A, 5B) ينقسم إلى يسار (4A, 1B) ويمين (1A, 4B). احسب Gini gain.",
                        "Gini(0.8/0.2) = 0.32.", "Gini(parent) = 0.5؛ الأبناء 0.32 لكل منهما ⇒ gain = 0.5 − 0.32 = 0.18."),
        "intermediate": _ex("مسار التقليم", "استخدم cost_complexity_pruning_path واختر ccp_alpha بـGridSearchCV.",
                            "ccp_alphas من المسار.", "```python\npath = DecisionTreeClassifier().cost_complexity_pruning_path(X, y)\n"
                            "GridSearchCV(DecisionTreeClassifier(), {'ccp_alpha': path.ccp_alphas}, cv=5).fit(X, y)\n```"),
        "research": _ex("Honest trees", "ما «الصدق» (Honesty) في Causal trees؟", "عينتان منفصلتان.",
                        "عينة لبناء التقسيمات وأخرى لتقدير قيم الأوراق؛ يمنع تحيز الاختيار ويعطي فترات ثقة صالحة (Athey & Imbens, 2016)."),
    },
    "ensemble_learning": {
        "beginner": _ex("تباين المتوسط", "10 نماذج σ² = 1 وρ = 0.5. تباين المتوسط؟", "ρσ² + (1−ρ)σ²/B.", "0.5 + 0.05 = 0.55."),
        "intermediate": _ex("Stacking", "ابنِ StackingClassifier من RF وSVC وLogistic مع meta = LogisticRegression.", "cv=5.",
                            "```python\nStackingClassifier([('rf', RandomForestClassifier()), ('svc', make_pipeline(StandardScaler(), SVC(probability=False)))],\n"
                            "                   final_estimator=LogisticRegression(), cv=5, stack_method='auto')\n```"),
        "research": _ex("Super Learner", "لماذا يُستخدم Super Learner كمتعلم إزعاج في DML/TMLE؟", "ضمان Oracle.",
                        "يؤدي تقاربيًا كأفضل تركيبة من المرشحين، فيقلل خطر اختيار متعلم إزعاج سيئ مسبقًا؛ يجب أن يُدرَّب داخل كل طية."),
    },
    "random_forest": {
        "beginner": _ex("OOB", "درّب RF مع oob_score=True وقارن OOB بدقة CV.", "oob_score_.", "متقاربان عادة (±1–2%)."),
        "intermediate": _ex("max_features", "ارسم Validation curve لـmax_features من 1 إلى p.", "validation_curve.",
                            "أفضل قيمة غالبًا بين √p وp/3؛ p كامل = Bagging (ارتباط أعلى)."),
        "research": _ex("حد Breiman", "فسّر PE ≤ ρ̄(1 − s²)/s².", "s القوة، ρ̄ الارتباط.",
                        "لخفض الخطأ: أشجار قوية (s كبير) وغير مترابطة (ρ̄ صغير)؛ max_features يقايض بينهما."),
    },
    "boosting": {
        "beginner": _ex("مرحلتان", "F₀ = ȳ = 5، y = (3, 7)، ν = 0.5، الشجرة الأولى تتنبأ بالبواقي تمامًا. F₁؟", "البواقي (−2, 2).",
                        "F₁ = 5 + 0.5·(−2, 2) = (4, 6)."),
        "intermediate": _ex("staged_predict", "ارسم خطأ الاختبار عبر المراحل لـGradientBoostingRegressor وحدد أفضل عدد.", "staged_predict.",
                            "```python\nerr = [mean_squared_error(y_te, p) for p in m.staged_predict(X_te)]\nnp.argmin(err) + 1\n```"),
        "research": _ex("AdaBoost كخسارة أسية", "بيّن أن AdaBoost يقلل مرحليًا الخسارة الأسية.", "Friedman, Hastie & Tibshirani (2000).",
                        "تقليل Σ exp(−yᵢ(F + αh)(xᵢ)) مرحليًا يعطي الأوزان wᵢ ∝ exp(−yᵢF(xᵢ)) وα = ½ log((1−err)/err) — تحديث AdaBoost نفسه."),
    },
    "hist_gradient_boosting": {
        "beginner": _ex("فئات أصلية", "حوّل الأعمدة الفئوية إلى category ودرّب HGB.", "astype('category').",
                        "```python\nX = X.astype({c: 'category' for c in cats}); HistGradientBoostingClassifier().fit(X, y)\n```"),
        "intermediate": _ex("رتابة", "أضف قيد رتابة موجبًا على credit_score وقارن الأداء.", "monotonic_cst.",
                            "الأداء غالبًا متقارب، والتنبؤ يصبح غير متناقص في درجة الائتمان — أسهل للتبرير."),
        "research": _ex("الطرح", "لماذا يكفي حساب مدرج ابن واحد؟", "الأب = الابن الأيسر + الأيمن.", "مدرج الأخ = مدرج الأب − مدرج الابن ⇒ نصف الحساب."),
    },
    "xgboost": {
        "beginner": _ex("وزن ورقة", "G = −6، H = 2، λ = 1. w*؟", "−G/(H + λ).", "6/3 = 2."),
        "intermediate": _ex("Early stopping", "درّب XGBClassifier بـ2000 جولة وearly_stopping_rounds=50 واطبع best_iteration.", "eval_set في fit.",
                            "```python\nm = XGBClassifier(n_estimators=2000, learning_rate=0.05, early_stopping_rounds=50)\n"
                            "m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False); m.best_iteration\n```"),
        "research": _ex("Gain", "اشتق صيغة Gain من الهدف التربيعي.", "عوّض w* في الهدف.",
                        "أفضل هدف لورقة = −½ G²/(H + λ) + γ؛ Gain = الهدف قبل التقسيم − مجموع الابنين."),
    },
    "lightgbm": {
        "beginner": _ex("num_leaves وmax_depth", "max_depth = 5. ما الحد الأقصى المعقول لـnum_leaves؟", "2^depth.", "32 (يُفضَّل أقل، مثل 20–31)."),
        "intermediate": _ex("fit حديث", "اكتب fit بـeval_X/eval_y ومراقبة التدريب والتحقق بأسماء.", "eval_names.",
                            "```python\nm.fit(X_tr, y_tr, eval_X=(X_tr, X_val), eval_y=(y_tr, y_val), eval_names=['train', 'valid'],\n"
                            "      callbacks=[lgb.early_stopping(50)])\n```"),
        "research": _ex("GOSS", "لماذا يُعاد وزن العينات ذات التدرج الصغير في GOSS؟", "تحيز التقدير.",
                        "لأن أخذ عينة منها فقط يقلل تمثيلها؛ الضرب في (1−a)/b يعيد تقدير الكسب غير متحيز تقريبًا."),
    },
    "catboost": {
        "beginner": _ex("cat_features", "درّب CatBoostClassifier على بيانات القروض مع cat_features.", "أسماء الأعمدة كنصوص.",
                        "```python\nCatBoostClassifier(verbose=0).fit(X, y, cat_features=['employment', 'region', 'city'])\n```"),
        "intermediate": _ex("One-hot أعمى", "قارن CV AUC مع One-hot مسبق مقابل cat_features.", "pd.get_dummies.",
                            "cat_features مساوٍ أو أفضل غالبًا، خاصة لـcity؛ One-hot يزيد الأبعاد ويضيع Target statistics."),
        "research": _ex("Prediction shift", "اربط Prediction shift في CatBoost بـCross-fitting في DML.", "استخدام الملاحظة نفسها.",
                        "كلاهما يتجنب استخدام الملاحظة في بناء الكمية التي ستُقيَّم عليها؛ Ordered boosting يستخدم الماضي فقط، وCross-fitting "
                        "يستخدم الطيات الأخرى فقط."),
    },
    "boosting_comparison": {
        "beginner": _ex("شغّل", "شغّل المقارنة بـ200 شجرة. هل الفروق أكبر من SD؟", "قارن الأعمدة.", "غالبًا لا: لا فائز واضح."),
        "intermediate": _ex("ضبط عادل", "اقترح ميزانية ضبط متساوية للمكتبات الأربع.", "Optuna بنفس عدد المحاولات.",
                            "50 محاولة Optuna لكل مكتبة على نفس الطيات مع فضاء بحث مكافئ (learning_rate، التعقيد، العشوائية، التنظيم)."),
        "research": _ex("Benchmark", "صمم تجربة مقارنة قابلة للنشر.", "عدة مجموعات بيانات.",
                        "≥ 20 مجموعة بيانات، Nested CV، ضبط متساوٍ، تقرير الزمن والذاكرة، واختبارات إحصائية على الرتب (Friedman + Nemenyi)."),
    },
}
