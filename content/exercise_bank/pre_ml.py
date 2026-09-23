"""Exercises: Before ML."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "what_is_ml": {
        "beginner": _ex("عرّف T وE وP", "لمسألة «توقع تعثر المقترضين»، اكتب T وE وP في جملة لكل منها.",
                        "P يجب أن يكون قابلًا للقياس على بيانات جديدة.",
                        "T: التنبؤ باحتمال تعثر طلب قرض جديد خلال 12 شهرًا. E: طلبات سابقة معروف مآلها. P: ROC-AUC "
                        "وRecall عند نسبة رفض 10% على طلبات الأشهر الستة الأخيرة (غير المستخدمة في التدريب)."),
        "intermediate": _ex("من قاعدة إلى نموذج", "بنك يستخدم قاعدة «ارفض إن كان الدين/الدخل > 0.5». كيف تحوّلها إلى مشروع ML "
                            "وتقيس هل يتفوق النموذج عليها؟",
                            "القاعدة الحالية هي Baseline.",
                            "اجمع طلبات تاريخية بمآلاتها، اعزل فترة زمنية أخيرة كاختبار، قيّم القاعدة الحالية كـBaseline "
                            "بنفس المقياس (مثل التكلفة المتوقعة)، ثم درّب نموذجًا وقارنه على الفترة نفسها."),
        "research": _ex("التنبؤ مقابل التفسير في ورقة", "اختر ورقة تطبيقية في مجالك تستخدم ML. هل تدّعي تنبؤًا أم تفسيرًا أم "
                        "أثرًا سببيًا؟ هل يتوافق التقييم مع الادعاء؟",
                        "ابحث عن كلمات مثل effect وimpact وdrivers.",
                        "كثير من الأوراق تقيّم بدقة تنبؤية ثم تستنتج «محركات» سببية من Feature importance؛ هذا عدم توافق "
                        "بين الادعاء والدليل (Shmueli, 2010)."),
    },
    "ml_vocabulary": {
        "beginner": _ex("صنّف", "صنّف: alpha في Ridge، coef_، n_jobs، solver، cluster_centers_، n_estimators.",
                        "هل يتعلمه fit أم تختاره أنت؟",
                        "Hyperparameters: alpha، n_estimators. Parameters: coef_، cluster_centers_. Solver option: solver. "
                        "Runtime: n_jobs."),
        "intermediate": _ex("get_params", "اكتب كودًا يطبع كل المعاملات الفائقة لـRandomForestClassifier() ثم يغيّر max_depth "
                            "إلى 5 باستخدام set_params.", "كل Estimator يملك get_params وset_params.",
                            "```python\nrf = RandomForestClassifier()\nprint(rf.get_params())\nrf.set_params(max_depth=5)\n```"),
        "research": _ex("مستويا التعلّم", "اشرح لماذا يحتاج اختيار المعاملات الفائقة بيانات مستقلة عن تقدير الأداء النهائي، "
                        "بلغة المخاطرة التجريبية.", "اختيار λ هو تعلّم من الدرجة Validation.",
                        "λ̂ = argmin R̂_val(f̂_λ) ⇒ R̂_val(f̂_λ̂) هو حد أدنى لمجموعة تقديرات متقلبة، فهو متحيز للأسفل كتقدير "
                        "للخطأ. تقدير غير متحيز يتطلب بيانات لم تشارك في اختيار λ (Test أو الحلقة الخارجية في Nested CV)."),
    },
    "prediction_vs_causality": {
        "beginner": _ex("صنّف الأسئلة", "صنّف: (1) من سيشتري؟ (2) هل يزيد الإعلان المبيعات؟ (3) ما المتغيرات المرتبطة بالشراء؟",
                        "ابحث عن «لو فعلنا».", "(1) تنبؤي، (2) سببي، (3) وصفي/تفسيري."),
        "intermediate": _ex("ارسم DAG", "ارسم DAG لـ: التدخين، أصابع صفراء، سرطان الرئة. أيها مُربك؟ وأيها نتيجة؟",
                            "التدخين يسبب الاثنين.",
                            "التدخين → أصابع صفراء، التدخين → سرطان. الأصابع الصفراء تتنبأ بالسرطان لكنها ليست سببًا؛ "
                            "التدخين مُربك للعلاقة بينهما."),
        "research": _ex("Bad control", "لماذا يحيّز إدخال «الوظيفة الحالية» كضابط عند تقدير أثر التعليم على الدخل؟",
                        "الوظيفة تتأثر بالتعليم.",
                        "الوظيفة وسيط (Mediator) بين التعليم والدخل؛ ضبطها يحذف جزءًا من الأثر، وقد تكون Collider إن "
                        "تأثرت بعوامل غير ملاحظة تؤثر في الدخل، فيُفتح مسار مربك."),
    },
    "task_types": {
        "beginner": _ex("حدد النوع", "حدد نوع المسألة: عدد المكالمات غدًا، تشخيص ورم خبيث/حميد، تجميع الأغاني.",
                        "انظر إلى شكل المخرج.", "Forecasting/Regression، Binary classification، Clustering."),
        "intermediate": _ex("Multilabel", "كيف تحوّل وسم المقالات بعدة موضوعات إلى مسائل ثنائية؟ وما عيب ذلك؟",
                            "Binary relevance.", "مصنّف ثنائي لكل وسم (MultiOutputClassifier/OneVsRest). العيب: يتجاهل "
                            "الارتباط بين الوسوم؛ البديل ClassifierChain."),
        "research": _ex("Ordinal", "اقترح طريقة تحترم الترتيب في تقييمات 1–5 باستخدام مصنفات ثنائية.",
                        "Frank & Hall (2001).", "درّب K−1 مصنفًا لـP(y > k)، ثم P(y = k) = P(y > k−1) − P(y > k)."),
    },
    "feature_matrix": {
        "beginner": _ex("ابنِ X", "من جدول القروض، ما الأعمدة التي تدخل X وما الذي يُستبعد؟", "المعرّفات والهدف.",
                        "X: age, income, loan_amount, employment, region, city, credit_score, n_accounts. y: approved. "
                        "تُستبعد المعرّفات وأي عمود يُسجَّل بعد القرار."),
        "intermediate": _ex("احسب p", "إن كان للمدينة 40 فئة والمنطقة 5 والتوظيف 4، وبقية الأعمدة 5 عددية، كم عمودًا بعد One-hot "
                            "كامل؟ وبعد drop='first'؟", "كل متغير فئوي بـk فئة يعطي k أعمدة.",
                            "5 + 40 + 5 + 4 = 54؛ مع drop='first': 5 + 39 + 4 + 3 = 51."),
        "research": _ex("الرتبة", "أثبت أن [1, one_hot(region)] ليست كاملة الرتبة.", "اجمع أعمدة One-hot.",
                        "كل صف ينتمي لمنطقة واحدة ⇒ Σ_j onehot_j = 1 = عمود الواحدات ⇒ ارتباط خطي تام ⇒ rank ≤ k."),
    },
    "ml_readiness": {
        "beginner": _ex("شغّل الفحص", "شغّل الفحص على «بيانات مختلطة» ثم فعّل «أضف مشكلات». أي البنود تغيرت؟",
                        "راقب Leakage وDuplicates.", "يظهر Duplicates بعدد الصفوف المضافة، وLeakage يلتقط العمود المشتق من الهدف."),
        "intermediate": _ex("وثيقة جاهزية", "اكتب وثيقة جاهزية (كقاموس Python) لمسألة التنبؤ بالمبيعات اليومية.",
                            "ركز على الزمن.", "target: sales(t+1)؛ unit: يوم؛ prediction_time: نهاية اليوم t؛ features: Lag "
                            "وRolling حتى t فقط؛ split: آخر 60 يومًا اختبار؛ CV: TimeSeriesSplit مع gap؛ shift: تغير "
                            "الأسعار والعطلات."),
        "research": _ex("MNAR والجاهزية", "كيف يؤثر فقد الدخل غير العشوائي (المرتفعو الدخل لا يصرّحون) على نموذج الموافقة؟",
                        "هل الفقد نفسه معلومة؟", "الفقد يحمل معلومات عن الهدف؛ إضافة مؤشر missing_indicator قد يحسن "
                        "التنبؤ لكنها قد تعكس سياسة جمع البيانات، فتتغير إن تغيرت السياسة (خطر انجراف)."),
    },
    "baselines": {
        "beginner": _ex("Dummy", "درّب DummyClassifier(strategy='prior') واحسب ROC-AUC. لماذا 0.5؟",
                        "ما شكل احتمالاته؟", "يعطي الاحتمال نفسه لكل حالة ⇒ لا ترتيب ⇒ AUC = 0.5."),
        "intermediate": _ex("Seasonal naive", "احسب MAE لـseasonal naive بفترة 7 على آخر 60 يومًا من بيانات المبيعات.",
                            "y_hat[t] = y[t-7].", "```python\ns = df.sales.to_numpy(); t = np.arange(len(s)-60, len(s))\n"
                            "np.mean(np.abs(s[t] - s[t-7]))\n```"),
        "research": _ex("Baseline الخبير", "لماذا قد يكون نموذج أفضل من قاعدة الخبير على الاختبار أسوأ منها في النشر؟",
                        "من صنع التسميات؟", "إن كانت التسميات التاريخية نتيجة قرارات الخبير (Selective labels)، فالنموذج "
                        "يتعلم على عينة مشروطة بالسياسة الحالية؛ تغييرها يغيّر التوزيع (Performative prediction)."),
    },
    "data_splits": {
        "beginner": _ex("قسّم", "اكتب train_test_split بنسبة 25% مع الطبقية وبذرة ثابتة.", "stratify=y.",
                        "```python\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)\n```"),
        "intermediate": _ex("فترة Bootstrap", "كيف تبني فترة ثقة 95% لدقة الاختبار؟", "أعد أخذ عينات من الاختبار.",
                            "كرر 2000 مرة: اسحب بإرجاع من فهارس الاختبار، احسب الدقة؛ خذ المئين 2.5 و97.5."),
        "research": _ex("تحيز إعادة الاستخدام", "إن قارنت 50 نموذجًا على نفس Test واخترت الأفضل، كيف يتحيز التقدير؟",
                        "القيمة القصوى لمتغيرات عشوائية.", "E[max_k Â_k] ≥ max_k E[Â_k]؛ التحيز يزداد مع عدد النماذج وتباين "
                        "التقدير (حجم Test الصغير). الحل: Test معزول أو تصحيح/‏Nested CV."),
    },
    "leakage": {
        "beginner": _ex("اكتشف التسرب", "في مسألة التنبؤ بمغادرة العملاء، هل «عدد مكالمات إلغاء الخدمة» خاصية آمنة؟",
                        "متى تحدث المكالمات؟", "غالبًا لا: تحدث في عملية المغادرة نفسها؛ آمنة فقط إن كانت محسوبة قبل "
                        "نافذة التنبؤ بفترة كافية."),
        "intermediate": _ex("أصلح الكود", "أعد كتابة: `X = SimpleImputer().fit_transform(X); cross_val_score(model, X, y)` دون تسرب.",
                            "Pipeline.", "```python\ncross_val_score(make_pipeline(SimpleImputer(), model), X, y, cv=5)\n```"),
        "research": _ex("Adversarial validation", "صف إجراءً لاكتشاف اختلاف التوزيع بين التدريب والاختبار.",
                        "صنّف مصدر الصف.", "أضف تسمية is_test، درّب مصنفًا يميزها بـCV؛ AUC ≫ 0.5 ⇒ التوزيعان مختلفان؛ "
                        "الخصائص الأهم في هذا المصنف تكشف مصدر الاختلاف أو التسرب."),
    },
    "pipelines": {
        "beginner": _ex("Pipeline بسيط", "ابنِ Pipeline من StandardScaler وKNeighborsClassifier واحسب CV accuracy.",
                        "make_pipeline.", "```python\ncross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y, cv=5)\n```"),
        "intermediate": _ex("ColumnTransformer", "أضف log1p للدخل داخل الـPipeline قبل القياس.", "FunctionTransformer.",
                            "```python\nnum = Pipeline([('imp', SimpleImputer(strategy='median')),\n                ('log', FunctionTransformer(np.log1p)),\n"
                            "                ('scale', StandardScaler())])\n```"),
        "research": _ex("Training-serving skew", "اشرح كيف يمنع حفظ الـPipeline الكامل اختلاف المعالجة بين التدريب والإنتاج، "
                        "وما الذي لا يمنعه.", "ماذا عن البيانات الخام نفسها؟",
                        "يضمن تطبيق الإحصاءات نفسها (متوسطات، فئات) في الإنتاج. لا يمنع اختلاف تعريف الأعمدة الخام أو "
                        "وحداتها أو توقيت توفرها؛ يحتاج ذلك Data contracts ومراقبة."),
    },
}
