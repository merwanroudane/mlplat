"""Exercises: supervised learning."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "linear_regression": {
        "beginner": _ex("فسّر المعامل", "ŷ = 50 + 2.5·hours − 4·absences. فسّر 2.5.", "مع ثبات الغياب.",
                        "كل ساعة دراسة إضافية ترتبط بزيادة 2.5 نقطة في المتوسط مع ثبات عدد الغيابات (ارتباط شرطي لا أثر سببي)."),
        "intermediate": _ex("HC3", "قدّر OLS بـstatsmodels مع أخطاء معيارية HC3 وقارنها بالكلاسيكية.", "fit(cov_type='HC3').",
                            "```python\nsm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3').summary()\n```\nمع عدم التجانس تكون HC3 أكبر غالبًا."),
        "research": _ex("FWL وDML", "اكتب خطوات تقدير معامل d في y = θd + g(X) + ε بطريقة FWL ثم وضّح ما يضيفه DML.",
                        "استبدل الإسقاط الخطي.", "FWL: ابقِ بواقي y وd بعد إسقاطهما خطيًا على X ثم انحدار بواقي على بواقي. DML: "
                        "استبدل الإسقاط بتنبؤات ML لـE[y|X] وE[d|X]، واحسبها خارج الطية (Cross-fitting) لتجنب تحيز الإفراط."),
    },
    "polynomial_regression": {
        "beginner": _ex("عدد الحدود", "كم عمودًا ينتج PolynomialFeatures(2) لخاصيتين (مع include_bias)؟", "1, x1, x2, ...",
                        "6: 1, x1, x2, x1², x1x2, x2²."),
        "intermediate": _ex("Pipeline آمن", "ابنِ Pipeline لدرجة 8 مع قياس وRidgeCV.", "الترتيب مهم.",
                            "```python\nmake_pipeline(PolynomialFeatures(8, include_bias=False), StandardScaler(), RidgeCV(np.logspace(-4, 3, 30)))\n```"),
        "research": _ex("Splines", "لماذا تُفضَّل Natural splines للاستقراء؟", "ما سلوكها خارج العقد الطرفية؟",
                        "تفرض خطية خارج الحدود فتتجنب الانفجار التكعيبي؛ الاستقراء يبقى محفوفًا بالمخاطر لكنه أكثر اعتدالًا."),
    },
    "regularization": {
        "beginner": _ex("اختر", "100 خاصية يُتوقع أن 5 منها فقط مهمة. أي تنظيم؟", "الندرة.", "Lasso (أو Elastic Net إن وُجد ترابط)."),
        "intermediate": _ex("LassoCV", "استخدم LassoCV على بيانات high_dim داخل Pipeline وأبلغ عن عدد المعاملات غير الصفرية.",
                            "alphas=50 (منذ 1.9).", "```python\npipe = make_pipeline(StandardScaler(), LassoCV(alphas=50, cv=5)).fit(X, y)\n"
                            "(pipe[-1].coef_ != 0).sum()\n```\nعادة 10–20: الحقيقية + بعض الضجيج."),
        "research": _ex("Post-double-selection", "صف إجراء Belloni et al. (2014) لتقدير أثر d مع ضوابط عالية الأبعاد.",
                        "اختياران.", "(1) Lasso لـy على X، (2) Lasso لـd على X، (3) OLS لـy على d واتحاد الضوابط المختارة. يحمي من "
                        "إسقاط مربكات ترتبط بـd بقوة وبـy بضعف؛ DML يعمم الفكرة لأي متعلم."),
    },
    "robust_regression": {
        "beginner": _ex("قارن", "في المختبر، ارفع نسبة الشواذ إلى 30%. أي مقدِّر بقي أقرب إلى الميل 2؟", "راقب الجدول.",
                        "عادة RANSAC أو Theil-Sen للرافعة، وHuber/Median للشواذ الرأسية."),
        "intermediate": _ex("outliers_", "درّب HuberRegressor واطبع عدد النقاط المعلّمة كشاذة.", "خاصية outliers_.",
                            "```python\nh = HuberRegressor().fit(X, y); h.outliers_.sum()\n```"),
        "research": _ex("الكفاءة", "لماذا epsilon = 1.35 في Huber؟", "الكفاءة تحت الطبيعية.",
                        "يعطي كفاءة تقاربية ≈ 95% مقارنة بـOLS حين تكون الأخطاء طبيعية فعلًا، مع متانة أمام الذيول الثقيلة."),
    },
    "classification_setup": {
        "beginner": _ex("صنّف", "هل تشخيص 5 أمراض متنافية مسألة Multiclass أم Multilabel؟", "هل يمكن وجود مرضين؟",
                        "إن كانت متنافية: Multiclass؛ إن أمكن اجتماعها: Multilabel."),
        "intermediate": _ex("OvR يدويًا", "درّب 3 مصنفات Logistic ثنائية (OvR) وقارن بـLogisticRegression متعدد الحدود.",
                            "OneVsRestClassifier.", "الدقة متقاربة غالبًا؛ احتمالات Softmax متسقة (مجموعها 1) بلا تطبيع لاحق."),
        "research": _ex("Macro vs micro", "متى يختلف Macro-F1 كثيرًا عن Micro-F1؟", "الفئات النادرة.",
                        "حين يكون الأداء ضعيفًا على الفئات النادرة: Micro يسيطر عليه الفئات الكبيرة، Macro يعطي كل فئة وزنًا متساويًا."),
    },
    "logistic_regression": {
        "beginner": _ex("احسب p", "β₀ = −1، β₁ = 0.5، x = 4. احسب p.", "σ(z).", "z = 1 ⇒ p = 1/(1 + e^{−1}) = 0.731."),
        "intermediate": _ex("L1 حديث", "اكتب LogisticRegression بعقوبة L1 بالطريقة غير المهجورة وقوة C = 0.1.", "l1_ratio.",
                            "```python\nLogisticRegression(l1_ratio=1.0, C=0.1, solver='saga', max_iter=5000)\n```"),
        "research": _ex("Non-collapsibility", "بيّن بالمحاكاة أن Odds ratio لـd يتغير عند إضافة متغير مستقل عن d يؤثر في y.",
                        "d ⫫ z.", "Odds ratio الشرطي (مع z) أكبر من الهامشي (دون z) رغم عدم وجود إرباك — خاصية لا تملكها "
                        "الفروق في الاحتمالات؛ لذلك يُفضَّل الإبلاغ عن فروق احتمالات للأثر السببي."),
    },
    "naive_bayes": {
        "beginner": _ex("يدويًا", "P(spam) = 0.4، P(free|spam) = 0.3، P(free|ham) = 0.02. رسالة فيها free فقط: P(spam|free)؟",
                        "Bayes.", "0.12 / (0.12 + 0.012) = 0.909."),
        "intermediate": _ex("alpha", "اضبط alpha لـMultinomialNB بـGridSearchCV على بيانات النصوص.", "np.logspace(-3, 1, 9).",
                            "```python\nGridSearchCV(make_pipeline(CountVectorizer(), MultinomialNB()), {'multinomialnb__alpha': np.logspace(-3, 1, 9)}, cv=5)\n```"),
        "research": _ex("توليدي vs تمييزي", "لخّص نتيجة Ng & Jordan (2002).", "سرعة التقارب مقابل الخطأ التقاربي.",
                        "NB يصل إلى خطئه التقاربي بـO(log p) عينة، Logistic بـO(p)؛ لكن خطأ Logistic التقاربي ≤ خطأ NB. مع بيانات "
                        "قليلة قد يفوز NB، ومع بيانات كثيرة يفوز Logistic."),
    },
    "lda_qda": {
        "beginner": _ex("عد المعاملات", "كم معامل تغاير يقدّر QDA لـ3 فئات وp = 10؟", "p(p+1)/2 لكل فئة.", "3 × 55 = 165."),
        "intermediate": _ex("LDA مقابل PCA", "أسقط بيانات wine على مكونين بـPCA وLDA وقارن الفصل.", "LDA موجَّه.",
                            "LDA يفصل الأصناف الثلاثة بوضوح، PCA يعظّم التباين دون اعتبار للفئات فيخلطها جزئيًا."),
        "research": _ex("الحد الخطي", "اشتق أن حد LDA بين فئتين خطي.", "δ₁(x) = δ₂(x).",
                        "xᵀΣ⁻¹(μ₁ − μ₂) = ½(μ₁ᵀΣ⁻¹μ₁ − μ₂ᵀΣ⁻¹μ₂) + log(π₂/π₁): معادلة خطية في x."),
    },
    "knn": {
        "beginner": _ex("صوّت", "جيران نقطة: (1, 1, 0, 0, 1, 0, 0) مع k = 7. التنبؤ واحتمال الفئة 1؟", "أغلبية.", "0، واحتمال 3/7 ≈ 0.43."),
        "intermediate": _ex("اضبط k", "اختر k بـGridSearchCV داخل Pipeline مع StandardScaler.", "kneighborsclassifier__n_neighbors.",
                            "```python\nGridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), {'kneighborsclassifier__n_neighbors': range(1, 50, 2)}, cv=5)\n```"),
        "research": _ex("حد Cover–Hart", "ماذا يعني أن خطأ 1-NN ≤ 2 × خطأ Bayes تقاربيًا؟", "مع n → ∞.",
                        "حتى أبسط قاعدة جوار تستخدم نصف المعلومات الإحصائية المتاحة على الأقل؛ لكن التقارب قد يكون بطيئًا "
                        "جدًا في الأبعاد العالية."),
    },
    "svm": {
        "beginner": _ex("الهامش", "w = (3, 4). ما عرض الهامش؟", "2/‖w‖.", "2/5 = 0.4."),
        "intermediate": _ex("احتمالات حديثة", "اكتب SVC بنواة RBF مع احتمالات دون probability=True.", "CalibratedClassifierCV.",
                            "```python\nCalibratedClassifierCV(make_pipeline(StandardScaler(), SVC()), method='sigmoid', ensemble=False, cv=5)\n```"),
        "research": _ex("Nystroem", "كيف تُقرّب SVM بنواة RBF على مليون صف؟", "kernel_approximation.",
                        "Nystroem(kernel='rbf', n_components=500) أو RBFSampler ثم LinearSVC/SGDClassifier: خطي في n."),
    },
}
