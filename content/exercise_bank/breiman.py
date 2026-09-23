"""Exercises: Leo Breiman's philosophy."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "breiman_two_cultures": {
        "beginner": _ex("صنّف الثقافة", "لكل حالة: (أ) انحدار خطي مع قيم p لتفسير أثر التعليم على الدخل؛ (ب) غابة عشوائية "
                        "تُقيَّم بـ5-fold CV للتنبؤ بتسرّب العملاء. أي ثقافة؟ وما معيار الصحة في كل منهما؟",
                        "ما الذي يُقدَّر؟ وكيف يُحكم عليه؟",
                        "(أ) نمذجة البيانات: معاملات + اختبارات ملاءمة وبواقي. (ب) خوارزمية: خطأ التنبؤ خارج العينة."),
        "intermediate": _ex("أعد إنتاج المختبر", "ولّد بيانات two_cultures_data(nonlinearity=1.5)، درّب Logistic وRandomForest، "
                            "واحسب Hosmer–Lemeshow على التدريب وAUC على اختبار 40%.",
                            "utils.breiman.hosmer_lemeshow وroc_auc_score.",
                            "```python\nX, y = two_cultures_data(1500, 1.5)\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, "
                            "stratify=y, random_state=0)\nlr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)\n"
                            "print(hosmer_lemeshow(ytr, lr.predict_proba(Xtr)[:, 1]))\nrf = RandomForestClassifier(300, "
                            "min_samples_leaf=5).fit(Xtr, ytr)\nprint(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]), "
                            "roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))\n```\nعلى بذرة المنصة: HL p ≈ 0.84 (لا رفض) "
                            "وAUC ≈ 0.63 مقابل ≈ 0.84."),
        "research": _ex("قوة الاختبار", "بمحاكاة Monte Carlo (200 تكرار) قدّر احتمال أن يرفض Hosmer–Lemeshow النموذج الخطي عند "
                        "α = 0.05 لقيم λ ∈ {0, 0.5, 1, 1.5, 2}. ماذا تستنتج عن نقد Breiman؟",
                        "كرر توليد البيانات بـseed مختلف واحسب نسبة p < 0.05.",
                        "عند λ = 0 تكون النسبة ≈ 5% (الحجم الاسمي). مع λ > 0 ترتفع ببطء، وتبقى القوة منخفضة لأن الاختبار "
                        "يجمع حسب الاحتمال المتنبأ به ولا يستهدف التفاعل x3·x4 أو x5² مباشرة؛ بينما فرق AUC خارج العينة يكشف "
                        "المواصفة الناقصة بوضوح. هذا يدعم دعوة Breiman لاستخدام الدقة التنبؤية فحصًا للنموذج."),
    },
    "breiman_three_lessons": {
        "beginner": _ex("اقرأ راشومون", "في المختبر مع k = 4 وتسامح 1%، كم نموذجًا «متساويًا» وكم متغيرًا مختلفًا يظهر فيها؟ "
                        "ماذا يعني ذلك لتفسير نموذج واحد؟", "انظر إلى المقاييس بجانب الرسم.",
                        "عدة نماذج بـR² متطابق تقريبًا ومتغيرات مختلفة ⇒ البيانات لا تميز بين القصص؛ تفسير نموذج واحد مضلل."),
        "intermediate": _ex("استقرار الجذر", "درّب DecisionTreeClassifier(max_depth=3) على 50 عينة Bootstrap من بياناتك واحسب "
                            "تكرار متغير الجذر. ثم قارن بترتيب permutation_importance لغابة.",
                            "tree_.feature[0] يعطي فهرس متغير الجذر.",
                            "```python\nroots = [X.columns[DecisionTreeClassifier(max_depth=3).fit(X.iloc[i], y.iloc[i]).tree_.feature[0]]\n"
                            "         for i in (rng.integers(0, len(y), len(y)) for _ in range(50))]\npd.Series(roots).value_counts()\n```"),
        "research": _ex("مدى الأهمية عبر مجموعة راشومون", "لكل نموذج في مجموعة راشومون (k = 4، ε = 3%) احسب معامل كل متغير. "
                        "أبلغ عن المدى [min, max] لكل متغير ولماذا هو أصدق من رقم واحد.",
                        "Fisher, Rudin & Dominici (2019): Model class reliance.",
                        "المدى يعكس عدم قدرة البيانات على الفصل بين المتغيرات المترابطة؛ متغير يتراوح أثره بين 0 وقيمة كبيرة "
                        "عبر نماذج متساوية الدقة لا يمكن الادعاء بأهميته أو عدمها. هذا تقرير «فئة النماذج» بدل «النموذج»."),
    },
    "breiman_legacy": {
        "beginner": _ex("OOB", "درّب RandomForestClassifier(oob_score=True) وقارن oob_score_ بـ5-fold CV.",
                        "oob_score=True ثم .oob_score_.", "التقديران متقاربان عادة؛ OOB يأتي من تدريب واحد."),
        "intermediate": _ex("Bagging والاستقرار", "قارن خطأ الاختبار لـDecisionTreeClassifier وLogisticRegression مفردين ومع "
                            "BaggingClassifier(n_estimators=50). أيهما يستفيد؟",
                            "utils.breiman.instability يقيس عدم الاستقرار.",
                            "الشجرة غير المستقرة تستفيد بوضوح؛ الانحدار اللوجستي المستقر بالكاد يتغير (وتحيزه يبقى)."),
        "research": _ex("اشتقاق تباين المتوسط", "لـB متغيرات عشوائية متساوية التباين σ² بارتباط زوجي ρ، أثبت أن "
                        "Var(المتوسط) = ρσ² + (1 − ρ)σ²/B، وفسّر دور max_features.",
                        "وسّع Var(ΣX_b)/B² = [Bσ² + B(B − 1)ρσ²]/B².",
                        "Var = σ²/B + (B − 1)ρσ²/B = ρσ² + (1 − ρ)σ²/B. مع B → ∞ يبقى ρσ²؛ الخصائص العشوائية (max_features) "
                        "تخفض ρ على حساب قوة كل شجرة، وهي المفاضلة في حد Breiman PE* ≤ ρ̄(1 − s²)/s²."),
    },
}
