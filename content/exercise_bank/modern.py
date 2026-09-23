"""Exercises: modern topics and bridges."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "modern_topics": {
        "beginner": _ex("Self-training", "درّب SelfTrainingClassifier مع 5% تسميات على moons.", "y_semi = -1 لغير المسمى.",
                        "SelfTrainingClassifier(LogisticRegression(), threshold=0.8).fit(X, y_semi)."),
        "intermediate": _ex("Active loop", "اكتب حلقة Active learning بـ5 جولات و10 تسميات لكل جولة.", "argsort(|p − 0.5|).",
                            "انظر كود المختبر في الصفحة (_active)."),
        "research": _ex("تقييم Active", "لماذا لا تقيّم على الحالات المختارة؟", "تحيز الاختيار.",
                        "لأنها مختارة لكونها صعبة وغير ممثلة؛ استخدم عينة اختبار عشوائية ثابتة منذ البداية."),
    },
    "time_series_ml": {
        "beginner": _ex("Lags", "أنشئ lag_1 وlag_7 وroll_mean_7 آمنة.", "shift ثم rolling.",
                        "df['lag_1'] = s.shift(1); df['lag_7'] = s.shift(7); df['roll'] = s.shift(1).rolling(7).mean()."),
        "intermediate": _ex("Walk-forward", "قيّم HGB بـTimeSeriesSplit(test_size=60, gap=7).", "cross_val_score.",
                            "cross_val_score(HistGradientBoostingRegressor(), X, y, cv=TimeSeriesSplit(5, test_size=60, gap=7), scoring='neg_mean_absolute_error')."),
        "research": _ex("Direct vs recursive", "صمم تجربة تقارن الاستراتيجيتين لأفق 14.", "نماذج لكل أفق.",
                        "Direct: 14 نموذجًا بأهداف y(t+h)؛ Recursive: نموذج خطوة واحدة يُغذّى بتنبؤاته؛ قارن MAE لكل h على نوافذ Walk-forward."),
    },
    "text_ml": {
        "beginner": _ex("TF-IDF", "أنشئ TfidfVectorizer(ngram_range=(1,2)) واطبع حجم المفردات.", "get_feature_names_out.", "len(vec.get_feature_names_out())."),
        "intermediate": _ex("Pipeline", "قارن MultinomialNB وLogisticRegression على TF-IDF بـCV.", "make_pipeline.", "انظر جدول المختبر."),
        "research": _ex("العربية", "ما خطوات المعالجة الخاصة بالنص العربي؟", "تطبيع.",
                        "توحيد الهمزات والتاء المربوطة، إزالة التشكيل والتطويل، التجذير/الجذوع، والتعامل مع اللهجات."),
    },
    "rl_bridge": {
        "beginner": _ex("عائد", "مكافآت (−0.1, −0.1, 10) وγ = 0.9. احسب G₀.", "Σγᵏr.", "−0.1 − 0.09 + 8.1 = 7.91."),
        "intermediate": _ex("ε", "قارن منحنيات التعلم لـε = 0 و0.2 و0.8.", "المختبر.", "0.2 يتعلم أسرع وأثبت عادة؛ 0.8 يستكشف كثيرًا فعائده أقل."),
        "research": _ex("Bandits", "اربط Contextual bandits بتقييم السياسات من البيانات التاريخية.", "IPW/DR.",
                        "قيمة سياسة جديدة تُقدَّر بإعادة وزن المكافآت الملاحظة باحتمالات السياسة القديمة (IPW) أو بدرجة مزدوجة المتانة."),
    },
    "dl_bridge": {
        "beginner": _ex("ReLU", "احسب ReLU(−2, 0.5, 3).", "max(0, z).", "(0, 0.5, 3)."),
        "intermediate": _ex("MLP", "درّب MLPClassifier بطبقتين من 16 وحدة على circles.", "hidden_layer_sizes=(16, 16).",
                            "make_pipeline(StandardScaler(), MLPClassifier((16, 16), max_iter=2000)).fit(X, y)."),
        "research": _ex("Tabular", "لماذا يتفوق التعزيز الشجري غالبًا على الشبكات في البيانات الجدولية؟", "الانحيازات الاستقرائية.",
                        "الأشجار تتعامل طبيعيًا مع الخصائص غير المتجانسة والحدود الحادة والخصائص غير المفيدة؛ الشبكات تفضّل الدوال الملساء."),
    },
}
