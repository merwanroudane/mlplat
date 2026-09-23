"""Quizzes: modern topics and bridges."""

QUIZZES: dict[str, list[dict]] = {
    "modern_topics": [
        {"q": "في SelfTrainingClassifier تُرمَّز الأمثلة غير المسماة بـ:", "type": "code", "options": ["NaN", "-1", "0", "None"], "answer": 1,
         "explain": "اصطلاح scikit-learn للتعلم شبه الموجّه."},
        {"q": "Uncertainty sampling يختار للتسمية:", "type": "mcq", "options": ["الحالات الأوضح", "الحالات التي احتمالها أقرب إلى 0.5", "عشوائيًا", "الشواذ"],
         "answer": 1, "explain": "الأكثر إرباكًا للنموذج."},
        {"q": "أول استدعاء لـpartial_fit لمصنف يتطلب:", "type": "code", "options": ["لا شيء", "classes=[...]", "sample_weight", "n_iter"], "answer": 1,
         "explain": "لأن الدفعة الأولى قد لا تحوي كل الفئات."},
    ],
    "time_series_ml": [
        {"q": "لماذا shift(1) قبل rolling؟", "type": "code", "options": ["للسرعة", "حتى لا يتضمن المتوسط قيمة اليوم الهدف (تسرب)", "للتنعيم", "لا حاجة"],
         "answer": 1, "explain": "rolling يشمل الصف الحالي افتراضيًا."},
        {"q": "KFold عشوائي على سلسلة زمنية يعطي خطأ:", "type": "chart", "options": ["أعلى من الحقيقي", "أقل من الحقيقي (متفائل)", "مساويًا", "عشوائيًا"],
         "answer": 1, "explain": "يتدرب على المستقبل ويختبر على ماضٍ مجاور."},
        {"q": "أفق التنبؤ 7 أيام. أقصر lag آمن:", "type": "hyper", "options": ["1", "7", "0", "30"], "answer": 1,
         "explain": "القيم الأحدث من 7 أيام غير معروفة وقت التنبؤ."},
    ],
    "text_ml": [
        {"q": "كلمة تظهر في كل الوثائق، وزن IDF لها:", "type": "mcq", "options": ["مرتفع", "أدنى قيمة", "سالب", "لانهائي"], "answer": 1,
         "explain": "لا تميّز بين الوثائق."},
        {"q": "لالتقاط «not good» نستخدم:", "type": "method", "options": ["unigrams فقط", "n-grams (1,2)", "حذف stopwords", "PCA"], "answer": 1,
         "explain": "الثنائيات تحفظ التركيب."},
    ],
    "rl_bridge": [
        {"q": "ε = 0 في ε-greedy يعني:", "type": "mcq", "options": ["استكشاف كامل", "استغلال فقط (قد يعلق في حل سيئ)", "عشوائي", "تعلم أسرع دائمًا"],
         "answer": 1, "explain": "لا يجرب بدائل."},
        {"q": "هدف TD في Q-learning:", "type": "mcq", "options": ["r", "r + γ max Q(s′, ·)", "Q(s, a)", "γ"], "answer": 1, "explain": "مكافأة الآن + أفضل قيمة بعدها."},
    ],
    "dl_bridge": [
        {"q": "شبكة بتنشيط identity وعدة طبقات:", "type": "mcq", "options": ["غير خطية", "مكافئة لنموذج خطي واحد", "لا تتدرب", "أعمق = أفضل"],
         "answer": 1, "explain": "تركيب خطي لخطيات = خطي."},
        {"q": "Backpropagation هو:", "type": "mcq", "options": ["خوارزمية تحسين", "حساب التدرجات بقاعدة السلسلة", "نوع طبقة", "تنظيم"], "answer": 1,
         "explain": "المحسّن (SGD/Adam) يستخدم هذه التدرجات."},
    ],
}
