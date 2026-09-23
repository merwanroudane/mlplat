"""Quizzes: hyperparameter optimisation."""

QUIZZES: dict[str, list[dict]] = {
    "hpo_foundations": [
        {"q": "ميزانية 25 محاولة ومعامل واحد مهم فقط. كم قيمة مختلفة له تختبر Grid 5×5؟", "type": "mcq",
         "options": ["25", "5", "1", "10"], "answer": 1, "explain": "Random يختبر 25 قيمة مختلفة."},
        {"q": "Successive halving يوفر الزمن لأنه:", "type": "mcq",
         "options": ["يتخطى CV", "يقيّم معظم الإعدادات بموارد قليلة ويرقّي الأفضل فقط", "يستخدم GPU", "يقلل فضاء البحث"], "answer": 1,
         "explain": "Multi-fidelity."},
        {"q": "مقياس مناسب لـC في SVC:", "type": "hyper", "options": ["خطي 0..100", "لوغاريتمي 1e-3..1e3", "صحيح 1..10", "فئوي"], "answer": 1,
         "explain": "أثر C نسبي/أُسّي."},
        {"q": "ما الاستيراد المطلوب لـHalvingRandomSearchCV في 1.9.1؟", "type": "code",
         "options": ["لا شيء", "from sklearn.experimental import enable_halving_search_cv", "import optuna", "from sklearn import halving"],
         "answer": 1, "explain": "ما زال تجريبيًا."},
    ],
    "bayesian_optimization": [
        {"q": "Expected improvement كبير حين:", "type": "mcq",
         "options": ["μ منخفض وσ صغير", "μ مرتفع أو σ كبير", "دائمًا عند الحدود", "لا يعتمد على σ"], "answer": 1,
         "explain": "يوازن الاستغلال (μ) والاستكشاف (σ)."},
        {"q": "Pruning في Optuna يتطلب:", "type": "code", "options": ["لا شيء", "trial.report(value, step) وtrial.should_prune()", "GPSampler", "n_jobs"],
         "answer": 1, "explain": "قيم وسيطة لكل خطوة."},
        {"q": "الـSampler الافتراضي في Optuna 5.0:", "type": "mcq", "options": ["Random", "TPE", "CMA-ES", "Grid"], "answer": 1,
         "explain": "TPE متعدد المتغيرات افتراضيًا في 5.0."},
        {"q": "best_value بعد 200 محاولة على 300 صف:", "type": "scenario", "options": ["تقدير غير متحيز", "متفائل؛ خطر فرط ملاءمة التحقق",
                                                                                    "متشائم", "يساوي Test"], "answer": 1,
         "explain": "القيمة القصوى لتقديرات ضجيجية."},
    ],
}
