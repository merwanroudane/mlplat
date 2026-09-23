"""Exercises: causal ML and double machine learning."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "causal_foundations": {
        "beginner": _ex("عرّف", "لدراسة أثر التطعيم على الاستشفاء، حدد D وY وX ومربكًا محتملًا.", "من يختار التطعيم؟",
                        "D: التطعيم؛ Y: الاستشفاء خلال 6 أشهر؛ X: العمر، الأمراض المزمنة؛ مربك: الحالة الصحية العامة (تؤثر في الإقبال والنتيجة)."),
        "intermediate": _ex("DAG", "ارسم DAG لـ: تدريب ← مهارة ← أجر، ودافعية تؤثر في التدريب والأجر. ماذا تضبط لأثر التدريب الكلي؟",
                            "لا تضبط الوسيط.", "اضبط الدافعية (مربك) إن قيست؛ لا تضبط المهارة (وسيط) لأنها جزء من الأثر الكلي."),
        "research": _ex("التداخل", "اقترح إجراءً عند ضعف التداخل.", "غيّر المعلمة المستهدفة.",
                        "قصّر المجتمع على منطقة التداخل (Crump et al.)، أو استهدف ATO بأوزان التداخل، وصرّح بتغير المجتمع المستهدف."),
    },
    "dml_core": {
        "beginner": _ex("الخطوات", "رتّب: حل θ، Cross-fitting، درجة متعامدة، ML للإزعاج، SE.", "ML أولًا.",
                        "ML للإزعاج ← درجة متعامدة ← Cross-fitting ← حل θ ← SE/CI."),
        "intermediate": _ex("من الصفر", "نفّذ DML-PLR بـ5 طيات وRandomForest باستخدام utils/causal.dml_plr وقارن بـOLS.", "make_plr(n=1000).",
                            "```python\ndf = make_plr(1000); X = df.filter(like='x').to_numpy()\nC.dml_plr(X, df.y.to_numpy(), df.d.to_numpy(), rf, rf).theta\n```\n≈ 0.5 مقابل OLS المتحيز."),
        "research": _ex("شرط المعدل", "لماذا يكفي o(n^{-1/4})؟", "حد الرتبة الثانية.",
                        "التحيز المتبقي ∝ ‖m̂ − m₀‖·‖ℓ̂ − ℓ₀‖؛ إن كان كل منهما o(n^{-1/4}) فالحاصل o(n^{-1/2}) ⇒ مهمل مقارنة بـSE ∝ n^{-1/2}."),
    },
    "neyman_orthogonality": {
        "beginner": _ex("حدس", "اشرح بجملة لماذا التعامد مفيد.", "حساسية.", "خطأ صغير في «التنظيف» لا يغيّر θ̂ إلا بمقدار مربعه."),
        "intermediate": _ex("تحقق عددي", "في المختبر، قارن ميل منحنى التحيز عند δ = 0 للدرجتين.", "مشتقة عددية.",
                            "غير المتعامدة: ميل ≠ 0؛ المتعامدة: ≈ 0."),
        "research": _ex("اشتقاق", "اشتق مشتقة Gateaux لدرجة partialling out في اتجاه Δm.", "استخدم E[V|X] = 0.",
                        "∂_r E[(U − θV + θrΔm)(V − rΔm)] = E[θΔm V] − E[(U − θV)Δm] = 0 لأن E[V|X] = E[ζ|X] = 0."),
    },
    "cross_fitting": {
        "beginner": _ex("طيات", "مع K = 4 وn = 1000، على كم صف يُدرَّب كل نموذج إزعاج؟", "(K−1)/K.", "750."),
        "intermediate": _ex("DoubleML", "استخدم set_sample_splitting لتمرير طياتك إلى DoubleMLPLR.", "قائمة قوائم.",
                            "```python\nplr.set_sample_splitting([[(tr, te) for tr, te in folds]])\n```"),
        "research": _ex("Donsker", "ما الذي يسمح به Cross-fitting نظريًا؟", "تعقيد الفضاء.",
                        "يلغي شرط Donsker على فضاء الإزعاج، فيسمح بمتعلمين عاليي التعقيد (غابات، تعزيز، شبكات)."),
    },
    "plr": {
        "beginner": _ex("بواقي", "احسب Û وV̂ بتنبؤات خارج الطية وارسم Û على V̂.", "cross_fit_predict.", "الميل = θ̂ ≈ 0.5."),
        "intermediate": _ex("IV-type", "قارن partialling out مع IV-type في DoubleMLPLR.", "ml_g مطلوب.", "متقاربان عادة؛ IV-type قد يكون أكثر كفاءة بإزعاج جيد."),
        "research": _ex("أثر غير متجانس", "مع θ(X) متغير، ماذا يقدّر PLR؟", "وزن التباين.",
                        "θ̄ = E[Var(D|X)θ(X)]/E[Var(D|X)]: متوسط موزون يعطي وزنًا أكبر لقيم X ذات التباين المتبقي الأكبر في D."),
    },
    "pliv": {
        "beginner": _ex("شروط", "لأداة «المسافة إلى الجامعة» لأثر التعليم على الدخل، ما خطر الاستبعاد؟", "المسافة وسوق العمل.",
                        "القرب من المدن الجامعية قد يرتبط بأسواق عمل أفضل ⇒ أثر مباشر على الدخل؛ اضبط الموقع في X."),
        "intermediate": _ex("مرحلة أولى", "احسب F للمرحلة الأولى على البواقي.", "انحدر D̃ على Z̃.", "F = t² للمعامل؛ < 10 تقليديًا علامة ضعف."),
        "research": _ex("Anderson–Rubin", "لماذا فترات AR قوية أمام الضعف؟", "لا تعتمد على قوة المرحلة الأولى.",
                        "تختبر كل θ₀ عبر انحدار Y − θ₀D على Z؛ توزيعها صحيح بغض النظر عن قوة الأداة."),
    },
    "irm": {
        "beginner": _ex("ATE vs ATT", "برنامج اختياري. أيهما يهم صانع القرار لتوسيعه للجميع؟", "من سيتأثر؟", "ATE (أو CATE لمن سيُضاف)؛ ATT يصف المشاركين الحاليين."),
        "intermediate": _ex("تداخل", "ارسم توزيع m̂(X) لكل مجموعة واحسب نسبة m̂ خارج [0.05, 0.95].", "res.extra['m_hat'].",
                            "إن كانت كبيرة فالتقدير يعتمد على الاستقراء؛ ناقش القص."),
        "research": _ex("كفاءة", "لماذا AIPW هو المقدِّر الكفء؟", "Efficient influence function.",
                        "درجته تساوي دالة التأثير الكفء لـATE تحت عدم الإرباك (Hahn, 1998)، فيبلغ حد التباين شبه المعلمي."),
    },
    "iivm": {
        "beginner": _ex("Wald", "ITT = 2، المرحلة الأولى = 0.4. LATE؟", "قسمة.", "5."),
        "intermediate": _ex("Subgroups", "كيف تعلن غياب Always-takers في DoubleMLIIVM؟", "subgroups.", "subgroups={'always_takers': False, 'never_takers': True}."),
        "research": _ex("تعميم", "متى يساوي LATE الـATE؟", "تجانس.", "إن كان الأثر متجانسًا، أو لا يختلف بين الأنواع (افتراض إضافي غير قابل للاختبار عادةً)."),
    },
    "dml_learners": {
        "beginner": _ex("قارن", "في المختبر، ما المتعلم الأقرب لـ0.5 مع لاخطية كاملة؟", "راجع الرسم.", "RF أو HGB عادة؛ OLS متحيز."),
        "intermediate": _ex("ضبط داخلي", "مرّر GridSearchCV كمتعلم إلى dml_plr.", "clone داخل الطيات.",
                            "```python\ntuned = GridSearchCV(RandomForestRegressor(), {'min_samples_leaf': [2, 10, 30]}, cv=3)\nC.dml_plr(X, y, d, tuned, tuned)\n```"),
        "research": _ex("حساسية المتعلم", "كيف تبلغ عن حساسية النتائج لاختيار المتعلم؟", "جدول.",
                        "جدول θ̂ وSE وخسائر الإزعاج لعدة متعلمين محددين مسبقًا، مع مبرر الاختيار الأساسي قبل رؤية النتائج."),
    },
    "hte": {
        "beginner": _ex("GATE", "احسب GATE لنصفي x1 (سالب/موجب).", "C.gate(phi, groups).", "≈ 0.2 و1.8 في DGP المنصة."),
        "intermediate": _ex("BLP", "استخدم irm.cate مع أساس (1, x1).", "DataFrame للأساس.", "β₀ ≈ 1، β₁ ≈ 1."),
        "research": _ex("Policy", "كيف تقيّم سياسة «عالج إن τ̂(x) > c»؟", "بيانات مستقلة.",
                        "قدّر τ̂ على جزء، وقيّم قيمة السياسة بدرجة AIPW على جزء مستقل؛ قارن بسياسات بسيطة (عالج الجميع/لا أحد)."),
    },
    "doubleml_package": {
        "beginner": _ex("DoubleMLData", "أنشئ DoubleMLData لـy وd وx1..x5.", "y_col, d_cols, x_cols.",
                        "dml.DoubleMLData(df, y_col='y', d_cols='d', x_cols=['x1', 'x2', 'x3', 'x4', 'x5'])."),
        "intermediate": _ex("Bootstrap", "احسب فترات مشتركة لمعالجتين.", "bootstrap + confint(joint=True).",
                            "plr.bootstrap(n_rep_boot=1000); plr.confint(joint=True)."),
        "research": _ex("DoWhy refutation", "صف اختبار دحض بـPlacebo treatment.", "استبدل D بعشوائي.",
                        "استبدل D بمتغير عشوائي مستقل وأعد التقدير؛ يجب أن يكون الأثر ≈ 0؛ غير ذلك يكشف خللًا في الإجراء."),
    },
    "panel_did": {
        "beginner": _ex("ATT(g, t)", "ماذا يعني ATT(4, 6)؟", "مجموعة وفترة.", "أثر المعالجة في الفترة 6 على الوحدات التي بدأت معالجتها في الفترة 4."),
        "intermediate": _ex("مقارنة", "قارن never_treated مع not_yet_treated في المختبر.", "control_group.",
                            "متقاربان هنا؛ not_yet_treated يستخدم بيانات أكثر بافتراض اتجاهات متوازية أقوى."),
        "research": _ex("Anticipation", "كيف تتعامل مع توقع المعالجة بفترة؟", "anticipation_periods=1.",
                        "تُزاح فترة الأساس إلى g − 2 بدل g − 1؛ يجب تبرير عدد فترات التوقع مسبقًا."),
    },
    "dml_extensions": {
        "beginner": _ex("RV", "RV = 25%. فسّر.", "قوة المربك.", "مربك يفسّر 25% من التباين المتبقي في Y وفي D معًا يكفي لإلغاء الأثر."),
        "intermediate": _ex("Benchmark", "استخدم sensitivity_benchmark مع x1 كمرجع.", "benchmarking_set.",
                            "irm.sensitivity_benchmark(benchmarking_set=['x1']) يعطي cf_y وcf_d «بقوة x1» لمقارنتها بـRV."),
        "research": _ex("كمّيات", "متى تفضّل QTE على ATE؟", "توزيع.", "حين يهم أثر البرنامج على أسفل التوزيع (فقر، مخاطر) أو يُتوقع أثر غير متماثل."),
    },
    "dml_monte_carlo": {
        "beginner": _ex("κ = 0", "شغّل المحاكاة مع κ = 0. ماذا يحدث للمقدِّر الساذج؟", "لا إرباك.", "غير متحيز: لا توجد مسارات خلفية."),
        "intermediate": _ex("Lasso وλ = 1", "قارن Lasso + poly(2) مع RF عند λ = 1.", "راجع التغطية.",
                            "Lasso + poly قد يلتقط جزءًا من اللاخطية؛ RF أقرب عادةً للتغطية الاسمية."),
        "research": _ex("تصميم", "صمم محاكاة لإظهار فشل DML بتداخل ضعيف في PLR.", "Var(V) صغير.",
                        "صغّر تباين V (أو كبّر κ) ليصبح Var(D|X) ضئيلًا: تضخم SE وتنخفض التغطية مع n صغير."),
    },
}
