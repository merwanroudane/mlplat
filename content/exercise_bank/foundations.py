"""Exercises: math and statistical-learning foundations."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "linear_algebra": {
        "beginner": _ex("مسافات", "احسب Euclidean وManhattan وCosine بين (1, 2) و(3, 0).", "Cosine = a·b/(‖a‖‖b‖).",
                        "Euclidean = √(4 + 4) = 2.83؛ Manhattan = 4؛ Cosine = 3/(√5·3) = 0.447."),
        "intermediate": _ex("OLS بالـlstsq", "حل المربعات الصغرى لـX = [[1,1],[1,2],[1,3]] وy = [1,2,2] بـnp.linalg.lstsq.",
                            "العمود الأول للحد الثابت.", "β ≈ (0.667, 0.5): ŷ = 0.667 + 0.5x."),
        "research": _ex("Ridge وSVD", "اشتق β̂_ridge بدلالة SVD: X = UΣVᵀ.", "(XᵀX + λI)⁻¹Xᵀy.",
                        "β̂ = V diag(σ_j/(σ_j² + λ)) Uᵀy ⇒ كل اتجاه ينكمش بمعامل σ_j²/(σ_j² + λ)؛ الاتجاهات ذات σ صغير "
                        "تنكمش أكثر."),
    },
    "calculus": {
        "beginner": _ex("مشتقة", "اشتق f(w) = (3 − 2w)² وأوجد النقطة الحرجة.", "قاعدة السلسلة.",
                        "f'(w) = −4(3 − 2w) = 0 ⇒ w = 1.5، وf''= 8 > 0 ⇒ صغرى."),
        "intermediate": _ex("تدرج MSE", "اشتق تدرج L(β) = (1/n)‖y − Xβ‖².", "اشتق بالنسبة لكل β_j.",
                            "∇L = −(2/n)Xᵀ(y − Xβ)؛ مساواته بالصفر تعطي المعادلات الطبيعية XᵀXβ = Xᵀy."),
        "research": _ex("Gradient checking", "اكتب دالة تتحقق عدديًا من تدرج أي f بفروق مركزية.", "[f(w + h e_j) − f(w − h e_j)]/(2h).",
                        "```python\ndef grad_check(f, g, w, h=1e-5):\n    num = np.array([(f(w + h*e) - f(w - h*e)) / (2*h) for e in np.eye(len(w))])\n"
                        "    return np.max(np.abs(num - g(w)))\n```"),
    },
    "probability": {
        "beginner": _ex("Bayes", "انتشار الاحتيال 0.5%، النموذج يلتقط 90% ويعطي إنذارًا كاذبًا لـ2% من السليم. ما Precision؟",
                        "P(fraud | alert).", "0.9·0.005 / (0.9·0.005 + 0.02·0.995) ≈ 0.184."),
        "intermediate": _ex("MLE لـPoisson", "اشتق MLE لـλ من عينة y₁..yₙ ~ Poisson(λ).", "log p = y log λ − λ − log y!.",
                            "∂/∂λ Σ(yᵢ log λ − λ) = Σyᵢ/λ − n = 0 ⇒ λ̂ = ȳ."),
        "research": _ex("Log loss كإمكان", "بيّن أن Log loss هو سالب لوغاريتم إمكان Bernoulli، واستنتج أن المُقلِّل الأمثل هو P(Y=1|X).",
                        "E[−Y log p − (1−Y) log(1−p) | X].", "المشتقة بالنسبة لـp: −π/p + (1−π)/(1−p) = 0 ⇒ p = π = P(Y=1|X)."),
    },
    "statistics_ml": {
        "beginner": _ex("خطأ معياري", "ما SE لمتوسط عينة n = 100 من مجتمع σ = 15؟", "σ/√n.", "15/10 = 1.5."),
        "intermediate": _ex("Bootstrap", "اكتب Bootstrap لفترة ثقة 95% لوسيط عينة.", "np.random.choice مع replace=True.",
                            "```python\nb = [np.median(rng.choice(x, len(x))) for _ in range(2000)]\nnp.percentile(b, [2.5, 97.5])\n```"),
        "research": _ex("VIF", "اشتق VIF لخاصيتين بارتباط ρ.", "Var(β̂₁) ∝ 1/(1 − R₁²).", "R₁² = ρ² ⇒ VIF = 1/(1 − ρ²)."),
    },
    "statistical_learning": {
        "beginner": _ex("قراءة منحنى", "في المختبر، ما الدرجة التي تعطي أقل Test MSE مع n = 40 وσ = 0.3؟", "انظر منحنى U.",
                        "عادة بين 3 و6؛ تتغير مع البذرة وn."),
        "intermediate": _ex("Learning curve", "ارسم learning_curve لـRandomForestRegressor على بيانات الانحدار وفسّره.",
                            "sklearn.model_selection.learning_curve.",
                            "فجوة متوسطة تضيق مع n ⇒ تباين معتدل؛ البيانات الإضافية تفيد قليلًا."),
        "research": _ex("التفاؤل", "استخدم صيغة التفاؤل لإثبات أن تفاؤل OLS = 2pσ²/n.", "Cov(ŷ, y) = σ² tr(H).",
                        "ŷ = Hy مع H مصفوفة الإسقاط ⇒ Σ Cov(ŷᵢ, yᵢ) = σ² tr(H) = σ² p ⇒ التفاؤل = 2pσ²/n."),
    },
    "bias_variance": {
        "beginner": _ex("صنّف", "نموذج: Train MSE = 0.9، Test = 1.0، σ² = 0.1. تحيز أم تباين؟", "قارن مع σ².",
                        "تحيز عالٍ (الخطأان مرتفعان ومتقاربان)."),
        "intermediate": _ex("محاكاة", "أعد المحاكاة بـTree depth. عند أي عمق يتساوى Bias² وVariance؟", "راقب المنحنيين.",
                            "يعتمد على n وσ؛ مع n = 30 وσ = 0.3 عادةً حول العمق 3–4."),
        "research": _ex("Bagging", "بيّن أن متوسط B نموذجًا بتباين σ² وارتباط ρ له تباين ρσ² + (1 − ρ)σ²/B.",
                        "Var(mean) = (1/B²) Σ Σ Cov.", "Var = (1/B²)[Bσ² + B(B−1)ρσ²] = ρσ² + (1 − ρ)σ²/B؛ لذلك يهدف RF إلى خفض ρ."),
    },
    "loss_functions": {
        "beginner": _ex("قيم", "احسب MSE وMAE وHuber(δ=1) لأخطاء r = (0.5, −2, 3).", "Huber: ½r² إن |r| ≤ δ وإلا δ(|r| − δ/2).",
                        "MSE = (0.25 + 4 + 9)/3 = 4.42؛ MAE = 1.83؛ Huber = (0.125 + 1.5 + 2.5)/3 = 1.375."),
        "intermediate": _ex("Quantile regression", "درّب QuantileRegressor لـτ = 0.1 و0.9 وارسم نطاق التنبؤ.", "solver='highs', alpha=0.",
                            "النطاق بين الخطين يغطي ~80% من النقاط على التدريب؛ للتغطية المضمونة انظر Conformal prediction."),
        "research": _ex("Pinball", "أثبت أن المُقلِّل الأمثل لـE[ρ_τ(Y − q)] هو المئين τ.", "اشتق بالنسبة لـq.",
                        "المشتقة = −τP(Y > q) + (1 − τ)P(Y ≤ q) = F(q) − τ = 0 ⇒ q = F⁻¹(τ)."),
    },
    "learning_theory": {
        "beginner": _ex("تمزيق", "هل تُمزِّق فترات [a, b] على خط 3 نقاط؟", "جرب التسمية (1, 0, 1).",
                        "لا: (1, 0, 1) لا تتحقق بفترة واحدة ⇒ VC = 2."),
        "intermediate": _ex("حد", "بـ|H| = 1000 وδ = 0.05، كم n لضمان ε = 0.05؟", "ε = √((ln|H| + ln(2/δ))/(2n)).",
                            "n = (6.91 + 3.69)/(2·0.0025) ≈ 2120."),
        "research": _ex("Donsker وCross-fitting", "لماذا يسمح Cross-fitting باستخدام غابات عشوائية كمتعلمي إزعاج دون شروط تعقيد؟",
                        "ما الذي يصبح مستقلًا؟", "مع Cross-fitting، تُقيَّم الدرجة على طية لم تُستخدم لتقدير الإزعاج؛ بمعلومية "
                        "الإزعاج المقدَّر تصبح الحدود مجاميع i.i.d.، فيكفي تقارب الإزعاج بمعدل o(n^{-1/4}) دون قيود Donsker."),
    },
}
