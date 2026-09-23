"""Exercises: optimization."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "optimization_intro": {
        "beginner": _ex("محدب؟", "هل f(w) = |w| محدبة؟ هل قابلة للاشتقاق عند 0؟", "ارسم الوتر.",
                        "محدبة (الوتر فوق المنحنى) لكنها غير قابلة للاشتقاق عند 0 — مثل عقوبة L1."),
        "intermediate": _ex("المعادلات الطبيعية", "اشتق β̂ لـRidge بمساواة التدرج بالصفر.", "J = ‖y − Xβ‖² + λ‖β‖².",
                            "∇J = −2Xᵀ(y − Xβ) + 2λβ = 0 ⇒ β̂ = (XᵀX + λI)⁻¹Xᵀy."),
        "research": _ex("تكلفة", "قارن تكلفة الحل المغلق O(np² + p³) مع GD O(np) لكل تكرار. متى يفضل كل منهما؟",
                        "κ يحدد عدد التكرارات.", "حين p صغير (≤ بضعة آلاف) الحل المغلق/Cholesky أسرع وأدق؛ حين p كبير أو "
                        "البيانات لا تتسع للذاكرة، GD/SGD أفضل، خاصة مع κ معتدل بعد القياس."),
    },
    "gradient_descent": {
        "beginner": _ex("خطوتان يدويًا", "J(θ) = (θ − 3)²، θ₀ = 0، η = 0.25. احسب θ₁ وθ₂.", "∇J = 2(θ − 3).",
                        "θ₁ = 0 − 0.25·(−6) = 1.5؛ θ₂ = 1.5 − 0.25·(−3) = 2.25."),
        "intermediate": _ex("Mini-batch", "اكتب حلقة mini-batch SGD لانحدار خطي بحجم دفعة 32.", "اخلط الفهارس كل epoch.",
                            "```python\nfor epoch in range(E):\n    idx = rng.permutation(n)\n    for s in range(0, n, 32):\n"
                            "        b = idx[s:s+32]\n        w -= eta * (-2 * X[b].T @ (y[b] - X[b] @ w) / len(b))\n```"),
        "research": _ex("معدل التقارب", "لدالة μ-strongly convex وL-smooth، اشتق عدد التكرارات لبلوغ دقة ε.",
                        "J_k − J* ≤ (1 − μ/L)^k (J_0 − J*).", "k ≥ (L/μ) log((J_0 − J*)/ε) ⇒ يتناسب خطيًا مع رقم التكييف κ = L/μ."),
    },
    "advanced_optimizers": {
        "beginner": _ex("Soft-threshold", "احسب S(z, t) لـz = (−2, 0.3, 1.5) وt = 0.5.", "sign(z)·max(|z| − t, 0).",
                        "(−1.5, 0, 1.0)."),
        "intermediate": _ex("L-BFGS بـSciPy", "استخدم scipy.optimize.minimize(method='L-BFGS-B') لتقليل Rosenbrock من (−1.5, 2).",
                            "مرّر jac.", "يتقارب إلى (1, 1) في بضع عشرات من التكرارات مقارنة بآلاف لـGD."),
        "research": _ex("Early stopping وRidge", "برهن حدسيًا لماذا يعادل Early stopping في GD تنظيم Ridge للانحدار الخطي.",
                        "حلّل كل اتجاه ذاتي.", "بعد k خطوة يصبح المكوّن على الاتجاه j: [1 − (1 − ηλ_j)^k]·β_OLS,j، وهو ≈ "
                        "λ_j/(λ_j + 1/(ηk)) لـRidge بـα ≈ 1/(ηk): الاتجاهات ذات λ_j الصغير لم تُتعلَّم بعد (انكماش)."),
    },
}
