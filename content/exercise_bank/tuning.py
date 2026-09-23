"""Exercises: hyperparameter optimisation."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "hpo_foundations": {
        "beginner": _ex("RandomizedSearchCV", "اضبط C وgamma لـSVC بـ30 محاولة لوغاريتمية.", "scipy.stats.loguniform.",
                        "```python\nRandomizedSearchCV(pipe, {'svc__C': loguniform(1e-2, 1e3), 'svc__gamma': loguniform(1e-4, 1)}, n_iter=30, cv=5)\n```"),
        "intermediate": _ex("فضاء شرطي", "اكتب param_grid لـSVC حيث degree يُضبط فقط مع kernel='poly'.", "قائمة قواميس.",
                            "```python\n[{'svc__kernel': ['rbf'], 'svc__gamma': [...]},\n {'svc__kernel': ['poly'], 'svc__degree': [2, 3, 4]}]\n```"),
        "research": _ex("تحيز الاختيار", "بالمحاكاة: 100 نموذج متساوي الأداء الحقيقي 0.8 مع ضجيج SD 0.02؛ ما متوسط max؟",
                        "E[max] لمتغيرات طبيعية.", "≈ 0.8 + 0.02·2.5 ≈ 0.85: تفاؤل 5 نقاط دون أي تحسن حقيقي."),
    },
    "bayesian_optimization": {
        "beginner": _ex("Study", "اكتب دراسة Optuna بـ20 محاولة لتعظيم −(x − 2)².", "suggest_float.",
                        "```python\nstudy = optuna.create_study(direction='maximize')\nstudy.optimize(lambda t: -(t.suggest_float('x', -10, 10) - 2) ** 2, n_trials=20)\n```"),
        "intermediate": _ex("Pruning", "أضف MedianPruner مع تقارير لكل طية.", "trial.report.",
                            "انظر utils/tuning.run_optuna: report(mean_so_far, step) ثم should_prune()."),
        "research": _ex("TPE", "اشرح لماذا تعظيم l(x)/g(x) يعادل تعظيم EI في TPE.", "Bergstra et al. (2011).",
                        "مع تعريف l وg ككثافتي المحاولات دون/فوق المئين γ، يتناسب EI طرديًا مع (γ + g(x)/l(x)·(1 − γ))⁻¹، فيُعظَّم بتعظيم l/g."),
    },
}
