"""Exercises: interpretability."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "model_inspection": {
        "beginner": _ex("permutation_importance", "احسبها لـRandomForest على مجموعة اختبار.", "sklearn.inspection.",
                        "permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)."),
        "intermediate": _ex("Grouped", "اكتب Permutation importance لمجموعة خصائص معًا.", "اخلط الأعمدة بالتبديل نفسه.",
                            "```python\nidx = rng.permutation(len(X)); Xp = X.copy(); Xp[group] = X[group].values[idx]\nbase - score(model, Xp, y)\n```"),
        "research": _ex("Drop-column", "قارن Permutation مع Drop-column importance.", "إعادة التدريب.",
                        "Drop-column يعيد التدريب دون الخاصية فيقيس قيمتها الحدية للنموذج الأمثل؛ مع الترابط تكون قريبة من صفر لكلا المترابطين."),
    },
    "pdp_ice": {
        "beginner": _ex("PDP", "ارسم PDP لـx3 بـPartialDependenceDisplay.", "from_estimator.",
                        "PartialDependenceDisplay.from_estimator(model, X, [2], kind='both')."),
        "intermediate": _ex("تفاعل", "كيف تكتشف التفاعل من ICE؟", "centered ICE.", "مرّر centered=True؛ تباعد المنحنيات بعد التمركز = تفاعل."),
        "research": _ex("PDP سببي", "متى يساوي PDP أثرًا سببيًا؟", "Backdoor.",
                        "إذا حققت الخصائص المثبتة x_C معيار Backdoor للعلاقة بين x_S وy وكان النموذج صحيحًا (Zhao & Hastie, 2021)."),
    },
    "shap": {
        "beginner": _ex("Waterfall", "اختر ملاحظة وفسّر أكبر مساهمة.", "Waterfall.", "مثال: x1 = 1.2 يرفع التنبؤ +3.5 عن الأساس."),
        "intermediate": _ex("خلفية", "قارن قيم SHAP بخلفيتين: كل البيانات مقابل فئة فرعية.", "Explainer(model, background).",
                            "القيم تتغير لأن السؤال تغير: «لماذا يختلف تنبؤ هذه الملاحظة عن متوسط هذه الخلفية؟»."),
        "research": _ex("Interventional vs observational", "اشرح الفرق وعلاقته بالسببية.", "Janzing et al. (2020).",
                        "Interventional يستبدل الخصائص الغائبة بقيم مستقلة من الخلفية (كسر الترابط، تفسير «سببي» للنموذج)؛ Observational يستخدم "
                        "التوزيع الشرطي (يوزع الفضل على المترابطات)."),
    },
}
