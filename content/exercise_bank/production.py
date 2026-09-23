"""Exercises: production ML."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "production_ml": {
        "beginner": _ex("Model card", "اكتب 5 حقول أساسية لـModel card.", "الاستخدام والقيود.",
                        "الاستخدام المقصود، البيانات، المقاييس (بما فيها حسب المجموعات)، القيود والمخاطر، الإصدارات وتاريخ التدريب."),
        "intermediate": _ex("Contract", "اكتب تحققًا بسيطًا لنطاقات الأعمدة قبل predict.", "assert/pandas.",
                            "```python\nbad = ~df['age'].between(18, 100)\nif bad.any(): log_and_quarantine(df[bad])\n```"),
        "research": _ex("Feedback loop", "كيف يلوث نموذج رفض القروض بيانات تدريبه المستقبلية؟", "Selective labels.",
                        "المرفوضون لا تُلاحظ نتائجهم، فتتعلم النماذج اللاحقة من عينة مشروطة بالنموذج السابق؛ الحلول: استكشاف محدود أو تجارب."),
    },
    "mlops": {
        "beginner": _ex("مكونات", "رتّب: Monitor، Train، Validate، Deploy، Register.", "دورة.", "Train → Register → Validate → Deploy → Monitor."),
        "intermediate": _ex("CI", "اقترح 3 اختبارات تُشغَّل آليًا عند تغيير الكود.", "pytest.",
                            "اختبار Schema، اختبار أداء على بيانات مرجعية ثابتة، اختبار أن التنبؤ حتمي مع بذرة ثابتة."),
        "research": _ex("A/B", "لماذا A/B test للنموذج سببي؟", "عشوائية.", "التخصيص العشوائي يلغي الإرباك فيقدّر أثر النموذج على مقياس الأعمال."),
    },
    "drift": {
        "beginner": _ex("PSI", "احسب PSI بين عينتين بـutils.metrics.psi.", "10 صناديق.", "psi(reference, current)."),
        "intermediate": _ex("سياسة", "اكتب سياسة تنبيه وإعادة تدريب.", "عتبات.", "PSI > 0.25 ⇒ تحقيق؛ AUC −0.03 ⇒ إعادة تدريب؛ فشل البوابة ⇒ تراجع."),
        "research": _ex("Covariate shift", "كيف تصحح نموذجًا لانزياح X مع P(y|X) ثابت؟", "نسبة الكثافات.",
                        "أعد وزن التدريب بـw(x) = p_new(x)/p_train(x) (تُقدَّر بمصنف يميز المصدرين)."),
    },
    "fairness": {
        "beginner": _ex("DP", "معدلات القبول 0.4 و0.25. DP gap؟", "فرق مطلق.", "0.15."),
        "intermediate": _ex("عتبات", "جد عتبتين لكل مجموعة تساويان TPR.", "ابحث في شبكة.", "لكل t_a ابحث عن t_b يعطي أقرب TPR؛ Hardt et al. (2016)."),
        "research": _ex("استحالة", "اشرح نتيجة Chouldechova (2017) بجملتين.", "الانتشار.",
                        "إذا اختلف الانتشار الأساسي بين المجموعتين، فمصنف معاير لا يمكن أن يساوي FPR وFNR معًا إلا إن كان مثاليًا."),
    },
    "reproducibility": {
        "beginner": _ex("بذور", "أضف random_state لكل مكوّن في Pipeline فيه RandomForest وKFold.", "كل مصدر عشوائية.",
                        "RandomForestClassifier(random_state=0) وKFold(shuffle=True, random_state=0)."),
        "intermediate": _ex("إصدارات", "اطبع إصدارات الحزم الأساسية برمجيًا.", "importlib.metadata.", "انظر utils/validation.package_versions."),
        "research": _ex("تقرير", "صمم قسم «Reproducibility» لورقة ML.", "قائمة.", "الكود (commit)، البيانات (hash)، الإصدارات، البذور، التقسيم، HPO، الأجهزة، الزمن."),
    },
}
