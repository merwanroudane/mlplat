"""Quizzes: unsupervised learning."""

QUIZZES: dict[str, list[dict]] = {
    "kmeans": [
        {"q": "في خطوة التحديث، المركز الجديد هو:", "type": "mcq", "options": ["الوسيط", "متوسط نقاط العنقود", "أبعد نقطة", "نقطة عشوائية"],
         "answer": 1, "explain": "المتوسط يقلل مجموع المربعات لمجموعة ثابتة."},
        {"q": "لماذا لا نختار k بأقل Inertia؟", "type": "hyper", "options": ["لأنها بطيئة", "لأنها تتناقص دائمًا مع k (k = n ⇒ 0)", "لأنها سالبة", "بل نختاره بها"],
         "answer": 1, "explain": "استخدم Elbow/Silhouette/الاستقرار."},
        {"q": "في Animation، ماذا حدث للـInertia عبر الخطوات؟", "type": "chart", "options": ["تزايدت", "لم تتزايد أبدًا", "تذبذبت", "ثابتة"],
         "answer": 1, "explain": "كلا الخطوتين تقلل الهدف أو تبقيه."},
        {"q": "K-Means على الهلالين:", "type": "scenario", "options": ["يفصلهما تمامًا", "يقطعهما بخط لأنه يفترض عناقيد كروية", "يرفض العمل", "يجد 3"],
         "answer": 1, "explain": "استخدم DBSCAN."},
    ],
    "hierarchical": [
        {"q": "أي ربط يلتقط الأشكال السلسلية غير المحدبة؟", "type": "mcq", "options": ["ward", "complete", "single", "average"], "answer": 2,
         "explain": "أقرب زوج يسمح بالتسلسل."},
        {"q": "ذاكرة التجميع الهرمي:", "type": "mcq", "options": ["O(n)", "O(n²)", "O(log n)", "O(1)"], "answer": 1, "explain": "مصفوفة المسافات."},
    ],
    "dbscan": [
        {"q": "نقطة ضمن eps من Core لكنها ليست Core:", "type": "mcq", "options": ["Noise", "Border", "Core", "Outlier"], "answer": 1,
         "explain": "تنضم للعنقود كحدودية."},
        {"q": "كيف تختار eps؟", "type": "hyper", "options": ["0.5 دائمًا", "ركبة منحنى k-distance مع k = min_samples", "Inertia", "عشوائيًا"],
         "answer": 1, "explain": "يفصل المناطق الكثيفة عن الضجيج."},
        {"q": "عنقود كثيف وآخر متفرق. الأنسب:", "type": "method", "options": ["DBSCAN بـeps واحد", "HDBSCAN", "K-Means", "PCA"], "answer": 1,
         "explain": "يتكيف مع الكثافات."},
    ],
    "gmm": [
        {"q": "المسؤولية γ_ik هي:", "type": "mcq", "options": ["مسافة", "احتمال انتماء النقطة i للمكوّن k", "وزن المكوّن", "BIC"], "answer": 1,
         "explain": "عضوية ناعمة من E-step."},
        {"q": "لاختيار K في GMM نستخدم:", "type": "method", "options": ["أعلى log-likelihood", "BIC", "Accuracy", "Inertia"], "answer": 1,
         "explain": "الأرجحية تزيد دائمًا مع K."},
    ],
    "pca": [
        {"q": "المكوّن الرئيسي الأول هو:", "type": "mcq", "options": ["أكبر خاصية", "المتجه الذاتي لأكبر قيمة ذاتية للتغاير", "المتوسط", "أول عمود"],
         "answer": 1, "explain": "اتجاه أقصى تباين."},
        {"q": "دون قياس على Wine، PC1 يسيطر عليه proline لأن:", "type": "chart", "options": ["أهم كيميائيًا", "مقياسه الأكبر يعطيه أكبر تباين",
                                                                                            "خطأ", "مرتبط بالهدف"], "answer": 1,
         "explain": "PCA حساس للوحدات."},
        {"q": "صح أم خطأ: إشارة المكوّن ذات معنى ثابت بين التشغيلات.", "type": "tf", "options": ["صح", "خطأ"], "answer": 1,
         "explain": "v و−v متجهان ذاتيان للقيمة نفسها."},
    ],
    "manifold": [
        {"q": "أي استنتاج مسموح من خريطة t-SNE؟", "type": "mcq", "options": ["المسافة بين العناقيد", "حجم العنقود", "النقاط المتجاورة متشابهة محليًا",
                                                                                 "عدد العناقيد الحقيقي"], "answer": 2,
         "explain": "t-SNE يحفظ الجوار المحلي فقط."},
        {"q": "لماذا Kernel PCA يفصل الدوائر المتحدة المركز؟", "type": "mcq", "options": ["يدورها", "يعمل في فضاء نواة غير خطي", "يقيسها", "صدفة"],
         "answer": 1, "explain": "RBF يلتقط البعد عن المركز."},
    ],
    "anomaly_detection": [
        {"q": "Isolation Forest يعتبر النقطة شاذة إذا:", "type": "mcq", "options": ["مسار عزلها طويل", "مسار عزلها قصير", "كثافتها عالية", "قريبة من المركز"],
         "answer": 1, "explain": "الشاذ يُعزل بتقسيمات قليلة."},
        {"q": "contamination يغيّر:", "type": "hyper", "options": ["score_samples", "العتبة فقط (predict)", "الأشجار", "القياس"], "answer": 1,
         "explain": "الدرجة المستمرة لا تتغير."},
        {"q": "لديك 5000 مثال احتيال مُعلَّم. الأنسب:", "type": "method", "options": ["Isolation Forest", "تصنيف موجّه غير متوازن", "K-Means", "PCA"],
         "answer": 1, "explain": "التسميات أثمن من افتراض «الغريب = المهم»."},
    ],
}
