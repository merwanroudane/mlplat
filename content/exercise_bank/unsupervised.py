"""Exercises: unsupervised learning."""


def _ex(title, task, hint, solution):
    return {"title": title, "task": task, "hint": hint, "solution": solution}


EXERCISES: dict[str, dict] = {
    "kmeans": {
        "beginner": _ex("خطوة يدوية", "نقاط 1, 2, 9, 10 ومراكز 1 و2. نفّذ تعيينًا وتحديثًا.", "أقرب مركز.",
                        "تعيين: {1} ⇒ c1، {2, 9, 10} ⇒ c2. تحديث: c1 = 1، c2 = 7. تعيين: {1, 2} و{9, 10}. تحديث: 1.5 و9.5."),
        "intermediate": _ex("Silhouette", "احسب silhouette_score لـk من 2 إلى 8 على blobs.", "sklearn.metrics.", "الأعلى عادة عند k = 4."),
        "research": _ex("استقرار", "صف إجراء استقرار العناقيد عبر Bootstrap.", "ARI بين تشغيلات.",
                        "أعد التجميع على عينات Bootstrap، طابق التسميات على التقاطع، احسب ARI المتوسط لكل k؛ k المستقر أكثر مصداقية."),
    },
    "hierarchical": {
        "beginner": _ex("Dendrogram", "ارسم Dendrogram بـscipy للأبواب الأربعة وقارن.", "scipy.cluster.hierarchy.dendrogram.",
                        "ward وcomplete متقاربان على الكتل؛ single يسلسل."),
        "intermediate": _ex("distance_threshold", "اقطع الشجرة بارتفاع بدل عدد عناقيد.", "n_clusters=None.",
                            "AgglomerativeClustering(n_clusters=None, distance_threshold=5.0)."),
        "research": _ex("Cophenetic", "احسب الترابط الكوفينيتيكي لكل linkage.", "scipy.cluster.hierarchy.cophenet.",
                        "average غالبًا الأعلى؛ يقيس مدى حفظ الشجرة للمسافات الأصلية."),
    },
    "dbscan": {
        "beginner": _ex("eps", "على moons مع min_samples = 5، جد eps يعطي عنقودين.", "منحنى k-distance.", "حوالي 0.2–0.3 بعد القياس."),
        "intermediate": _ex("HDBSCAN", "قارن DBSCAN وHDBSCAN على «varied densities» بـARI.", "min_cluster_size=15.",
                            "HDBSCAN أعلى عادةً لأنه يتكيف مع الكثافة."),
        "research": _ex("تعقيد", "لماذا يتدهور DBSCAN في الأبعاد العالية؟", "فهارس مكانية + لعنة الأبعاد.",
                        "الفهارس تفقد كفاءتها (O(n²))، والمسافات تتقارب فيصبح eps بلا معنى."),
    },
    "gmm": {
        "beginner": _ex("مسؤوليات", "اطبع predict_proba لأول 5 نقاط.", "GaussianMixture.predict_proba.", "صفوف مجموعها 1."),
        "intermediate": _ex("BIC", "اختر K ونوع التغاير بـBIC على بياناتك.", "حلقة مزدوجة.", "انظر جدول BIC في الصفحة."),
        "research": _ex("K-Means كحالة حدية", "بيّن أن K-Means = GMM بتغاير σ²I وσ → 0.", "γ عند σ → 0.",
                        "γ_ik → 1 لأقرب مركز و0 لغيره ⇒ تعيين صلب؛ M-step يصبح المتوسط ⇒ Lloyd."),
    },
    "pca": {
        "beginner": _ex("تباين مفسَّر", "كم مكوّنًا يحتاج Wine المقاس لبلوغ 80%؟", "np.cumsum(explained_variance_ratio_).", "5 مكونات تقريبًا."),
        "intermediate": _ex("PCA(0.95)", "مرّر n_components=0.95 واطبع n_components_.", "كسر = نسبة تباين.", "PCA(0.95).fit(Xs).n_components_."),
        "research": _ex("PCR vs PLS", "لماذا قد يفشل PCR تنبؤيًا؟", "الاتجاهات تُختار دون y.",
                        "قد يكون الاتجاه المهم لـy منخفض التباين في X فيُحذف؛ PLS يختار اتجاهات تعظّم التغاير مع y."),
    },
    "manifold": {
        "beginner": _ex("perplexity", "شغّل t-SNE بـperplexity 5 و50. ماذا تغيّر؟", "بنية محلية/عالمية.", "5: جزر صغيرة كثيرة؛ 50: كتل أكبر أنعم."),
        "intermediate": _ex("Kernel PCA", "اضبط gamma لـKernel PCA لفصل الدوائر.", "RBF.", "gamma بين 2 و15 عادة بعد القياس."),
        "research": _ex("الاستقرار", "كيف تتحقق أن عنقودًا في t-SNE حقيقي؟", "تحقق في الفضاء الأصلي.",
                        "كرر بعدة بذور/perplexity، واختبر التجميع في الفضاء الأصلي (Silhouette، استقرار)، وقارن مع PCA."),
    },
    "anomaly_detection": {
        "beginner": _ex("Isolation Forest", "درّب IsolationForest واطبع أعلى 10 درجات شذوذ.", "score_samples (أقل = أشذ).",
                        "np.argsort(model.score_samples(X))[:10]."),
        "intermediate": _ex("Novelty", "درّب LOF(novelty=True) على بيانات طبيعية فقط وطبّقه على بيانات جديدة.", "fit على النظيف.",
                            "LocalOutlierFactor(novelty=True).fit(X_normal).predict(X_new)."),
        "research": _ex("تقييم بلا تسميات", "كيف تقيّم كاشف شذوذ دون تسميات؟", "مراجعة بشرية/حقن.",
                        "حقن شذوذ اصطناعي معروف، أو مراجعة خبراء لعينة الأعلى درجة (Precision@k)، أو مقارنة الاستقرار عبر الطرق."),
    },
}
