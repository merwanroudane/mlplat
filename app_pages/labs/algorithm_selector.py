import pandas as pd
import streamlit as st

from components.callouts import mistakes, warning
from content.algorithm_metadata import ALGORITHMS
from core.curriculum import get_module
from core.page import footer, page_header

page_header("algorithm_selector")
warning("هذا مساعد قرار لا حَكَم: يعرض **مرشحين** بمزاياهم وعيوبهم لتقارنهم بالتحقق المتقاطع على بياناتك. لا يوجد «فائز مطلق» "
        "(No Free Lunch).", title=":material/balance: كيف تستخدمه")

with st.form("selector"):
    c1, c2, c3 = st.columns(3)
    task = c1.selectbox("1. المهمة", ["classification", "regression", "clustering", "anomaly", "dimred"],
                        format_func={"classification": "تصنيف", "regression": "انحدار", "clustering": "تجميع", "anomaly": "كشف شذوذ",
                                     "dimred": "تقليل أبعاد"}.get)
    n = c2.selectbox("2. عدد العينات", ["< 1,000", "1,000 – 100,000", "> 100,000"])
    p = c3.selectbox("3. عدد الخصائص", ["< 20", "20 – 500", "> 500 (متفرقة/نصوص)"])
    c4, c5, c6 = st.columns(3)
    linear = c4.selectbox("4. العلاقة المتوقعة", ["غير معروفة", "قريبة من الخطية", "غير خطية/تفاعلات"])
    cats = c5.selectbox("5. متغيرات فئوية", ["لا/قليلة", "كثيرة أو عالية الكاردينالية"])
    missing = c6.selectbox("6. قيم مفقودة", ["لا", "نعم ولا أريد التعويض"])
    c7, c8, c9 = st.columns(3)
    imbalance = c7.selectbox("7. عدم توازن (للتصنيف)", ["لا", "نعم"])
    interp = c8.selectbox("8. أهمية التفسير", ["منخفضة", "متوسطة", "عالية (منظم/تبرير)"])
    latency = c9.selectbox("9. زمن التنبؤ", ["غير مهم", "منخفض جدًا مطلوب"])
    c10, c11, c12 = st.columns(3)
    proba = c10.selectbox("10. احتمالات موثوقة", ["غير مطلوبة", "مطلوبة"])
    structure = c11.selectbox("11. بنية زمنية/مجموعات", ["لا", "زمنية", "مجموعات/Panel"])
    budget = c12.selectbox("12. ميزانية الحوسبة", ["محدودة", "متوسطة", "كبيرة"])
    submitted = st.form_submit_button("اعرض المرشحين", type="primary", icon=":material/assistant_direction:")

if submitted or st.session_state.get("sel_done"):
    st.session_state["sel_done"] = True
    rows = []
    for a in ALGORITHMS:
        if task not in a.tasks:
            continue
        if a.module == "imb_ensembles" and imbalance != "نعم":
            continue  # balanced ensembles only make sense for imbalanced classification
        score, pros, cons = 0.0, [], []
        if a.module == "imb_ensembles":
            score += 0.5; pros.append("مصمم لعدم التوازن (إعادة موازنة داخل كل مكوّن)")
            cons.append("احتمالات غير معايرة؛ قارنه بنموذج قوي + عتبة مضبوطة")
        if n == "> 100,000":
            if a.id in ("svm", "knn", "hierarchical", "lof", "ocsvm"):
                score -= 2; cons.append("لا يتوسع جيدًا مع n كبير")
            if a.id in ("hist_gb", "lightgbm", "xgboost", "logistic", "linear_regression", "ridge", "isolation_forest", "kmeans"):
                score += 1; pros.append("يتوسع مع n كبير")
        if n == "< 1,000":
            if a.ensemble and a.id in ("lightgbm", "xgboost", "catboost"):
                score -= 0.5; cons.append("يحتاج ضبطًا حذرًا مع بيانات قليلة")
            if a.linear or a.id in ("gaussian_nb", "lda"):
                score += 1; pros.append("مستقر مع بيانات قليلة")
        if p.startswith("> 500"):
            if a.id in ("logistic", "lasso", "ridge", "elastic_net", "multinomial_nb", "svm"):
                score += 1.5; pros.append("ممتاز في الأبعاد العالية المتفرقة")
            if a.id in ("knn", "qda", "gmm", "hierarchical"):
                score -= 1.5; cons.append("يتدهور في الأبعاد العالية")
        if linear == "قريبة من الخطية" and a.linear:
            score += 1.5; pros.append("يطابق الافتراض الخطي")
        if linear == "غير خطية/تفاعلات":
            if a.linear:
                score -= 1; cons.append("يحتاج خصائص يدوية للاخطية")
            elif a.supervised:
                score += 1; pros.append("يلتقط اللاخطية والتفاعلات")
        if cats.startswith("كثيرة"):
            if a.categorical_native:
                score += 1.5; pros.append("يدعم الفئات أصليًا")
            elif a.supervised:
                cons.append("يحتاج ترميزًا (One-hot/Target encoding)")
        if missing.startswith("نعم"):
            if a.missing_native:
                score += 1; pros.append("يدعم NaN أصليًا")
            else:
                score -= 0.5; cons.append("يحتاج Imputer في Pipeline")
        if imbalance == "نعم" and task == "classification" and a.id in ("logistic", "random_forest", "hist_gb", "xgboost", "lightgbm", "catboost"):
            pros.append("يدعم class_weight/sample_weight وضبط العتبة")
        if interp.startswith("عالية"):
            if a.interpretability == "high":
                score += 2; pros.append("قابل للتفسير مباشرة")
            elif a.interpretability == "low":
                score -= 1.5; cons.append("صندوق أسود (يحتاج SHAP/PDP)")
        if latency.startswith("منخفض"):
            if a.id in ("knn", "svm", "ocsvm", "lof"):
                score -= 1.5; cons.append("تنبؤ بطيء")
            if a.linear or a.id == "catboost":
                score += 0.5; pros.append("تنبؤ سريع")
        if proba == "مطلوبة":
            if a.probabilistic or a.id == "logistic":
                score += 1; pros.append("احتمالات معقولة المعايرة")
            elif a.supervised and task == "classification":
                cons.append("عاير بـCalibratedClassifierCV")
        if budget == "محدودة" and a.id in ("svm", "catboost", "gradient_boosting", "mlp"):
            score -= 0.5; cons.append("مكلف نسبيًا للتدريب/الضبط")
        rows.append({"algo": a, "score": score, "pros": list(dict.fromkeys(pros)), "cons": list(dict.fromkeys(cons))})
    rows.sort(key=lambda r: -r["score"])
    if structure != "لا":
        st.info("لديك بنية " + ("زمنية: استخدم TimeSeriesSplit وخصائص Lag دون تسرب." if structure == "زمنية" else
                                "مجمّعة: استخدم GroupKFold / StratifiedGroupKFold.") + " هذا يخص **التحقق** بصرف النظر عن الخوارزمية.",
                icon=":material/view_week:")
    st.markdown(f"### مرشحون ({len(rows)}) — مرتبون حسب ملاءمة الإجابات، لا حسب «الأفضل»")
    for r in rows[:6]:
        a = r["algo"]
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{a.name}** · `{a.estimator}`  \n{a.problem}")
                if r["pros"]:
                    st.markdown(":green[**مزايا لحالتك:**] " + "؛ ".join(r["pros"]))
                if r["cons"]:
                    st.markdown(":orange[**انتبه:**] " + "؛ ".join(r["cons"]))
            with c2:
                st.page_link(get_module(a.module).file, label="افتح الوحدة", icon=":material/arrow_back:")
    with st.expander("كل الخوارزميات المطابقة للمهمة"):
        st.dataframe(pd.DataFrame([{"algorithm": r["algo"].name, "fit score": r["score"], "pros": "؛ ".join(r["pros"]),
                                    "cautions": "؛ ".join(r["cons"])} for r in rows]), hide_index=True, width="stretch")
    st.page_link(get_module("model_comparison").file, label="الخطوة التالية: قارن المرشحين بإنصاف في مختبر المقارنة",
                 icon=":material/leaderboard:")
mistakes(["اعتبار أعلى مرشح «الأفضل» دون تحقق.", "تجاهل البنية الزمنية/المجمّعة في التحقق.", "نسيان Baseline."])
footer()
