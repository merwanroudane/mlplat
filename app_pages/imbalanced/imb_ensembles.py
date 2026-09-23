import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from components.algorithm_profile import algorithm_profile, hyperparameter_table
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least, log_experiment
from utils.datasets import xy
from utils.imbalance import ENSEMBLES, cv_evaluate, make_balanced_ensemble
from utils.plotting import bars, plot

page_header("imb_ensembles")
imb = optional("imblearn")

st.markdown("إعادة العيّنة مرة واحدة ترمي معلومات (Undersampling) أو تكرر (Oversampling). **التجميعات المتوازنة** تعيد العيّنة "
            "**داخل كل مكوّن** فترى الأغلبية كلها عبر المكونات، وتقلل التباين بالتجميع — فكرة Breiman نفسها مطبقة على عدم التوازن.")
mermaid("flowchart LR\n  D[training fold] --> S1[balanced sample 1] --> M1[model 1]\n  D --> S2[balanced sample 2] --> M2[model 2]\n"
        "  D --> S3[balanced sample B] --> M3[model B]\n  M1 --> A[average / vote]\n  M2 --> A\n  M3 --> A")
comparison_table([
    {"التجميع": "BalancedRandomForestClassifier", "المكوّن": "شجرة عشوائية", "إعادة العيّنة": "Bootstrap متوازن لكل شجرة",
     "المرجع": "Chen, Liaw & Breiman (2004)"},
    {"التجميع": "BalancedBaggingClassifier", "المكوّن": "أي مقدر (شجرة افتراضيًا)",
     "إعادة العيّنة": "RandomUnderSampler لكل حقيبة (أو أي sampler)", "المرجع": "Bagging + imbalanced-learn"},
    {"التجميع": "EasyEnsembleClassifier", "المكوّن": "AdaBoost كامل", "إعادة العيّنة": "عيّنة متوازنة لكل AdaBoost",
     "المرجع": "Liu, Wu & Zhou (2009)"},
    {"التجميع": "RUSBoostClassifier", "المكوّن": "مقدر ضعيف في Boosting", "إعادة العيّنة": "RandomUnderSampler قبل كل مرحلة",
     "المرجع": "Seiffert et al. (2010)"},
    {"التجميع": "RandomForest(class_weight='balanced_subsample')", "المكوّن": "شجرة", "إعادة العيّنة": "لا؛ أوزان لكل Bootstrap",
     "المرجع": "scikit-learn"},
])

st.markdown("## بطاقات الخوارزميات والمعاملات الموثقة")
tabs = st.tabs(["Balanced Random Forest", "Balanced Bagging", "EasyEnsemble", "RUSBoost"])
for tab, (aid, est) in zip(tabs, [("balanced_rf", "BalancedRandomForestClassifier"),
                                  ("balanced_bagging", "BalancedBaggingClassifier"),
                                  ("easy_ensemble", "EasyEnsembleClassifier"), ("rusboost", "RUSBoostClassifier")]):
    with tab:
        algorithm_profile(aid)
        hyperparameter_table(est)
warning("منذ imbalanced-learn 0.13 أصبحت قيم BalancedRandomForestClassifier الافتراضية sampling_strategy='all' "
        "وreplacement=True وbootstrap=False (تطابق خوارزمية Chen et al.). كود قديم يعتمد على القيم السابقة سيعطي نتائج مختلفة. "
        "القيم أعلاه مقروءة من المكتبة المثبتة ومختبرة آليًا.", title="تغيّر في القيم الافتراضية")
warning("RUSBoost بمقدّره الافتراضي (جذع بعمق 1) وlearning_rate=1.0 قد يفشل أثناء fit برسالة «worse than random» على بيانات "
        "متداخلة (حدث ذلك على بيانات الاحتيال في المنصة بعمق 1 و2). المختبر يستخدم شجرة بعمق 3 وlearning_rate=0.1.",
        title="هشاشة RUSBoost")

st.markdown("## مختبر التجميعات المتوازنة · Balanced Ensemble Lab")
st.caption("بيانات الاحتيال (5%)، 5-fold stratified CV، القرار بـpredict (العتبة 0.5). نقارن بالبدائل البسيطة بإنصاف.")


@st.cache_data(show_spinner="تدريب ثمانية تجميعات × 5 طيات…")
def _compare() -> pd.DataFrame:
    X, y = xy("imbalanced")
    models = {
        "RandomForest": RandomForestClassifier(100, random_state=0, n_jobs=1),
        "RandomForest balanced_subsample": RandomForestClassifier(100, class_weight="balanced_subsample",
                                                                  min_samples_leaf=3, random_state=0, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=0),
        "HistGradientBoosting balanced": HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
    }
    for name in ENSEMBLES:
        models[name] = make_balanced_ensemble(name, n_estimators=10 if name == "EasyEnsembleClassifier" else 100)
    rows = []
    for name, m in models.items():
        t = time.perf_counter()
        r = cv_evaluate(m, X, y, n_splits=5).mean(numeric_only=True)
        rows.append({"model": name, **{k: r[k] for k in ("PR-AUC (AP)", "ROC-AUC", "recall", "precision", "F1", "MCC",
                                                          "Brier", "mean p")}, "seconds": time.perf_counter() - t})
    return pd.DataFrame(rows)


if imb is None:
    missing_notice("imblearn")
else:
    if st.button("شغّل المقارنة (≈ 30 ثانية)", key="ens_run", type="primary", icon=":material/play_arrow:"):
        st.session_state["ens_done"] = True
    if st.session_state.get("ens_done"):
        res = _compare()
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        plot(bars(res["model"], res["PR-AUC (AP)"], title="PR-AUC (ranking quality)", horizontal=True), height=380)
        best = res.sort_values("PR-AUC (AP)", ascending=False).iloc[0]
        top_recall = res.sort_values("recall", ascending=False).iloc[0]
        st.markdown(f"- أفضل ترتيب (PR-AUC): **{best['model']}** = {best['PR-AUC (AP)']:.3f}.\n"
                    f"- أعلى Recall عند 0.5: **{top_recall['model']}** = {top_recall['recall']:.3f} "
                    f"(Precision = {top_recall['precision']:.3f}).\n"
                    "- قارن «mean p» بالانتشار (≈ 0.05): التجميعات المتوازنة ترفعه كثيرًا ⇒ احتمالاتها ليست مخاطر حقيقية.\n"
                    "- الاستنتاج المنصف يتطلب مقارنة التجميع المتوازن بنموذج قوي **مع عتبة مضبوطة**، لا بالعتبة 0.5.")
        if st.button("سجّل التجربة", key="ens_log", icon=":material/bookmark_add:"):
            log_experiment("Balanced Ensemble Lab", "8 ensembles", {"cv": 5},
                           {r["model"]: round(r["PR-AUC (AP)"], 4) for _, r in res.iterrows()}, seed=0, dataset="imbalanced",
                           split="5-fold stratified")
            st.toast("سُجّلت.", icon=":material/check:")
intuition("التجميعات المتوازنة ممتازة حين تريد Recall عاليًا «جاهزًا» وسريعًا. لكنها لا تضيف معلومات: ما تكسبه في Recall عند "
          "0.5 يمكن غالبًا الحصول عليه من غابة عادية بعتبة أقل.")

if at_least("advanced"):
    st.markdown("## متقدم: لماذا يعمل BalancedRandomForest؟")
    st.markdown("كل شجرة تتعلم حدًا متوازنًا (لا تنحاز للأغلبية)، والتجميع عبر أشجار رأت أجزاء مختلفة من الأغلبية يقلل التباين "
                "الناتج عن رمي معظمها في كل شجرة. مع أقلية صغيرة جدًا، كل شجرة تُبنى على ~2·n₁ صف فقط — تباين الشجرة المفردة "
                "كبير، فزد n_estimators.")
if at_least("research"):
    researcher_note([
        "EasyEnsemble مكلف (n_estimators × مراحل AdaBoost)؛ BalancedBagging مع شجرة بديل أرخص بسلوك مشابه.",
        "BalancedBaggingClassifier(sampler=SMOTE()) يعطي عائلة «SMOTEBagging»؛ جرّبه كمعامل فائق لا كحقيقة.",
        "عند نشر نموذج متوازن كمصدر مخاطر، عاير احتمالاته أو صحّحها للانتشار الحقيقي (الصفحة التالية).",
    ])
why("قارن دائمًا بـ: نموذج قوي عادي + عتبة مضبوطة بالتحقق.", "هذا الخط الأساس هو ما يجب أن يتفوق عليه أي تجميع متوازن.")
st.markdown("### المراجع")
st.markdown(cite("chen2004brf", "liu2009", "seiffert2010", "breiman1996", "lemaitre2017"))
mistakes(["مقارنة التجميع المتوازن بغابة عادية عند العتبة 0.5 فقط.", "الاعتماد على القيم الافتراضية القديمة لـBalancedRandomForest.",
          "استخدام احتمالات التجميع المتوازن كمخاطر.", "RUSBoost بالإعدادات الافتراضية دون فحص الاستقرار."])
page_footer("imb_ensembles",
            takeaways=["إعادة العيّنة داخل كل مكوّن + تجميع = معلومات أكثر وتباين أقل.",
                       "BRF وBalancedBagging يقبلان NaN؛ EasyEnsemble وRUSBoost لا.",
                       "الاحتمالات غير معايرة؛ قارن بنموذج قوي بعتبة مضبوطة."])
