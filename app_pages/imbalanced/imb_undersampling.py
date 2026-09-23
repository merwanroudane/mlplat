import time

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from components.sampler_viz import sampler_visualizer
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least
from utils.datasets import xy
from utils.imbalance import cv_evaluate, make_sampler
from utils.plotting import plot, bars

page_header("imb_undersampling")
imb = optional("imblearn")

st.markdown("Undersampling يقلل الأغلبية بدل تضخيم الأقلية. ثلاث فلسفات مختلفة تمامًا تحمل الاسم نفسه:")
comparison_table([
    {"العائلة": "عشوائي", "الطرق": "RandomUnderSampler", "الهدف": "موازنة سريعة", "ما يُحذف": "أي مثال من الأغلبية عشوائيًا"},
    {"العائلة": "توليد نماذج أولية", "الطرق": "ClusterCentroids", "الهدف": "تلخيص الأغلبية",
     "ما يُحذف": "الأغلبية كلها تُستبدل بمراكز K-Means"},
    {"العائلة": "اختيار موجّه", "الطرق": "NearMiss-1/2/3", "الهدف": "الإبقاء على أمثلة الأغلبية القريبة من الأقلية",
     "ما يُحذف": "أمثلة الأغلبية البعيدة عن الحد"},
    {"العائلة": "تنظيف (Cleaning)", "الطرق": "TomekLinks، ENN، RENN، AllKNN، CNN، OSS، NCR، IHT",
     "الهدف": "إزالة الضجيج والتداخل لا الموازنة", "ما يُحذف": "أمثلة الأغلبية الغامضة أو الزائدة"},
    {"العائلة": "هجين", "الطرق": "SMOTEENN، SMOTETomek", "الهدف": "Oversampling ثم تنظيف",
     "ما يُحذف": "نقاط (أصلية أو اصطناعية) في منطقة التداخل"},
])
warning("طرق التنظيف لا تعطي توازنًا 1:1: TomekLinks مثلًا يحذف بضعة أمثلة فقط. هدفها **حد أنظف**، لا نسب متساوية.")

st.markdown("## التعريفات الأساسية")
definition("Tomek link (Tomek, 1976)",
           "زوج (a, b) من فئتين مختلفتين بحيث يكون كل منهما أقرب جار للآخر. الزوج إما على الحد أو أحدهما ضجيج؛ TomekLinks "
           "يحذف عنصر الأغلبية من كل زوج.")
definition("Edited Nearest Neighbours (Wilson, 1972)",
           "يحذف مثال الأغلبية إذا خالفه جيرانه (kind_sel='all': أي جار مخالف؛ 'mode': أغلبية الجيران). RENN يكرره حتى "
           "الاستقرار، وAllKNN يطبقه بجوار يكبر من 1 إلى k.")
definition("Condensed Nearest Neighbour (Hart, 1968)",
           "يبني مجموعة جزئية «متسقة»: كل مثال مُصنّف صحيحًا بـ1-NN على المجموعة الجزئية. يحتفظ بأمثلة الحد ويحذف الداخلية. "
           "OneSidedSelection = CNN + Tomek links.")
formula(r"\text{NearMiss-1: keep } x \in \text{maj minimizing } \frac{1}{k}\sum_{j=1}^{k} d\big(x, \text{nn}^{\,\text{min}}_j(x)\big)",
        symbols={"k": "n_neighbors (3 افتراضيًا)", r"\text{nn}^{\text{min}}_j(x)": "الجار رقم j من الأقلية"},
        intuition="NearMiss-2 يستخدم أبعد k أمثلة أقلية؛ NearMiss-3 يختار لكل مثال أقلية جيرانه من الأغلبية.")

st.markdown("## مُصوِّر الحذف · Sampler Visualizer")
st.caption("× = مثال أغلبية محذوف. قارن RandomUnderSampler (حذف في كل مكان) بـNearMiss (حذف البعيد) وTomekLinks/ENN (حذف "
           "عند الحد فقط) وClusterCentroids (استبدال بمراكز) والهجين SMOTEENN.")
sampler_visualizer("under", families=("under", "clean", "hybrid"), default="TomekLinks")
intuition("NearMiss-1 يُبقي أمثلة الأغلبية الأقرب إلى الأقلية: حد أوضح للنموذج، لكنه يرمي «الصورة الكاملة» للأغلبية ويتأثر "
          "بشدة بالقيم الشاذة. طرق التنظيف أكثر تحفظًا.")

st.markdown("## التكلفة والأداء داخل CV")


@st.cache_data(show_spinner="5-fold CV لعدة Undersamplers…")
def _compare() -> pd.DataFrame:
    from imblearn.pipeline import make_pipeline as imb_pipe
    X, y = xy("imbalanced")
    lr = lambda: LogisticRegression(max_iter=2000)  # noqa: E731
    models = {"no resampling": make_pipeline(StandardScaler(), lr())}
    for name in ("RandomUnderSampler", "NearMiss-1", "NearMiss-3", "ClusterCentroids", "TomekLinks",
                 "EditedNearestNeighbours", "OneSidedSelection", "NeighbourhoodCleaningRule", "SMOTEENN", "SMOTETomek"):
        models[name] = imb_pipe(StandardScaler(), make_sampler(name), lr())
    rows = []
    for name, m in models.items():
        t = time.perf_counter()
        r = cv_evaluate(m, X, y, n_splits=5).mean(numeric_only=True)
        rows.append({"strategy": name, **{k: r[k] for k in ("PR-AUC (AP)", "ROC-AUC", "recall", "precision", "MCC", "mean p")},
                     "seconds": time.perf_counter() - t})
    return pd.DataFrame(rows)


if imb is None:
    missing_notice("imblearn")
else:
    if st.button("شغّل المقارنة", key="under_run", type="primary", icon=":material/play_arrow:"):
        st.session_state["under_done"] = True
    if st.session_state.get("under_done"):
        res = _compare()
        st.dataframe(res.round(3), hide_index=True, width="stretch")
        plot(bars(res["strategy"], res["PR-AUC (AP)"], title="PR-AUC by strategy", horizontal=True), height=400)
        worst = res.sort_values("PR-AUC (AP)").iloc[0]
        st.markdown(f"أضعف ترتيب هنا: **{worst['strategy']}** (PR-AUC = {worst['PR-AUC (AP)']:.3f}). طرق التنظيف الخفيفة "
                    "(Tomek، ENN) تكاد لا تغيّر النموذج لأنها تحذف أمثلة قليلة؛ الطرق العدوانية ترفع Recall وتفقد معلومات.")

if at_least("advanced"):
    st.markdown("## متقدم: التباين الإضافي في RandomUnderSampler")
    formula(r"\text{kept majority} = n_1 \ll n_0 \ \Rightarrow\ \operatorname{Var}(\hat f)\ \uparrow",
            intuition="كل نموذج يرى n₁ مثالًا من الأغلبية فقط. التجميع عبر عيّنات متعددة (EasyEnsemble، BalancedBagging، "
                      "BalancedRandomForest) يستعيد المعلومات المهدرة — موضوع الصفحة التالية.")
if at_least("research"):
    researcher_note([
        "Batista et al. (2004) قارنوا طرق الموازنة ووجدوا أن الهجين (SMOTE + Tomek/ENN) مفيد خاصة حين تكون الأقلية قليلة جدًا.",
        "InstanceHardnessThreshold يعتمد على تقدير احتمالي بـCV داخلي (cv=5): أضف ذلك إلى تكلفة الحوسبة.",
        "الحذف الموجّه يغيّر توزيع الأغلبية لا نسبتها فقط ⇒ تصحيح الأولوية البسيط لا يكفي لاستعادة المعايرة.",
    ])
why("إن كان لديك بيانات ضخمة، RandomUnderSampler + تجميع خيار حوسبي ممتاز؛ مع بيانات صغيرة، الحذف يرمي معلومات ثمينة.",
    "القرار يعتمد على عدد الأمثلة المطلق لا على النسبة.")
st.markdown("### المراجع")
st.markdown(cite("tomek1976", "wilson1972", "hart1968", "batista2004", "lemaitre2017"))
mistakes(["توقع توازن 1:1 من TomekLinks.", "NearMiss على بيانات فيها قيم شاذة.",
          "Undersampling على بيانات صغيرة أصلًا.", "التقييم على بيانات بعد الحذف."])
page_footer("imb_undersampling",
            takeaways=["عشوائي، نماذج أولية، اختيار موجّه، تنظيف، هجين — أهداف مختلفة.",
                       "التنظيف يحسّن الحد ولا يوازن.", "Undersampling يرفع التباين؛ التجميع يعالجه."])
