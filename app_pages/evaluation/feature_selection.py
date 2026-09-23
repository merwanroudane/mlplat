import numpy as np
import streamlit as st
from sklearn.feature_selection import (RFECV, SelectFromModel, SelectKBest, SequentialFeatureSelector, VarianceThreshold,
                                       f_regression, mutual_info_regression)
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from core.page import page_footer, page_header, page_link
from core.state import at_least
from utils.datasets import xy
from utils.plotting import lines, plot

page_header("feature_selection")

comparison_table([
    {"العائلة": "Filter", "الفكرة": "إحصائية لكل خاصية مستقلة عن النموذج", "أمثلة": "VarianceThreshold، SelectKBest(f_regression / mutual_info)",
     "المزايا": "سريع", "العيوب": "يتجاهل التفاعلات والتكرار"},
    {"العائلة": "Wrapper", "الفكرة": "بحث عن مجموعات بتقييم نموذج", "أمثلة": "RFE، RFECV، SequentialFeatureSelector",
     "المزايا": "يراعي النموذج", "العيوب": "مكلف وخطر Overfitting"},
    {"العائلة": "Embedded", "الفكرة": "الاختيار جزء من التدريب", "أمثلة": "Lasso (L1)، أهمية الأشجار + SelectFromModel",
     "المزايا": "متوازن", "العيوب": "يرث تحيزات النموذج"},
])
warning("أي اختيار يتعلم من y يجب أن يحدث داخل CV (داخل Pipeline). رأينا في وحدة التسرب أن الاختيار على كل البيانات يعطي دقة "
        "عالية من ضجيج خالص.")
page_link("leakage", "راجع: Feature-selection leakage", ":material/water_drop:")

X, y = xy("high_dim")
st.markdown(f"البيانات: n = {len(y)}، p = {X.shape[1]}، والحقيقة: الخصائص x000..x009 فقط غير صفرية.")

st.markdown("## Selection Lab")
method = st.selectbox("الطريقة", ["VarianceThreshold", "SelectKBest (f_regression)", "SelectKBest (mutual information)",
                                  "L1 / SelectFromModel(LassoCV)", "RFECV (Ridge)", "SequentialFeatureSelector (forward)"], key="fs_m")
k = st.slider("k (للطرق التي تحتاجه)", 2, 40, 10, key="fs_k")
cv = KFold(5, shuffle=True, random_state=0)


def _selector(method: str, k: int):
    if method == "VarianceThreshold":
        return VarianceThreshold(threshold=0.5)
    if method.startswith("SelectKBest (f"):
        return SelectKBest(f_regression, k=k)
    if method.startswith("SelectKBest (m"):
        return SelectKBest(lambda X, y: mutual_info_regression(X, y, random_state=0), k=k)
    if method.startswith("L1"):
        return SelectFromModel(LassoCV(cv=3, random_state=0))
    if method.startswith("RFECV"):
        return RFECV(Ridge(alpha=1.0), step=5, cv=3, min_features_to_select=2)
    return SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=3)


@st.cache_data(show_spinner="يختار داخل كل طية ويقيّم…", max_entries=24)
def _run(method: str, k: int):
    pipe = make_pipeline(StandardScaler(), _selector(method, k), LinearRegression())
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    pipe.fit(X, y)
    mask = pipe[1].get_support()
    extra = None
    if method.startswith("RFECV"):
        extra = pipe[1].cv_results_["mean_test_score"], pipe[1].cv_results_["n_features"]
    return scores, mask, extra


scores, mask, extra = _run(method, k)
chosen = np.array(X.columns)[mask]
true = set(f"x{i:03d}" for i in range(10))
with st.container(horizontal=True):
    st.metric("CV R² (selection inside folds)", f"{scores.mean():.3f}", f"± {scores.std():.3f}", delta_color="off", border=True)
    st.metric("خصائص مختارة", int(mask.sum()), border=True)
    st.metric("حقيقية ملتقطة (من 10)", len(true & set(chosen)), border=True)
    st.metric("ضجيج مختار", len(set(chosen) - true), border=True)
full = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=cv, scoring="r2").mean()
st.caption(f"للمقارنة: كل الخصائص الـ100 دون اختيار ⇒ CV R² = {full:.3f} (n = 200 مع p = 100: تباين عالٍ).")
if extra is not None:
    sc, nf = extra
    plot(lines(nf, {"RFECV mean CV score": sc}, title="RFECV: score vs number of features", xaxis="n features", yaxis="R²",
               markers=True), height=300)
if method == "VarianceThreshold":
    st.info("VarianceThreshold لا يرى y إطلاقًا (غير موجّه): يحذف الخصائص شبه الثابتة فقط. بعد القياس كل التباينات = 1، لذا يُطبَّق "
            "عادة على البيانات الخام.")
intuition("Mutual information يلتقط علاقات غير خطية لكنه متقلب مع n صغير؛ f_regression خطي وسريع. RFE يحذف الأضعف تكراريًا.")

st.markdown("## Permutation importance كأداة اختيار؟")
why("لا تستخدم Permutation importance المحسوبة على التدريب للاختيار ثم تقيّم على البيانات نفسها.",
    "هي أداة تفسير لنموذج مدرّب؛ للاختيار ضعها داخل حلقة CV أو استخدم RFECV/SelectFromModel.")
page_link("model_inspection", "Permutation importance في وحدة التفسير", ":material/search_insights:")

if at_least("advanced"):
    st.markdown("## متقدم: الاستقرار")
    st.markdown("اختيار Lasso قد يتغير كثيرًا عبر عينات Bootstrap مع الخصائص المترابطة؛ **Stability selection** يحتفظ بالخصائص "
                "المختارة في أكثر من نسبة معينة من العينات الفرعية.")
if at_least("research"):
    researcher_note(["الاستدلال بعد الاختيار (Post-selection inference) يتطلب تصحيحًا؛ p-values العادية بعد الاختيار متفائلة.",
                     "في DML لا نختار الضوابط بالتنبؤ بـy وحده: Double selection (Belloni et al., 2014) يختار ما يتنبأ بـy أو بـD."])
mistakes(["الاختيار على كل البيانات ثم CV.", "الاعتماد على Filter وحده مع تفاعلات.", "تفسير الخاصية غير المختارة كعديمة الصلة."])
page_footer("feature_selection",
            takeaways=["Filter/Wrapper/Embedded ثلاث عائلات بمقايضات مختلفة.", "الاختيار الموجّه يحدث داخل CV.",
                       "الاختيار غير مستقر مع الخصائص المترابطة."])
