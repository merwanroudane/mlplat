import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from components.callouts import intuition, mistakes, why
from components.model_compare import compare_models, render_comparison
from core.page import page_footer, page_header
from core.state import log_experiment
from utils.datasets import DATASET_INFO, xy

page_header("model_comparison")

CANDIDATES = {
    "Dummy (prior)": lambda: DummyClassifier(strategy="prior"),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Gaussian NB": lambda: GaussianNB(),
    "kNN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC()),
    "Random Forest": lambda: RandomForestClassifier(200, random_state=0, n_jobs=1),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0),
}
c1, c2, c3 = st.columns(3)
ds = c1.selectbox("البيانات", ["classification", "imbalanced"], format_func=lambda n: DATASET_INFO[n].title, key="mc_ds")
metric = c2.selectbox("المقياس", ["roc_auc", "average_precision", "balanced_accuracy", "neg_log_loss"], key="mc_metric")
k = c3.slider("طيات CV", 3, 10, 5, key="mc_k")
chosen = st.pills("النماذج", list(CANDIDATES), selection_mode="multi", default=["Dummy (prior)", "Logistic Regression", "Random Forest",
                                                                               "HistGradientBoosting"], key="mc_models")


@st.cache_data(show_spinner="يقيّم كل النماذج على الطيات نفسها…", max_entries=16)
def _run(ds, metric, k, chosen):
    X, y = xy(ds)
    if metric == "neg_log_loss":
        models = {m: CANDIDATES[m]() for m in chosen if m != "SVM (RBF)"}
    else:
        models = {m: CANDIDATES[m]() for m in chosen}
    return compare_models(models, X, y, scoring=metric, cv=k)


if st.button("قارن", key="mc_go", type="primary", icon=":material/leaderboard:") and chosen:
    st.session_state["mc_args"] = (ds, metric, k, tuple(chosen))
if st.session_state.get("mc_args"):
    scores = _run(*st.session_state["mc_args"])
    render_comparison(scores, st.session_state["mc_args"][1])
    if "SVM (RBF)" in st.session_state["mc_args"][3] and st.session_state["mc_args"][1] == "neg_log_loss":
        st.caption("SVC استُبعد من log loss لأنه لا يعطي احتمالات (انظر وحدة SVM والمعايرة).")
    if st.button("سجّل النتائج", key="mcmp_log", icon=":material/history:", type="tertiary"):
        summ = scores.groupby("model")["validation"].mean()
        for m, v in summ.items():
            log_experiment("Model Comparison Lab", m, {"cv": st.session_state["mc_args"][2]}, {st.session_state["mc_args"][1]: v},
                           seed=0, dataset=st.session_state["mc_args"][0], split=f"Stratified {st.session_state['mc_args'][2]}-fold shared")
        st.toast("سُجّلت.", icon=":material/check:")
why("قارن على الطيات **نفسها** وأبلغ عن التشتت والفروق المزدوجة.", "الطيات المشتركة تزيل مصدر تباين بين النماذج؛ الفرق المزدوج "
    "أدق من مقارنة متوسطين مستقلين.")
intuition("إذا كان الفرق بين الأول والثاني أصغر من تقلبه عبر الطيات، فاختر الأبسط/الأسرع/الأسهل تفسيرًا.")
mistakes(["إعلان فائز من فرق ضمن الضجيج.", "مقارنة نماذج مضبوطة بأخرى غير مضبوطة.", "نسيان Baseline."])
page_footer("model_comparison", takeaways=["نفس الطيات، نفس المقياس، تشتت مُبلغ عنه.", "الفروق المزدوجة أدق.",
                                           "لا فائز مطلق: اختر بالأدلة والتكلفة والتفسير."])
