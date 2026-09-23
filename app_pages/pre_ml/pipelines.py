import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.code_lab import explain_code
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least, log_experiment
from utils.datasets import xy

page_header("pipelines")

st.markdown("## fit وtransform وfit_transform")
comparison_table([
    {"الدالة": "fit(X)", "ماذا تتعلم؟": "إحصاءات من البيانات (متوسط، انحراف، فئات، وسيط)", "أين؟": "التدريب فقط"},
    {"الدالة": "transform(X)", "ماذا تتعلم؟": "لا شيء — تطبق ما تعلّمته fit", "أين؟": "التدريب والتحقق والاختبار"},
    {"الدالة": "fit_transform(X)", "ماذا تتعلم؟": "fit ثم transform على البيانات نفسها", "أين؟": "التدريب فقط"},
])
intuition("StandardScaler «يحفظ» متوسط التدريب وانحرافه في `mean_` و`scale_`. إن حسبتهما على كل البيانات، فقد رأى "
          "النموذج شيئًا من الاختبار.")

st.markdown("## Pipeline + ColumnTransformer لبيانات مختلطة")
mermaid("""
flowchart LR
  X[X raw] --> CT{ColumnTransformer}
  CT -->|numeric cols| N1[SimpleImputer median] --> N2[StandardScaler]
  CT -->|categorical cols| C1[SimpleImputer most_frequent] --> C2[OneHotEncoder]
  N2 --> M[concat]
  C2 --> M
  M --> CLF[LogisticRegression]
""")
code = """from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "income", "loan_amount", "credit_score", "n_accounts"]
cat_cols = ["employment", "region", "city"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat_cols),
])
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=2000))])"""
st.code(code, language="python")
explain_code('param_grid = {"clf__C": [0.01, 0.1, 1, 10], "prep__cat__onehot__min_frequency": [5, 20, 50]}', [
    ("clf__C", "المعامل C في الخطوة المسماة clf"),
    ("prep__cat__onehot__min_frequency", "مسار متداخل: prep ← cat ← onehot ← min_frequency"),
    ("__ (شرطتان)", "فاصل المستويات في أسماء المعاملات المتداخلة"),
], output="قاموس يمرَّر إلى GridSearchCV", interpretation="يمكن ضبط المعالجة والنموذج معًا داخل CV دون أي تسرب.")

st.markdown("## مختبر Pipeline: شغّل، واضبط، وانظر داخل الـPipeline")
X, y = xy("mixed")
num_cols = ["age", "income", "loan_amount", "credit_score", "n_accounts"]
cat_cols = ["employment", "region", "city"]
c1, c2, c3 = st.columns(3)
num_imp = c1.selectbox("تعويض العددي", ["median", "mean"], key="pl_imp")
min_freq = c2.select_slider("min_frequency للمدن", [1, 5, 20, 50, 100], value=20, key="pl_mf")
C = c3.select_slider("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key="pl_C")


def _pipe(num_imp: str, min_freq: int, C: float) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy=num_imp)), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                                                   sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", pre), ("clf", LogisticRegression(C=C, max_iter=3000))])


@st.cache_data(show_spinner="يدرّب Pipeline داخل 5 طيات…", max_entries=32)
def _evaluate(num_imp: str, min_freq: int, C: float):
    pipe = _pipe(num_imp, min_freq, C)
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    pipe.fit(X, y)
    names = pipe[:-1].get_feature_names_out()
    return scores, list(names), pipe.named_steps["prep"].named_transformers_["num"]["scale"].mean_


scores, names, means = _evaluate(num_imp, min_freq, C)
with st.container(horizontal=True):
    st.metric("ROC-AUC (5-fold)", f"{scores.mean():.3f}", f"± {scores.std():.3f}", delta_color="off", border=True)
    st.metric("عدد الخصائص بعد المعالجة", len(names), border=True)
    st.metric("أعمدة المدن", sum(n.startswith("city_") for n in names), border=True)
with st.expander("أسماء الخصائص الناتجة (get_feature_names_out) ومتوسطات Scaler المتعلَّمة"):
    st.write(", ".join(names))
    st.dataframe(pd.DataFrame({"column": num_cols, "learned mean_": np.round(means, 2)}), hide_index=True)
if st.button("سجّل في سجل التجارب", key="pl_log", icon=":material/history:", type="tertiary"):
    log_experiment("Pipeline Lab", "LogisticRegression pipeline", {"num_impute": num_imp, "min_frequency": min_freq, "C": C},
                   {"roc_auc_mean": scores.mean(), "roc_auc_sd": scores.std()}, seed=0, dataset="mixed",
                   split="StratifiedKFold(5, shuffle, rs=0)")
    st.toast("سُجّلت.", icon=":material/check:")

st.markdown("### ضبط المعالجة والنموذج معًا")
if st.button("شغّل GridSearchCV على (min_frequency × C)", key="pl_grid", icon=":material/grid_on:"):
    st.session_state["pl_grid_on"] = True
if st.session_state.get("pl_grid_on"):
    @st.cache_data(show_spinner="12 تركيبة × 3 طيات…")
    def _grid():
        gs = GridSearchCV(_pipe("median", 20, 1.0), {"prep__cat__onehot__min_frequency": [5, 20, 100],
                                                     "clf__C": [0.01, 0.1, 1.0, 10.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc")
        gs.fit(X, y)
        res = pd.DataFrame(gs.cv_results_)[["param_prep__cat__onehot__min_frequency", "param_clf__C", "mean_test_score",
                                            "std_test_score", "rank_test_score"]]
        return res.sort_values("rank_test_score"), gs.best_params_
    res, best = _grid()
    st.dataframe(res.round(4), hide_index=True, width="stretch")
    st.success(f"أفضل تركيبة: `{best}` — لكن تذكّر: درجة best_score_ متفائلة لأنها استُخدمت للاختيار (انظر Nested CV).")

why("Pipeline ليس رفاهية تنظيمية؛ هو الضمان الهيكلي ضد التسرب.",
    "cross_val_score(pipeline) يعيد fit لكل الخطوات داخل كل طية، ويحفظ النموذج النهائي المعالجةَ والنموذجَ في كائن واحد "
    "للنشر، فلا تختلف المعالجة بين التدريب والإنتاج (Training-serving skew).")

if at_least("advanced"):
    st.markdown("## متقدم: أدوات مفيدة")
    comparison_table([
        {"الأداة": "make_pipeline / make_column_transformer", "الفائدة": "أسماء تلقائية للخطوات"},
        {"الأداة": "make_column_selector(dtype_include=...)", "الفائدة": "اختيار الأعمدة حسب النوع وقت fit"},
        {"الأداة": "FunctionTransformer", "الفائدة": "تحويل بلا حالة (log1p) داخل الـPipeline"},
        {"الأداة": "TransformedTargetRegressor", "الفائدة": "تحويل y (مثل log) وعكسه تلقائيًا"},
        {"الأداة": "set_output(transform='pandas')", "الفائدة": "مخرجات DataFrame بأسماء أعمدة"},
        {"الأداة": "Pipeline(memory=...)", "الفائدة": "تخزين نتائج المعالجة المكلفة أثناء Grid search"},
    ])
if at_least("research"):
    researcher_note(["احفظ Pipeline كاملًا مع إصدارات المكتبات؛ Pickle غير آمن مع ملفات غير موثوقة (انظر وحدة الإنتاج).",
                     "Resampling (SMOTE) يحتاج imblearn.pipeline.Pipeline لأنه يعمل في fit فقط لا في predict."])
    st.markdown(cite("pedregosa2011", "kaufman2012"))
mistakes(["fit_transform على بيانات الاختبار.", "معالجة يدوية خارج الـPipeline ثم CV.",
          "نسيان handle_unknown فيفشل النموذج مع فئة جديدة في الإنتاج."])
page_footer("pipelines",
            takeaways=["fit يتعلم، transform يطبق.", "ColumnTransformer يعالج كل نوع أعمدة بطريقته.",
                       "step__param يسمح بضبط المعالجة والنموذج معًا دون تسرب."])
