import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from components.callouts import definition, mistakes, note, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab, explain_code
from components.diagrams import mermaid
from core.page import page_footer, page_header, page_link
from core.state import at_least
from utils.datasets import xy

page_header("ml_vocabulary")

st.markdown("## ثلاث كلمات يخلطها الجميع")
comparison_table([
    {"المفهوم": "Algorithm · الخوارزمية", "ما هو؟": "إجراء تعلّم عام", "مثال": "CART: ابحث عن أفضل تقسيم ثم كرر",
     "في الكود": "الصنف DecisionTreeClassifier"},
    {"المفهوم": "Estimator · المقدِّر", "ما هو؟": "كائن يحمل الخوارزمية + إعداداتها", "مثال":
        "DecisionTreeClassifier(max_depth=3)", "في الكود": "كائن قبل fit()"},
    {"المفهوم": "Model · النموذج", "ما هو؟": "ناتج التعلّم على بيانات محددة", "مثال": "شجرة بعتبات معينة",
     "في الكود": "الكائن بعد fit() (خصائص تنتهي بـ _)"},
])
mermaid("""
flowchart LR
  A["Algorithm<br/>(CART)"] --> E["Estimator<br/>DecisionTreeClassifier(max_depth=3)"]
  H["Hyperparameters<br/>max_depth=3"] --> E
  D["Training data<br/>X, y"] --> F(("fit()"))
  E --> F
  F --> M["Fitted model<br/>tree_ , feature_importances_"]
  M --> P(("predict(X_new)"))
""")

st.markdown("## المعاملات مقابل المعاملات الفائقة")
definition("Parameter (معامل متعلَّم)", "قيمة **يتعلمها** النموذج من البيانات أثناء `fit()`؛ تظهر في scikit-learn "
           "كخصائص تنتهي بشرطة سفلية: `coef_`، `intercept_`، `tree_`، `cluster_centers_`، `support_vectors_`.")
definition("Hyperparameter (معامل فائق)", "قيمة **تختارها أنت قبل** التدريب وتتحكم في كيفية التعلّم أو في سعة النموذج: "
           "`C` و`max_depth` و`n_neighbors` و`learning_rate`. تُمرَّر إلى المُنشئ وتُقرأ بـ`get_params()`.")
comparison_table([
    {"النوع": "Model parameter", "من يحدده؟": "البيانات عبر fit", "أمثلة": "coef_, tree_ thresholds, centers",
     "هل نضبطه بـCV؟": "لا — يُتعلَّم"},
    {"النوع": "Hyperparameter", "من يحدده؟": "أنت قبل fit", "أمثلة": "C, max_depth, alpha, n_neighbors",
     "هل نضبطه بـCV؟": "نعم"},
    {"النوع": "Function argument", "من يحدده؟": "أنت عند الاستدعاء", "أمثلة": "train_test_split(test_size=0.2)",
     "هل نضبطه بـCV؟": "عادة لا"},
    {"النوع": "Solver option", "من يحدده؟": "أنت", "أمثلة": "solver='lbfgs', tol, max_iter",
     "هل نضبطه بـCV؟": "لا غالبًا (يؤثر في الحساب لا في الحل المثالي)"},
    {"النوع": "Runtime configuration", "من يحدده؟": "البيئة", "أمثلة": "n_jobs, verbose, device='cuda'",
     "هل نضبطه بـCV؟": "لا — لا يغير النتائج (إلا العشوائية)"},
])
note("`random_state` حالة خاصة: ليس معاملًا فائقًا يجب «ضبطه» لتحسين الدرجة؛ تثبيته لإعادة الإنتاج، وتغييره لقياس "
     "حساسية النتائج للعشوائية. اختيار البذرة التي تعطي أعلى درجة هو شكل من الغش (Seed hacking).",
     title=":material/casino: عن random_state")

st.markdown("## جرّب بنفسك: ما الذي يتغير بعد fit()؟")
X, y = xy("classification")


def _params():
    c1, c2 = st.columns(2)
    algo = c1.segmented_control("الخوارزمية", ["LogisticRegression", "DecisionTreeClassifier"],
                                default="DecisionTreeClassifier", key="voc_algo", required=True)
    hp = c2.slider("C" if algo == "LogisticRegression" else "max_depth", 1, 10, 3, key="voc_hp")
    return {"algo": algo, "hp": hp}


def _code(p):
    arg = f"C={p['hp']}" if p["algo"] == "LogisticRegression" else f"max_depth={p['hp']}, random_state=0"
    return (f"from sklearn.{'linear_model' if p['algo'] == 'LogisticRegression' else 'tree'} import {p['algo']}\n\n"
            f"est = {p['algo']}({arg})     # estimator + hyperparameters\n"
            "print(est.get_params())            # hyperparameters: known before fit\n"
            "est.fit(X, y)                      # learning happens here\n"
            "# fitted attributes end with '_' : these are the learned parameters")


def _run(algo, hp):
    est = LogisticRegression(C=hp, max_iter=2000) if algo == "LogisticRegression" else \
        DecisionTreeClassifier(max_depth=hp, random_state=0)
    before = {k: v for k, v in est.get_params().items() if v is not None}
    est.fit(X, y)
    learned = sorted(a for a in vars(est) if a.endswith("_") and not a.startswith("_"))
    rows = []
    for a in learned:
        v = getattr(est, a)
        shape = getattr(v, "shape", None)
        rows.append({"fitted attribute": a, "type": type(v).__name__, "shape": str(shape) if shape is not None else "-"})
    extra = ""
    if algo == "LogisticRegression":
        extra = f"**coef_ (أول 5):** `{np.round(est.coef_[0][:5], 3).tolist()}`"
    else:
        extra = f"**عدد الأوراق المتعلَّم:** {est.get_n_leaves()} · **العمق الفعلي:** {est.get_depth()}"
    return [f"**Hyperparameters (قبل fit):** `{before}`", pd.DataFrame(rows), extra]


code_lab("vocab", "Estimator → fit → learned parameters", _code, _run, _params,
         explanation="الخصائص المنتهية بـ `_` لم تكن موجودة قبل fit؛ هي المعاملات المتعلَّمة. أما get_params فيعيد "
                     "المعاملات الفائقة التي حددتها (أو قيمها الافتراضية).")

st.markdown("## واجهة scikit-learn الموحّدة")
explain_code("model.fit(X_train, y_train).predict(X_test)", [
    ("model", "Estimator: خوارزمية + معاملات فائقة"),
    (".fit(X_train, y_train)", "يتعلم المعاملات من بيانات التدريب ويعيد الكائن نفسه"),
    (".predict(X_test)", "يطبق النموذج المتعلَّم على بيانات جديدة"),
], output="مصفوفة تنبؤات بطول X_test", interpretation="لا تعلّم في predict: النموذج ثابت بعد fit.")
comparison_table([
    {"الدالة": "fit(X, y)", "من يملكها": "كل Estimator", "ماذا تفعل": "تتعلم من البيانات"},
    {"الدالة": "predict(X)", "من يملكها": "Predictors", "ماذا تفعل": "تنبؤات (فئات أو قيم)"},
    {"الدالة": "predict_proba(X)", "من يملكها": "مصنفات احتمالية", "ماذا تفعل": "احتمالات لكل فئة"},
    {"الدالة": "decision_function(X)", "من يملكها": "مصنفات هامشية (SVM, Logistic)", "ماذا تفعل": "درجات غير محدودة"},
    {"الدالة": "transform(X)", "من يملكها": "Transformers (Scaler, PCA)", "ماذا تفعل": "تحويل البيانات"},
    {"الدالة": "score(X, y)", "من يملكها": "Predictors", "ماذا تفعل": "Accuracy للتصنيف، R² للانحدار"},
    {"الدالة": "get_params / set_params", "من يملكها": "كل Estimator", "ماذا تفعل": "قراءة/تعديل المعاملات الفائقة"},
])
why("اجعل كل معالجة تتعلم من البيانات (Scaler, Imputer, Encoder) Estimator داخل Pipeline.",
    "لأن fit يجب أن يرى بيانات التدريب فقط؛ الواجهة الموحّدة تجعل CV يعيد fit لكل خطوة داخل كل طية تلقائيًا.")
page_link("pipelines", "انتقل إلى: Pipelines", ":material/account_tree:")

if at_least("advanced"):
    st.markdown("## متقدم: المعاملات الفائقة تحدد فضاء الفرضيات")
    st.markdown("يمكن النظر إلى كل قيمة للمعاملات الفائقة $\\lambda$ كفضاء فرضيات $\\mathcal F_\\lambda$؛ يحدد `fit` "
                "العضو الأفضل $\\hat f_\\lambda \\in \\mathcal F_\\lambda$ على التدريب، ويختار الضبط $\\hat\\lambda$ "
                "على التحقق. لذا هناك **مستويان** من التعلّم، ويحتاج كلاهما بيانات منفصلة لتقدير أداء غير متحيز.")
    st.latex(r"\hat\lambda = \arg\min_{\lambda} \widehat{R}_{\text{val}}\big(\hat f_\lambda\big), \qquad "
             r"\hat f_\lambda = \arg\min_{f\in\mathcal F_\lambda} \widehat R_{\text{train}}(f)")
if at_least("research"):
    researcher_note(["عند النشر: أبلغ عن كل المعاملات الفائقة (بما فيها الافتراضية) ونسخة المكتبة؛ الافتراضيات تتغير "
                     "بين الإصدارات (مثال: `penalty` في LogisticRegression مهجور منذ 1.8).",
                     "ميّز في الورقة بين «ما ضُبط» و«ما ثُبّت» وفضاء البحث والميزانية."])
    page_link("hyperparameter_encyclopedia", "موسوعة المعاملات الفائقة (قيم افتراضية متحقَّق منها)", ":material/menu_book:")

mistakes(["اعتبار n_jobs أو verbose معاملات تؤثر في الدقة.", "ضبط random_state لتعظيم الدرجة.",
          "تعديل النموذج بعد fit ثم نسيان أن المعاملات المتعلَّمة لم تتغير."])
page_footer("ml_vocabulary",
            takeaways=["Algorithm إجراء، Estimator كائن بإعداداته، Model ناتج fit.",
                       "المعاملات تُتعلَّم (_)، والمعاملات الفائقة تُختار قبل التدريب وتُضبط بالتحقق.",
                       "واجهة fit/predict/transform الموحّدة هي ما يجعل Pipelines وCV آمنة."])
