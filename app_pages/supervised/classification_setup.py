import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import xy

page_header("classification_setup")

st.markdown("## أربعة أشكال للتصنيف")
comparison_table([
    {"الشكل": "Binary", "y": "{0, 1}", "مثال": "احتيال/سليم", "المخرج": "p = P(y=1|x)"},
    {"الشكل": "Multiclass", "y": "{1..K} (فئة واحدة)", "مثال": "نوع الوثيقة", "المخرج": "متجه احتمالات مجموعها 1"},
    {"الشكل": "Multilabel", "y": "مجموعة جزئية من {1..K}", "مثال": "وسوم مقال", "المخرج": "احتمال مستقل لكل وسم"},
    {"الشكل": "Ordinal", "y": "فئات مرتبة", "مثال": "خفيف < متوسط < شديد", "المخرج": "P(y > k) لكل عتبة"},
])

st.markdown("## الدرجة، الاحتمال، القرار")
comparison_table([
    {"المخرج": "decision_function(x)", "المدى": "(−∞, ∞)", "المعنى": "بُعد موقّع عن الحد", "مثال": "SVM، Logistic (log-odds)"},
    {"المخرج": "predict_proba(x)", "المدى": "[0, 1]", "المعنى": "احتمال (قد يحتاج معايرة)", "مثال": "Logistic، RF، NB"},
    {"المخرج": "predict(x)", "المدى": "فئة", "المعنى": "قرار = درجة مقارنة بعتبة", "مثال": "العتبة 0.5 افتراضيًا للاحتمال"},
])
why("افصل بين النموذج (الدرجات/الاحتمالات) والقرار (العتبة).",
    "العتبة تعتمد على تكلفة الأخطاء وتتغير دون إعادة تدريب؛ الاحتمالات يمكن معايرتها بشكل منفصل.")

st.markdown("## Multiclass: Softmax مقابل OvR مقابل OvO")
formula(r"P(y=k\mid x) = \frac{e^{w_k^\top x + b_k}}{\sum_{j=1}^K e^{w_j^\top x + b_j}}", title="Softmax (multinomial)",
        intuition="درجة لكل فئة ثم تطبيع أُسّي. LogisticRegression في scikit-learn يستخدم Multinomial لكل الحلول عدا liblinear.")
comparison_table([
    {"الاستراتيجية": "Multinomial (softmax)", "عدد النماذج": "1 (K متجه أوزان)", "ملاحظات": "احتمالات متسقة؛ افتراضي في LogisticRegression"},
    {"الاستراتيجية": "One-vs-Rest (OvR)", "عدد النماذج": "K", "ملاحظات": "كل فئة مقابل البقية؛ فئات غير متوازنة داخليًا"},
    {"الاستراتيجية": "One-vs-One (OvO)", "عدد النماذج": "K(K−1)/2", "ملاحظات": "تصويت؛ SVC يستخدمه داخليًا للتدريب"},
])
X, y = xy("multiclass")
Xa = X.to_numpy()
strategy = st.segmented_control("الاستراتيجية", ["Multinomial", "One-vs-Rest", "One-vs-One"], default="Multinomial",
                                key="cs_strat", required=True)
model ={"Multinomial": LogisticRegression(max_iter=2000),
         "One-vs-Rest": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
         "One-vs-One": OneVsOneClassifier(LogisticRegression(max_iter=2000))}[strategy].fit(Xa, y)
n_models = {"Multinomial": 1, "One-vs-Rest": 3, "One-vs-One": 3}[strategy]
boundary_chart(model, Xa, y.to_numpy(), title=f"{strategy}: {n_models} underlying model(s), train accuracy {model.score(Xa, y):.3f}",
               show_proba=False)
intuition("مع 3 فئات يتساوى عدد نماذج OvR وOvO (3)، لكن مع 10 فئات: OvR = 10 وOvO = 45.")

st.markdown("## Multilabel في كود")
st.code("""from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
Y = np.column_stack([is_sports, is_politics, is_economy])     # n × 3 binary matrix
MultiOutputClassifier(LogisticRegression()).fit(X, Y)         # independent classifiers
ClassifierChain(LogisticRegression(), order="random", random_state=0).fit(X, Y)  # models label correlation""",
        language="python")

if at_least("advanced"):
    st.markdown("## متقدم: التصنيف الترتيبي بمصنفات ثنائية")
    st.markdown("لـK فئة مرتبة ندرّب K−1 مصنفًا لـP(y > k)، ثم $P(y=k) = P(y>k-1) - P(y>k)$. البديل الإحصائي: "
                "Proportional odds (statsmodels OrderedModel).")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(600, 2))
    lat = z @ np.array([1.5, -1.0]) + rng.logistic(size=600)
    yo = np.digitize(lat, [-1.5, 0, 1.5])
    probs = np.column_stack([LogisticRegression().fit(z, (yo > k).astype(int)).predict_proba(z)[:, 1] for k in range(3)])
    p_k = np.column_stack([1 - probs[:, 0], probs[:, 0] - probs[:, 1], probs[:, 1] - probs[:, 2], probs[:, 2]])
    st.dataframe(pd.DataFrame(p_k[:5], columns=["P(y=0)", "P(y=1)", "P(y=2)", "P(y=3)"]).round(3), hide_index=True)
if at_least("research"):
    researcher_note(["Macro-averaging يعامل الفئات بالتساوي (يكشف ضعف الفئات النادرة)؛ Micro يرجّح حسب التكرار.",
                     "في Multilabel، Hamming loss وSubset accuracy يقيسان أشياء مختلفة جدًا؛ اختر حسب الاستخدام."])
mistakes(["معاملة predict_proba لـSVM/Trees كاحتمالات معايرة.", "استخدام Accuracy للتصنيف متعدد الفئات غير المتوازن.",
          "معاملة مسألة ترتيبية كاسمية."])
page_footer("classification_setup",
            takeaways=["Binary/Multiclass/Multilabel/Ordinal أربع مسائل مختلفة.",
                       "الدرجة ≠ الاحتمال ≠ القرار.", "Softmax وOvR وOvO استراتيجيات للفئات المتعددة."])
