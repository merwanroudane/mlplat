import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset, toy_2d
from utils.plotting import plot

page_header("naive_bayes")
algorithm_profile("gaussian_nb")

formula(r"P(y=k\mid x) \propto P(y=k)\prod_{j=1}^p P(x_j\mid y=k)",
        title="Naive Bayes classifier",
        symbols={"P(y=k)": "الأولوية (نسبة الفئة)", r"P(x_j\mid y=k)": "توزيع الخاصية j داخل الفئة k",
                 r"\prod": "الضرب ممكن بافتراض الاستقلال الشرطي"},
        intuition="«ساذج» لأنه يفترض أن الخصائص مستقلة بمعلومية الفئة. الافتراض خاطئ غالبًا، لكن الترتيب بين الفئات يبقى "
                  "صحيحًا كثيرًا — فالتصنيف جيد والاحتمالات مفرطة الثقة.",
        example="رسالة فيها «مجاني» و«فوز»: P(spam|x) ∝ P(spam)·P(«مجاني»|spam)·P(«فوز»|spam).")
formula(r"\hat y = \arg\max_k\Big[\log P(y=k) + \sum_j \log P(x_j\mid y=k)\Big]", title="In log space (numerically stable)")

comparison_table([
    {"المتغير": "GaussianNB", "P(x_j | y)": "طبيعي N(μ_jk, σ²_jk)", "البيانات": "عددية مستمرة", "معامل رئيسي": "var_smoothing"},
    {"المتغير": "MultinomialNB", "P(x_j | y)": "متعدد الحدود على العدّ", "البيانات": "عدّ الكلمات / TF-IDF", "معامل رئيسي": "alpha (تنعيم)"},
    {"المتغير": "ComplementNB", "P(x_j | y)": "يُقدَّر من الفئات المكمّلة", "البيانات": "نصوص غير متوازنة", "معامل رئيسي": "alpha, norm"},
    {"المتغير": "BernoulliNB", "P(x_j | y)": "Bernoulli (وجود/غياب)", "البيانات": "خصائص ثنائية", "معامل رئيسي": "alpha, binarize"},
    {"المتغير": "CategoricalNB", "P(x_j | y)": "فئوي", "البيانات": "فئات مرمَّزة بأعداد", "معامل رئيسي": "alpha, min_categories"},
])

st.markdown("## GaussianNB: ما الذي يتعلمه؟")
X, y = toy_2d("blobs", n=300, noise=0.3)
nb = GaussianNB().fit(X, y)
st.dataframe(pd.DataFrame({"class": nb.classes_, "prior": nb.class_prior_, "μ(x₁)": nb.theta_[:, 0], "μ(x₂)": nb.theta_[:, 1],
                           "σ²(x₁)": nb.var_[:, 0], "σ²(x₂)": nb.var_[:, 1]}).round(3), hide_index=True, width="stretch")
boundary_chart(nb, X, y, title="GaussianNB: axis-aligned Gaussians ⇒ quadratic boundaries", show_proba=False)
feature = st.segmented_control("اعرض التوزيعات الشرطية لـ", ["x₁", "x₂"], default="x₁", key="nb_feat", required=True)
j = 0 if feature == "x₁" else 1
grid = np.linspace(X[:, j].min() - 1, X[:, j].max() + 1, 300)
fig = go.Figure()
for k, color in zip(nb.classes_, [PALETTE["sky"], PALETTE["coral"], PALETTE["purple"]]):
    fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, nb.theta_[k, j], np.sqrt(nb.var_[k, j])), name=f"P({feature} | y={k})",
                             line=dict(color=color, width=2.6), fill="tozeroy"))
fig.update_layout(height=300, title="Class-conditional densities (the 'likelihoods' NB multiplies)")
plot(fig)

st.markdown("## النصوص: Multinomial وComplement وBernoulli")
df = load_dataset("text")


@st.cache_data(show_spinner=False)
def _text_scores():
    rows = []
    for name, vec, clf in (("MultinomialNB + counts", CountVectorizer(ngram_range=(1, 2)), MultinomialNB()),
                           ("ComplementNB + TF-IDF", TfidfVectorizer(ngram_range=(1, 2)), ComplementNB()),
                           ("BernoulliNB + binary", CountVectorizer(binary=True, ngram_range=(1, 2)), BernoulliNB()),
                           ("GaussianNB on dense counts (wrong tool)", CountVectorizer(), None)):
        if clf is None:
            from sklearn.preprocessing import FunctionTransformer
            pipe = make_pipeline(vec, FunctionTransformer(lambda m: m.toarray(), accept_sparse=True), GaussianNB())
        else:
            pipe = make_pipeline(vec, clf)
        s = cross_val_score(pipe, df["text"], df["label"], cv=5, scoring="accuracy")
        rows.append({"pipeline": name, "CV accuracy": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


st.dataframe(_text_scores().round(3), hide_index=True, width="stretch")
pipe = make_pipeline(CountVectorizer(), MultinomialNB()).fit(df["text"], df["label"])
vocab = np.array(pipe[0].get_feature_names_out())
logp = pipe[1].feature_log_prob_
diff = logp[1] - logp[0]
top_pos, top_neg = vocab[np.argsort(-diff)[:8]], vocab[np.argsort(diff)[:8]]
c1, c2 = st.columns(2)
c1.markdown("**أقوى كلمات الفئة الإيجابية:** " + ", ".join(f"`{w}`" for w in top_pos))
c2.markdown("**أقوى كلمات الفئة السلبية:** " + ", ".join(f"`{w}`" for w in top_neg))
why("MultinomialNB خط أساس ممتاز للنصوص.", "سريع جدًا، يعمل مع بيانات قليلة، ويصعب التفوق عليه كثيرًا في النصوص القصيرة؛ "
    "لكنه لا يفهم النفي («not great») إلا عبر N-grams.")

hyperparameter_table("GaussianNB")
hyperparameter_table("MultinomialNB")

if at_least("advanced"):
    st.markdown("## متقدم: لماذا تنجح «السذاجة»؟")
    intuition("التصنيف يحتاج فقط أن تكون الفئة الصحيحة صاحبة أعلى درجة؛ الأخطاء في الاحتمالات قد لا تغير الترتيب. "
              "لكن الخصائص المكررة تُحتسب مرتين فتتضخم الثقة. Domingos & Pazzani (1997) حللا متى يكون NB أمثليًا رغم كسر الافتراض.")
    st.markdown("**التعقيد:** تدريب O(np) بتمريرة واحدة، و`partial_fit` للتعلّم المتزايد.")
if at_least("research"):
    researcher_note(["NB نموذج توليدي؛ Logistic تمييزي مقابل له (Ng & Jordan, 2002): NB يتقارب أسرع مع بيانات قليلة، "
                     "Logistic أفضل تقاربيًا.",
                     "Laplace smoothing (alpha=1) = Prior ديريكليه منتظم؛ اضبطه بـCV في النصوص."])
template_checklist({3: "صيغة Bayes", 5: "حدود القرار", 6: "الإمكان اللوغاريتمي", 7: "عدّ وتقدير متوسطات/تباينات",
                    8: "جدول theta_ وvar_", 22: "GaussianNB/MultinomialNB", 23: "مختبر النصوص والتوزيعات"})
mistakes(["استخدام GaussianNB على عدّ كلمات متفرق.", "الثقة باحتمالات NB دون معايرة.", "alpha = 0 مع كلمات جديدة."])
page_footer("naive_bayes",
            takeaways=["Bayes + استقلال شرطي = مصنّف سريع جدًا.", "اختر المتغير حسب نوع الخصائص.",
                       "تصنيف جيد غالبًا، احتمالات مفرطة الثقة."])
