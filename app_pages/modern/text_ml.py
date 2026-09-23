import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import load_dataset
from utils.plotting import bars, heatmap, plot

page_header("text_ml")

df = load_dataset("text")
st.dataframe(df.sample(6, random_state=1), hide_index=True, width="stretch")

st.markdown("## من نص إلى مصفوفة")
formula(r"\text{tf-idf}(w, d) = \text{tf}(w, d)\cdot\Big(\log\frac{1+n}{1+\text{df}(w)} + 1\Big)",
        title="TF-IDF (scikit-learn default: smooth_idf=True, then L2 row normalisation)",
        symbols={"tf(w, d)": "تكرار الكلمة في الوثيقة", "df(w)": "عدد الوثائق التي تحتوي الكلمة", "n": "عدد الوثائق"},
        intuition="كلمة تظهر في كل الوثائق («the») وزنها منخفض؛ كلمة نادرة ومميزة وزنها مرتفع.")
docs = ["the battery is great", "the screen is great and fast", "the battery stopped working"]
ngram = st.segmented_control("n-grams", ["(1,1)", "(1,2)"], default="(1,1)", key="tx_ng", required=True)
rng_ = (1, 1) if ngram == "(1,1)" else (1, 2)
bow = CountVectorizer(ngram_range=rng_).fit(docs)
tfidf = TfidfVectorizer(ngram_range=rng_).fit(docs)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Bag of Words (counts)**")
    st.dataframe(pd.DataFrame(bow.transform(docs).toarray(), columns=bow.get_feature_names_out(), index=[f"doc{i}" for i in range(3)]),
                 width="stretch")
with c2:
    st.markdown("**TF-IDF**")
    st.dataframe(pd.DataFrame(tfidf.transform(docs).toarray(), columns=tfidf.get_feature_names_out(),
                              index=[f"doc{i}" for i in range(3)]).round(2), width="stretch")
sim = cosine_similarity(tfidf.transform(docs))
plot(heatmap(sim, [f"doc{i}" for i in range(3)], [f"doc{i}" for i in range(3)], title="Cosine similarity between TF-IDF vectors",
             colorscale=[[0, "#FCFCFF"], [1, "#7048E8"]]), height=300)
intuition("N-grams تلتقط تراكيب مثل «not great» و«stopped working» التي تضيع في كلمات مفردة.")

st.markdown("## Text Classifier Lab")


@st.cache_data(show_spinner="يقارن المصنفات الخطية…")
def _compare():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, pipe in (("MultinomialNB + counts (1,2)", make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())),
                       ("Logistic + TF-IDF (1,1)", make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))),
                       ("Logistic + TF-IDF (1,2)", make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))),
                       ("Linear SVM + TF-IDF (1,2)", make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC()))):
        s = cross_val_score(pipe, df["text"], df["label"], cv=cv)
        rows.append({"pipeline": name, "accuracy": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


res = _compare()
st.dataframe(res.round(3), hide_index=True, width="stretch")
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000)).fit(df["text"], df["label"])
coef = pipe[-1].coef_[0]
vocab = np.array(pipe[0].get_feature_names_out())
o = np.r_[np.argsort(coef)[:8], np.argsort(coef)[-8:]]
plot(bars(vocab[o], coef[o], title="Most negative / positive n-gram weights (logistic)", horizontal=True,
          color=[PALETTE["coral"] if c < 0 else PALETTE["teal"] for c in coef[o]]), height=420)
text = st.text_input("جرّب جملة (بالإنجليزية)", "the screen is not great and very slow", key="tx_try")
if text:
    p = pipe.predict_proba([text])[0, 1]
    st.metric("P(positive)", f"{p:.2f}")
if st.button("سجّل", key="tx_log", icon=":material/history:", type="tertiary"):
    log_experiment("Text Classifier Lab", "TF-IDF(1,2) + LogisticRegression", {"ngram_range": "(1,2)"},
                   {"cv_accuracy": float(res.iloc[2]["accuracy"])}, seed=0, dataset="text", split="StratifiedKFold(5)")
    st.toast("سُجّل.", icon=":material/check:")
why("ضع Vectorizer داخل Pipeline.", "المفردات وأوزان IDF تُتعلَّم من البيانات؛ حسابها على كل النصوص قبل CV تسرّب.")

st.markdown("## التضمينات (Embeddings) كمفهوم")
comparison_table([
    {"التمثيل": "Bag of Words / TF-IDF", "الأبعاد": "حجم المفردات (آلاف، متفرق)", "المعنى": "لا يعرف أن great ≈ excellent"},
    {"التمثيل": "Word embeddings", "الأبعاد": "مئات (كثيف)", "المعنى": "كلمات متقاربة المعنى متقاربة الاتجاه"},
    {"التمثيل": "Sentence embeddings", "الأبعاد": "مئات", "المعنى": "تمثيل الجملة كاملة؛ تشابه جيب التمام للبحث الدلالي"},
])
st.caption("التضمينات تأتي من نماذج مدرّبة مسبقًا (Transfer learning)؛ تُغطى في منصة التعلّم العميق. هنا نستخدم TF-IDF ليبقى التطبيق "
           "خفيفًا وبلا تنزيل نماذج.")

if at_least("research"):
    researcher_note(["خطوط أساس NB/Logistic/SVM على TF-IDF تبقى قوية جدًا في التصنيف النصي القصير.",
                     "للعربية: التجزئة والتطبيع (الهمزات، التشكيل) والجذور تحديات إضافية قبل الـVectorizer."])
mistakes(["Vectorizer خارج Pipeline.", "حذف كلمات النفي كـstopwords.", "Accuracy وحدها مع فئات غير متوازنة."])
page_footer("text_ml",
            takeaways=["BoW/TF-IDF + نموذج خطي خط أساس قوي.", "N-grams تلتقط النفي والتراكيب.", "التضمينات تلتقط المعنى (جسر للتعلم العميق)."])
