import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.semi_supervised import LabelSpreading

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from config import RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d, xy
from utils.plotting import lines, plot, scatter_classes

page_header("modern_topics")

comparison_table([
    {"الموضوع": "Semi-supervised", "المشكلة": "تسميات قليلة وبيانات كثيرة", "أدوات scikit-learn": "SelfTrainingClassifier، LabelSpreading/LabelPropagation"},
    {"الموضوع": "Active learning", "المشكلة": "التسمية مكلفة: أي الحالات نسمّي؟", "أدوات scikit-learn": "حلقة يدوية + Uncertainty sampling"},
    {"الموضوع": "Online / incremental", "المشكلة": "بيانات متدفقة أو لا تتسع للذاكرة", "أدوات scikit-learn": "partial_fit (SGD، NB، MiniBatchKMeans، IncrementalPCA)"},
    {"الموضوع": "Concept drift", "المشكلة": "العلاقة P(y|X) تتغير", "أدوات scikit-learn": "مراقبة + إعادة تدريب (وحدة Drift)"},
    {"الموضوع": "Transfer learning (مفهوم)", "المشكلة": "مهمة جديدة ببيانات قليلة", "أدوات scikit-learn": "تمثيلات مُدرّبة مسبقًا كخصائص"},
    {"الموضوع": "Self-supervised (مفهوم)", "المشكلة": "تعلّم تمثيلات دون تسميات", "أدوات scikit-learn": "خارج نطاق المنصة (Deep learning)"},
])

st.markdown("## Semi-supervised: Label Spreading")
X, y = toy_2d("moons", n=300, noise=0.12)
frac = st.select_slider("نسبة البيانات المُسمّاة", [0.01, 0.02, 0.05, 0.1, 0.3], value=0.02, key="mt_frac")
rng = np.random.default_rng(1)
labeled = np.zeros(len(y), bool)
for c in (0, 1):
    idx = np.where(y == c)[0]
    labeled[rng.choice(idx, max(1, int(frac * len(idx))), replace=False)] = True
y_semi = np.where(labeled, y, -1)
ls = LabelSpreading(kernel="knn", n_neighbors=7).fit(X, y_semi)
sup = LogisticRegression().fit(X[labeled], y[labeled])
c1, c2 = st.columns(2)
with c1:
    fig = scatter_classes(X[~labeled], ls.transduction_[~labeled])
    fig.add_trace(go.Scatter(x=X[labeled, 0], y=X[labeled, 1], mode="markers", name="labelled",
                             marker=dict(symbol="star", size=16, color="#212529")))
    fig.update_layout(title=f"LabelSpreading: accuracy {np.mean(ls.transduction_ == y):.1%} using {labeled.sum()} labels", height=360)
    plot(fig)
with c2:
    plot(scatter_classes(X, sup.predict(X)).update_layout(title=f"Supervised on labels only: accuracy {sup.score(X, y):.1%}", height=360))
intuition("الانتشار عبر رسم الجيران يستغل «افتراض العنقود/المنحنى»: النقاط المتصلة بكثافة تشترك في الفئة. إن كُسر هذا الافتراض فقد يضر "
          "التعلّم شبه الموجّه.")
st.code("""SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, y_semi)   # -1 = unlabelled
LabelSpreading(kernel="knn", n_neighbors=7).fit(X, y_semi).transduction_""", language="python")

st.markdown("## Active Learning Lab: Uncertainty sampling")
Xc, yc = xy("classification")
Xc, yc = Xc.to_numpy(), yc.to_numpy()
pool_idx = np.arange(700)
test_idx = np.arange(700, 1000)


@st.cache_data(show_spinner="يحاكي حلقة التسمية…")
def _active(rounds: int = 20, batch: int = 10):
    curves = {}
    for strategy in ("uncertainty", "random"):
        r = np.random.default_rng(RANDOM_SEED)
        lab = list(r.choice(pool_idx, 10, replace=False))
        accs = []
        for _ in range(rounds):
            m = LogisticRegression(max_iter=2000).fit(Xc[lab], yc[lab])
            accs.append(m.score(Xc[test_idx], yc[test_idx]))
            rest = np.setdiff1d(pool_idx, lab)
            if strategy == "uncertainty":
                p = m.predict_proba(Xc[rest])[:, 1]
                pick = rest[np.argsort(np.abs(p - 0.5))[:batch]]
            else:
                pick = r.choice(rest, batch, replace=False)
            lab += list(pick)
        curves[strategy] = accs
    return curves


curves = _active()
plot(lines(10 + 10 * np.arange(20), {"uncertainty sampling": curves["uncertainty"], "random sampling": curves["random"]},
           title="Test accuracy vs number of labels requested", xaxis="labels", yaxis="accuracy", markers=True), height=320)
why("استخدم Active learning حين تكون التسمية مكلفة (خبراء، مختبرات).",
    "اختيار الحالات الأكثر إرباكًا للنموذج يصل إلى دقة معينة بتسميات أقل — لكن العينة المُسمّاة تصبح غير عشوائية (احذر في التقييم).")

st.markdown("## Online learning: partial_fit عبر الدفعات")
st.caption("تيار بيانات يصل على 20 دفعة؛ عند الدفعة 10 تنقلب العلاقة جزئيًا (Concept drift). نموذج يتعلم تزايديًا مقابل نموذج جامد.")


@st.cache_data(show_spinner=False)
def _stream():
    r = np.random.default_rng(0)
    online = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.05, random_state=0)
    frozen = None
    acc_on, acc_fr = [], []
    for b in range(20):
        Xb = r.normal(size=(200, 2))
        w = np.array([2.0, -1.0]) if b < 10 else np.array([-1.0, 2.0])
        yb = (Xb @ w + r.normal(scale=0.5, size=200) > 0).astype(int)
        if b > 0:
            acc_on.append(online.score(Xb, yb))
            acc_fr.append(frozen.score(Xb, yb))
        online.partial_fit(Xb, yb, classes=[0, 1])
        if b == 0:
            frozen = LogisticRegression().fit(Xb, yb)
    return acc_on, acc_fr


acc_on, acc_fr = _stream()
fig = lines(np.arange(1, 20), {"online SGD (partial_fit)": acc_on, "frozen model (trained on batch 0)": acc_fr},
            title="Accuracy on each new batch (evaluate, then learn)", xaxis="batch", yaxis="accuracy", markers=True)
fig.add_vline(x=10, line=dict(color=PALETTE["coral"], dash="dash"), annotation_text="concept drift")
plot(fig, height=320)
page_link("drift", "المزيد: وحدة Drift والمراقبة", ":material/monitoring:")

if at_least("advanced"):
    st.markdown("## متقدم: Transfer وSelf-supervised (مفاهيم)")
    st.markdown("- **Transfer:** استخدم تمثيلًا مُدرَّبًا على بيانات ضخمة (مثل Embeddings نصية) كخصائص لنموذج خفيف على بياناتك.\n"
                "- **Self-supervised:** مهمة مصطنعة من البيانات نفسها (توقّع الجزء المحجوب) تنتج تمثيلات قوية؛ أساس النماذج اللغوية.\n"
                "- كلاهما يُدرَّس بعمق في منصة التعلّم العميق المستقلة لاحقًا.")
if at_least("research"):
    researcher_note(["في Active learning، العينة المُسمّاة متحيزة؛ لتقدير الأداء استخدم عينة اختبار عشوائية منفصلة.",
                     "Gama et al. (2014) مسح شامل لتكيّف النماذج مع انجراف المفهوم."])
    st.markdown(cite("gama2014"))
mistakes(["تقييم Active learning على الحالات التي اختارها.", "Semi-supervised حين لا يتحقق افتراض العنقود.",
          "partial_fit دون تمرير classes في أول استدعاء."])
page_footer("modern_topics",
            takeaways=["Semi-supervised يستغل البيانات غير المسماة بافتراضات بنيوية.", "Active learning يوفر التسميات باختيار ذكي.",
                       "partial_fit يسمح بالتعلم التزايدي والتكيّف مع الانجراف."])
