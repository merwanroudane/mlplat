import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import NMF, PCA, KernelPCA
from sklearn.manifold import TSNE

from components.algorithm_profile import algorithm_profile, hyperparameter_table
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.registry import optional
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import toy_2d
from utils.plotting import plot, scatter_classes

page_header("manifold")
algorithm_profile("tsne")

comparison_table([
    {"الطريقة": "PCA", "النوع": "خطي", "يحفظ": "التباين العالمي", "transform لبيانات جديدة": "نعم", "الاستخدام": "ضغط، معالجة مسبقة"},
    {"الطريقة": "Kernel PCA", "النوع": "غير خطي (نواة)", "يحفظ": "بنية في فضاء النواة", "transform لبيانات جديدة": "نعم", "الاستخدام": "بنية غير خطية معتدلة"},
    {"الطريقة": "NMF", "النوع": "خطي بقيود لا سالبة", "يحفظ": "أجزاء جمعية", "transform لبيانات جديدة": "نعم", "الاستخدام": "نصوص، صور، طيف"},
    {"الطريقة": "t-SNE", "النوع": "غير خطي", "يحفظ": "الجيران المحليين", "transform لبيانات جديدة": "لا", "الاستخدام": "تصوير فقط"},
    {"الطريقة": "UMAP (اختياري)", "النوع": "غير خطي", "يحفظ": "المحلي + بعض العالمي", "transform لبيانات جديدة": "نعم", "الاستخدام": "تصوير، معالجة"},
])

st.markdown("## Kernel PCA: حين يفشل PCA الخطي")
X, y = toy_2d("circles", n=400, noise=0.1)
gamma = st.select_slider("gamma (RBF)", [0.1, 1.0, 5.0, 10.0, 30.0], value=5.0, key="mf_gamma")
Zp = PCA(2).fit_transform(X)
Zk = KernelPCA(2, kernel="rbf", gamma=gamma).fit_transform(X)
c1, c2 = st.columns(2)
with c1:
    plot(scatter_classes(Zp, y).update_layout(title="PCA: just a rotation — circles stay nested", height=360))
with c2:
    plot(scatter_classes(Zk, y).update_layout(title=f"Kernel PCA (RBF, γ={gamma}): circles become separable", height=360))

st.markdown("## Embedding Lab: t-SNE على الأرقام المكتوبة (8×8)")
digits = load_digits()
n = st.select_slider("عدد الصور", [300, 600, 1000], value=600, key="mf_n")
c1, c2 = st.columns(2)
perp = c1.select_slider("perplexity", [2, 5, 15, 30, 50, 100], value=30, key="mf_perp")
method = c2.segmented_control("الطريقة", ["t-SNE", "PCA", "UMAP"], default="t-SNE", key="mf_method", required=True)


@st.cache_data(show_spinner="يحسب التضمين…", max_entries=16)
def _embed(method: str, perp: int, n: int):
    Xd, yd = digits.data[:n], digits.target[:n]
    if method == "PCA":
        return PCA(2).fit_transform(Xd), yd
    if method == "UMAP":
        umap = optional("umap")
        if umap is None:
            return None, yd
        return umap.UMAP(n_neighbors=max(2, perp), random_state=0).fit_transform(Xd), yd
    return TSNE(2, perplexity=min(perp, n - 1), init="pca", learning_rate="auto", random_state=0).fit_transform(Xd), yd


Z, yd = _embed(method, perp, n)
if Z is None:
    st.info("UMAP (حزمة umap-learn) اختيارية وغير مثبتة هنا؛ اعرض t-SNE أو PCA، أو ثبّت `pip install umap-learn` محليًا.",
            icon=":material/extension_off:")
else:
    fig = go.Figure()
    for d in range(10):
        m = yd == d
        fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers+text" if m.sum() < 5 else "markers", name=str(d),
                                 marker=dict(size=6, color=SEQUENCE[d % len(SEQUENCE)], symbol=["circle", "diamond", "square",
                                             "triangle-up", "x", "star", "cross", "triangle-down", "pentagon", "hexagon"][d])))
    fig.update_layout(title=f"{method} of {n} handwritten digits (64 pixels → 2D)", height=480)
    plot(fig)
warning("لا تقرأ من t-SNE: أحجام العناقيد، أو المسافات بين العناقيد، أو الكثافة. t-SNE يحفظ **من هو جار من** فقط؛ والشكل "
        "يتغير مع perplexity والبذرة. جرّب perplexity = 2 ثم 100.")
intuition("perplexity ≈ عدد الجيران الفعّال: صغيرة ⇒ بنية محلية جدًا (جزر صغيرة)، كبيرة ⇒ بنية أكثر شمولًا.")
hyperparameter_table("TSNE")

st.markdown("## NMF: أجزاء جمعية")
k = st.slider("عدد المكونات", 4, 16, 8, key="mf_nmf")
nmf = NMF(k, init="nndsvda", max_iter=500, random_state=0).fit(digits.data[:1000])
cols = st.columns(min(k, 8))
for j in range(k):
    with cols[j % len(cols)]:
        fig = go.Figure(go.Heatmap(z=nmf.components_[j].reshape(8, 8)[::-1], colorscale="Blues", showscale=False))
        fig.update_layout(height=120, margin=dict(l=2, r=2, t=18, b=2), title=dict(text=f"part {j + 1}", font=dict(size=11)),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        plot(fig)
st.caption("كل صورة رقم ≈ مجموع **موجب** لهذه الأجزاء (ضربات القلم). على عكس PCA الذي يسمح بالطرح فتصعب قراءة مكوناته.")
why("استخدم t-SNE/UMAP للاستكشاف البصري لا كخصائص لنموذج أو كدليل على عناقيد.",
    "التضمين غير مستقر وغير خطي ولا يحفظ المسافات؛ أي استنتاج كمي يجب أن يُتحقق منه في الفضاء الأصلي.")

if at_least("advanced"):
    st.markdown("## متقدم: الأهداف")
    st.latex(r"\text{t-SNE: } \min_Y \mathrm{KL}(P\,\|\,Q),\ p_{j|i}\propto e^{-\|x_i-x_j\|^2/2\sigma_i^2},\ q_{ij}\propto(1+\|y_i-y_j\|^2)^{-1}")
    st.markdown("الذيل الثقيل لتوزيع t في الفضاء المنخفض يسمح للنقاط غير المتجاورة بالتباعد (يحل مشكلة الازدحام). "
                "Barnes–Hut يجعل التعقيد O(n log n).")
    sr, t = make_swiss_roll(800, noise=0.05, random_state=0)
    st.caption("مثال كلاسيكي آخر: Swiss roll — PCA يضغطه فوق بعضه، بينما الطرق المحلية تفرده.")
if at_least("research"):
    researcher_note(["van der Maaten & Hinton (2008) لـt-SNE؛ McInnes et al. (2018) لـUMAP.",
                     "Kobak & Berens (2019) وغيرهم بيّنوا أن التهيئة (PCA) وقيم المعاملات تحدد كثيرًا مما يُظن «بنية»."])
    st.markdown(cite("vandermaaten2008", "mcinnes2018"))
mistakes(["قراءة المسافات بين عناقيد t-SNE.", "استخدام t-SNE لبيانات جديدة (لا transform).", "استنتاج عدد العناقيد من خريطة t-SNE."])
page_footer("manifold",
            takeaways=["Kernel PCA يلتقط بنية غير خطية مع transform.", "t-SNE للتصوير المحلي فقط؛ المسافات العالمية بلا معنى.",
                       "NMF يعطي أجزاء جمعية قابلة للتفسير."])
