import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from components.algorithm_profile import algorithm_profile
from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import load_dataset, toy_2d
from utils.plotting import plot, scatter_classes

page_header("hierarchical")
algorithm_profile("hierarchical")

st.markdown("## التجميع التجميعي (Agglomerative)")
intuition("ابدأ بكل نقطة عنقودًا مستقلًا؛ في كل خطوة ادمج أقرب عنقودين؛ استمر حتى يبقى عنقود واحد. النتيجة شجرة (Dendrogram) "
          "تقطعها عند الارتفاع الذي يعطي عدد العناقيد المطلوب.")
comparison_table([
    {"الربط (linkage)": "ward", "مسافة العنقودين": "الزيادة في مجموع المربعات عند الدمج", "الشكل الناتج": "عناقيد متقاربة الحجم كروية (مثل K-Means)"},
    {"الربط (linkage)": "complete", "مسافة العنقودين": "أبعد زوج", "الشكل الناتج": "عناقيد متماسكة"},
    {"الربط (linkage)": "average", "مسافة العنقودين": "متوسط كل الأزواج", "الشكل الناتج": "وسط"},
    {"الربط (linkage)": "single", "مسافة العنقودين": "أقرب زوج", "الشكل الناتج": "سلاسل طويلة (Chaining)، يلتقط الأشكال غير المحدبة"},
])

st.markdown("## Linkage Lab")
c1, c2, c3 = st.columns(3)
data = c1.segmented_control("البيانات", ["blobs", "moons"], default="blobs", key="hc_data", required=True)
method = c2.selectbox("linkage", ["ward", "complete", "average", "single"], key="hc_link")
k = c3.slider("عدد العناقيد", 2, 8, 4 if data == "blobs" else 2, key="hc_k")
if data == "blobs":
    df = load_dataset("blobs").sample(150, random_state=0)
    X, truth = df[["x1", "x2"]].to_numpy(), df["cluster_true"].to_numpy()
else:
    X, truth = toy_2d("moons", n=150, noise=0.06)
Z = linkage(X, method=method)
labels = fcluster(Z, k, criterion="maxclust") - 1
c1, c2 = st.columns(2)
with c1:
    fig = ff.create_dendrogram(X, linkagefun=lambda _: Z, color_threshold=Z[-(k - 1), 2] if k > 1 else 0)
    fig.update_layout(title=f"Dendrogram ({method})", height=400, xaxis=dict(showticklabels=False))
    fig.add_hline(y=(Z[-(k - 1), 2] + Z[-k, 2]) / 2, line=dict(dash="dash", color="#212529"), annotation_text=f"cut → {k} clusters")
    plot(fig)
with c2:
    plot(scatter_classes(X, labels).update_layout(title=f"{method}: silhouette {silhouette_score(X, labels):.2f} · "
                                                        f"ARI vs truth {adjusted_rand_score(truth, labels):.2f}", height=400))
intuition("على الهلالين: single linkage يتتبع الشكل (سلسلة نقاط متجاورة)، بينما ward يقطعهما كـK-Means. على الكتل: ward "
          "والأنواع الأخرى متقاربة، وsingle قد يدمج كتلًا عبر جسر من نقاط.")
st.code("""from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
AgglomerativeClustering(n_clusters=None, distance_threshold=5.0, linkage="average")   # cut by height instead""",
        language="python")
sk = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
st.caption(f"اتفاق scikit-learn مع SciPy (ARI) = {adjusted_rand_score(labels, sk):.3f}.")

if at_least("advanced"):
    st.markdown("## متقدم: التعقيد والقيود")
    st.markdown("- الذاكرة O(n²) لمصفوفة المسافات؛ الزمن O(n² log n) إلى O(n³) ⇒ عشرات الآلاف كحد عملي.\n"
                "- `connectivity` يقيد الدمج بجيران رسم بياني (بنية مكانية).\n"
                "- المسافة المترابطة (Cophenetic correlation) تقيس مدى تمثيل الشجرة للمسافات الأصلية.")
if at_least("research"):
    researcher_note(["Ward (1963) يدمج ما يقلل الزيادة في التباين داخل العناقيد — نفس هدف K-Means لكن جشعًا وهرميًا.",
                     "الشجرة تعطي عناقيد متداخلة على كل المستويات؛ مفيدة للتصنيفات الهرمية (Taxonomies)."])
    st.markdown(cite("ward1963", "esl"))
mistakes(["ward مع مسافة غير إقليدية.", "تطبيقه على مئات الآلاف من النقاط.", "قراءة ترتيب أوراق الشجرة أفقيًا كتشابه."])
page_footer("hierarchical",
            takeaways=["الدمج التدريجي ينتج Dendrogram يُقطع عند أي مستوى.", "نوع الربط يحدد شكل العناقيد.",
                       "مكلف في الذاكرة: O(n²)."])
