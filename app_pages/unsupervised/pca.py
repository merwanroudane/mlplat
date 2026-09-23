import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.code_lab import code_lab
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.models import PCAScratch
from utils.plotting import bars, heatmap, lines, plot

page_header("pca")
algorithm_profile("pca")

formula(r"w_1 = \arg\max_{\|w\|=1} \operatorname{Var}(Xw) = \arg\max_{\|w\|=1} w^\top\Sigma w\ \Rightarrow\ \Sigma w_1 = \lambda_1 w_1",
        title="First principal component = top eigenvector of the covariance",
        symbols={r"\Sigma": "مصفوفة التغاير للبيانات المركزية", r"\lambda_1": "أكبر قيمة ذاتية = التباين على w₁"},
        intuition="ابحث عن الاتجاه الذي تنتشر فيه البيانات أكثر؛ ثم أفضل اتجاه عمودي عليه؛ وهكذا.",
        example="بيانات طول/وزن مترابطة بقوة: المكوّن الأول ≈ «الحجم العام»، والثاني ≈ «النحافة مقابل الامتلاء».")
formula(r"X_c = U\Sigma V^\top,\quad \text{components} = V,\quad \text{scores} = X_cV = U\Sigma,\quad \lambda_j = \frac{\sigma_j^2}{n-1}",
        title="PCA via SVD (what scikit-learn does)",
        intuition="لا حاجة لتكوين مصفوفة التغاير صراحة؛ SVD للبيانات المركزية أدق عدديًا.")

st.markdown("## الهندسة في بعدين")
rho = st.slider("الترابط بين x₁ وx₂", -0.95, 0.95, 0.8, 0.05, key="pca_rho")
rng = np.random.default_rng(0)
X2 = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 300) * [2.0, 1.0]
p2 = PCA(2).fit(X2)
fig = go.Figure(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers", marker=dict(color=PALETTE["muted"], opacity=0.5), name="data"))
for j in range(2):
    v = p2.components_[j] * 2.5 * np.sqrt(p2.explained_variance_[j])
    fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers", name=f"PC{j + 1} ({p2.explained_variance_ratio_[j]:.0%})",
                             line=dict(color=SEQUENCE[j], width=4)))
fig.update_layout(height=420, yaxis=dict(scaleanchor="x"), title="Principal axes (length ∝ √variance)")
plot(fig)

st.markdown("## PCA Lab على بيانات Wine (13 خاصية كيميائية)")
wine = load_wine()
Xw = pd.DataFrame(wine.data, columns=wine.feature_names)
scale = st.toggle("StandardScaler قبل PCA", value=True, key="pca_scale")
Xs = StandardScaler().fit_transform(Xw) if scale else Xw.to_numpy()
pca = PCA().fit(Xs)
c1, c2 = st.columns(2)
with c1:
    evr = pca.explained_variance_ratio_
    fig = bars([f"PC{i + 1}" for i in range(len(evr))], evr, title="Scree plot", color=PALETTE["sky"], text_fmt=".2f")
    fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(evr))], y=np.cumsum(evr), mode="lines+markers", name="cumulative",
                             line=dict(color=PALETTE["coral"])))
    plot(fig, height=340)
with c2:
    Z = pca.transform(Xs)
    fig = go.Figure()
    for k, sym in zip(range(3), ["circle", "diamond", "square"]):
        m = wine.target == k
        fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=f"cultivar {k}", marker=dict(symbol=sym, size=7,
                                                                                                          color=SEQUENCE[k])))
    fig.update_layout(title="Scores on PC1–PC2 (labels not used by PCA)", xaxis_title="PC1", yaxis_title="PC2", height=340)
    plot(fig)
if not scale:
    st.warning(f"دون قياس، PC1 يفسّر {evr[0]:.0%} من التباين لأنه يلتقط تقريبًا خاصية `proline` وحدها (مقياسها بالمئات). PCA حساس "
               "للوحدات.")
load = pd.DataFrame(pca.components_[:3].T, index=wine.feature_names, columns=["PC1", "PC2", "PC3"])
plot(heatmap(load.to_numpy(), load.columns, load.index, title="Loadings (weights of original features)",
             colorscale=[[0, "#7048E8"], [0.5, "#FCFCFF"], [1, "#F76707"]], zmid=0), height=460)
intuition("التحميلات تقول «ما الذي يعنيه كل مكوّن»؛ الدرجات (Scores) تقول «أين تقع كل ملاحظة» عليه.")

st.markdown("## كم مكوّنًا؟ PCA قبل النموذج")


@st.cache_data(show_spinner=False)
def _k_curve():
    ks = list(range(1, 14))
    sc = [cross_val_score(make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=2000)), Xw, wine.target, cv=5).mean()
          for k in ks]
    return ks, sc


ks, sc = _k_curve()
plot(lines(ks, {"5-fold accuracy": sc}, title="Logistic regression on the first k components", xaxis="k", markers=True), height=300)
why("ضع PCA داخل الـPipeline عند استخدامه قبل نموذج.", "إحصاءات PCA (المتوسط والمكونات) تُتعلَّم من البيانات؛ حسابها على كل "
    "البيانات قبل CV تسرّب طفيف لكنه منهجي.")

st.markdown("## من الصفر مقابل scikit-learn")


def _code(p):
    return ("# utils/models.py — PCA via SVD\n"
            "Xc = X - X.mean(0)\nU, S, Vt = np.linalg.svd(Xc, full_matrices=False)\n"
            f"components = Vt[:{p['k']}]\nexplained_variance = S[:{p['k']}] ** 2 / (n - 1)\nscores = Xc @ components.T\n\n"
            f"sk = PCA(n_components={p['k']}).fit(X)")


def _run(k):
    Xs_ = StandardScaler().fit_transform(Xw)
    a = PCAScratch(k).fit(Xs_)
    b = PCA(k).fit(Xs_)
    sign_agree = [abs(np.dot(a.components_[j], b.components_[j])) for j in range(k)]
    return pd.DataFrame({"component": [f"PC{j + 1}" for j in range(k)], "scratch explained var": a.explained_variance_,
                         "sklearn explained var": b.explained_variance_, "|cos| between directions": sign_agree}).round(6)


code_lab("pca_code", "PCAScratch vs sklearn.decomposition.PCA", _code, _run,
         lambda: {"k": st.slider("n_components", 1, 6, 3, key="pcac_k")},
         explanation="التباين المفسَّر متطابق، والاتجاهات متطابقة حتى الإشارة (|cos| = 1): إشارة المتجه الذاتي اعتباطية.")

if at_least("advanced"):
    st.markdown("## متقدم: Whitening، إعادة البناء، والتعقيد")
    st.markdown("- **إعادة البناء:** X̂ = Z Vₖᵀ + μ؛ الخطأ = مجموع القيم الذاتية المحذوفة.\n"
                "- **whiten=True:** يقسم الدرجات على √λ فتصبح بتباين 1 (مفيد لبعض النماذج، يضخّم الضجيج).\n"
                "- **التعقيد:** SVD كامل O(min(np², n²p))؛ `svd_solver='randomized'` للمكونات القليلة من بيانات كبيرة.\n"
                "- **IncrementalPCA** للبيانات التي لا تتسع للذاكرة.")
if at_least("research"):
    researcher_note(["Pearson (1901) قدّم الفكرة كأفضل مستوى ملاءمة؛ Hotelling (1933) صاغها كمكونات تباين.",
                     "PCA في الانحدار (PCR) يختار الاتجاهات دون النظر إلى y ⇒ قد يحذف اتجاهًا منخفض التباين ومهمًا للتنبؤ؛ "
                     "PLS بديل موجّه.", "التفسير السببي للمكونات غير مبرر؛ هي إحداثيات إحصائية."])
    st.markdown(cite("pearson1901", "esl"))
template_checklist({3: "مسألة تعظيم التباين وSVD", 5: "الهندسة في بعدين", 8: "components_ وexplained_variance_", 12: "مفتاح القياس",
                    19: "Scree plot", 20: "Loadings", 21: "PCAScratch", 22: "sklearn PCA", 23: "PCA Lab"})
mistakes(["PCA دون قياس لوحدات مختلفة.", "تفسير المكونات كعوامل سببية.", "PCA على كل البيانات قبل CV.",
          "مقارنة إشارات المكونات بين تشغيلين."])
page_footer("pca",
            takeaways=["PCA = اتجاهات التباين الأقصى المتعامدة = SVD للبيانات المركزية.", "Scree plot وLoadings وScores أدوات القراءة.",
                       "القياس يحدد النتيجة؛ ضعه في Pipeline."])
