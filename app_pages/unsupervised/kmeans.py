import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.animation import stepper
from components.callouts import mistakes, researcher_note, why
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import load_dataset, toy_2d
from utils.models import kmeans_history
from utils.plotting import CLASS_COLORS, SYMBOLS, lines, plot

page_header("kmeans")
algorithm_profile("kmeans")

formula(r"\min_{C_1..C_k,\ \mu_1..\mu_k}\ \sum_{j=1}^k\sum_{x_i\in C_j}\|x_i-\mu_j\|^2", title="K-Means objective (inertia)",
        symbols={r"\mu_j": "مركز العنقود j", "C_j": "النقاط المعيّنة له"},
        intuition="اعثر على k مراكز بحيث تكون كل نقطة قريبة من مركزها. Lloyd يتناوب: عيّن ← حدّث، والهدف لا يزيد أبدًا.",
        example="بعد التعيين: المتوسط هو المركز الأمثل لمجموعة ثابتة؛ بعد التحديث: أقرب مركز هو التعيين الأمثل.")

st.markdown("## K-Means Lab: Animation خطوة بخطوة")
c1, c2, c3, c4 = st.columns(4)
data = c1.segmented_control("البيانات", ["blobs", "moons"], default="blobs", key="km_data", required=True)
k = c2.slider("k", 2, 8, 4, key="km_k")
init = c3.segmented_control("البداية", ["random", "k-means++"], default="random", key="km_init", required=True)
seed = c4.number_input("seed", 0, 50, 3, key="km_seed")
X = load_dataset("blobs")[["x1", "x2"]].to_numpy() if data == "blobs" else toy_2d("moons", n=300, noise=0.08)[0]
hist = kmeans_history(X, k, n_iter=15, seed=int(seed), init=init)


def _frame(i: int) -> None:
    h = hist[i]
    fig = go.Figure()
    if h["labels"] is None:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color="#ADB5BD", size=7), name="unassigned"))
    else:
        for j in range(k):
            m = h["labels"] == j
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=f"cluster {j}",
                                     marker=dict(color=CLASS_COLORS[j % len(CLASS_COLORS)], symbol=SYMBOLS[j % len(SYMBOLS)], size=7,
                                                 opacity=0.7)))
    if i > 0 and hist[i - 1]["centers"] is not None and h["phase"] == "update":
        prev = hist[i - 1]["centers"]
        for j in range(k):
            fig.add_annotation(x=h["centers"][j, 0], y=h["centers"][j, 1], ax=prev[j, 0], ay=prev[j, 1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#212529")
    fig.add_trace(go.Scatter(x=h["centers"][:, 0], y=h["centers"][:, 1], mode="markers", name="centres",
                             marker=dict(symbol="x", size=18, color="#212529", line=dict(width=3))))
    title = {"init": f"Initial centres ({init})", "assign": "Assign: each point → nearest centre",
             "update": "Update: each centre → mean of its points"}[h["phase"]]
    fig.update_layout(title=title + (f" · inertia = {h['inertia']:.1f}" if h["inertia"] is not None else ""), height=440)
    plot(fig)


labels = ["init"] + [f"{'assign' if j % 2 == 0 else 'update'} {j // 2 + 1}" for j in range(len(hist) - 1)]
stepper(f"km_{data}_{k}_{init}_{seed}", len(hist), _frame, labels=labels)
inertias = [h["inertia"] for h in hist if h["inertia"] is not None]
plot(lines(np.arange(1, len(inertias) + 1), {"inertia": inertias}, title="Inertia never increases (monotone convergence)",
           xaxis="half-step", yaxis="inertia", markers=True), height=280)
if data == "moons":
    st.warning("K-Means يفترض عناقيد كروية محدبة؛ على الهلالين يقطعهما بخط مستقيم. استخدم DBSCAN أو Spectral clustering.")

st.markdown("## اختيار k: Elbow وSilhouette")
Xb = load_dataset("blobs")
Xk = Xb[["x1", "x2"]].to_numpy()


@st.cache_data(show_spinner=False)
def _k_scan():
    ks = range(2, 10)
    inert, sil = [], []
    for kk in ks:
        m = KMeans(kk, n_init="auto", random_state=0).fit(Xk)
        inert.append(m.inertia_)
        sil.append(silhouette_score(Xk, m.labels_))
    return list(ks), inert, sil


ks, inert, sil = _k_scan()
c1, c2 = st.columns(2)
with c1:
    plot(lines(ks, {"inertia": inert}, title="Elbow: inertia vs k", xaxis="k", markers=True), height=300)
with c2:
    plot(lines(ks, {"mean silhouette": sil}, title="Silhouette vs k (higher is better)", xaxis="k", markers=True), height=300)
formula(r"s(i) = \frac{b(i)-a(i)}{\max\{a(i), b(i)\}}", title="Silhouette coefficient",
        symbols={"a(i)": "متوسط المسافة إلى نقاط عنقودها", "b(i)": "متوسط المسافة إلى أقرب عنقود آخر"},
        intuition="قرب 1: موضوعة جيدًا؛ قرب 0: على الحدود؛ سالب: ربما في العنقود الخطأ.")
km = KMeans(4, n_init="auto", random_state=0).fit(Xk)
s = silhouette_samples(Xk, km.labels_)
st.caption(f"مع k = 4: متوسط Silhouette = {s.mean():.3f}، ونسبة النقاط ذات s < 0 = {(s < 0).mean():.1%}. "
           f"ARI مع العناقيد الحقيقية = {adjusted_rand_score(Xb['cluster_true'], km.labels_):.3f} (متاح فقط لأن البيانات اصطناعية).")
why("لا تختر k بمقياس واحد.", "Inertia تنخفض دائمًا مع k. اجمع Elbow وSilhouette واستقرار العناقيد عبر Bootstrap ومعنى العناقيد للمجال.")

st.markdown("## من الصفر مقابل scikit-learn")
st.code("""# utils/models.py — Lloyd's algorithm
for _ in range(n_iter):
    labels = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)           # assign
    C = np.array([X[labels == j].mean(0) for j in range(k)])               # update
# library
KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=0).fit(X)""", language="python")
scratch_final = kmeans_history(Xk, 4, n_iter=50, seed=0, init="k-means++")[-1]
st.caption(f"Inertia (scratch, k-means++) = {scratch_final['inertia']:.2f} · sklearn = {km.inertia_:.2f} — نفس الهدف، والفرق من "
           "البداية فقط (sklearn يأخذ أفضل n_init بدايات).")
hyperparameter_table("KMeans")

if at_least("advanced"):
    st.markdown("## متقدم: k-means++ والتعقيد")
    st.markdown("- **k-means++:** اختر كل مركز جديد باحتمال ∝ D(x)² ⇒ ضمان تقريب O(log k) للهدف الأمثل في التوقع.\n"
                "- **التعقيد:** O(n·k·p) لكل تكرار؛ MiniBatchKMeans للبيانات الضخمة.\n"
                "- **القياس إلزامي:** المسافة الإقليدية.\n- مسألة الهدف الأمثل عالميًا NP-hard؛ Lloyd يصل لحد أدنى محلي.")
if at_least("research"):
    researcher_note(["K-Means = حالة حدّية لـGMM بتغايرات σ²I متساوية وσ → 0 (تعيين صلب).",
                     "العناقيد ليست «حقيقة»: تحقق من الاستقرار (Clustering stability) ومن الصلاحية الخارجية قبل تفسيرها."])
    st.markdown(cite("lloyd1982", "esl"))
template_checklist({3: "هدف Inertia", 5: "Animation", 7: "Lloyd: assign/update", 8: "cluster_centers_", 9: "جدول KMeans",
                    19: "Elbow/Silhouette", 21: "kmeans_history", 22: "KMeans", 23: "K-Means Lab", 24: "k/init/seed"})
mistakes(["دون قياس.", "k من Inertia وحدها.", "K-Means لأشكال غير كروية.", "تفسير العناقيد كفئات حقيقية دون تحقق."])
page_footer("kmeans",
            takeaways=["Lloyd يتناوب بين التعيين والتحديث؛ Inertia تتناقص رتيبًا.", "k-means++ وn_init يحسّنان البداية.",
                       "اختر k بأدلة متعددة ومعرفة المجال."])
