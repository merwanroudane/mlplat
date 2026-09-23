import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.animation import stepper
from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d
from utils.plotting import CLASS_COLORS, lines, plot

page_header("dbscan")
algorithm_profile("dbscan")

definition("نقاط DBSCAN الثلاث",
           "**Core:** لها ≥ min_samples نقطة (بما فيها نفسها) ضمن نصف قطر eps. **Border:** ليست Core لكنها ضمن eps من نقطة Core. "
           "**Noise:** ليست هذا ولا ذاك. العنقود = كل النقاط القابلة للوصول كثافيًا من Core واحدة.")


def _dataset(name):
    if name == "moons":
        X, y = toy_2d("moons", n=300, noise=0.07)
    elif name == "circles":
        X, y = toy_2d("circles", n=300, noise=0.1)
    else:  # varied densities
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal([0, 0], 0.25, (150, 2)), rng.normal([3, 0], 0.9, (150, 2)), rng.uniform(-2, 6, (30, 2))])
        y = np.r_[np.zeros(150), np.ones(150), -np.ones(30)].astype(int)
    return StandardScaler().fit_transform(X), y


st.markdown("## DBSCAN Lab: eps وmin_samples")
c1, c2, c3 = st.columns(3)
data = c1.segmented_control("البيانات", ["moons", "circles", "varied densities"], default="moons", key="db_data", required=True)
eps = c2.slider("eps", 0.05, 1.0, 0.25, 0.01, key="db_eps")
ms = c3.slider("min_samples", 2, 30, 5, key="db_ms")
X, truth = _dataset(data)


def _plot_db(labels, core_mask, title):
    fig = go.Figure()
    for j in sorted(set(labels)):
        m = labels == j
        if j == -1:
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name="noise",
                                     marker=dict(symbol="x", size=7, color="#868E96")))
            continue
        color = CLASS_COLORS[j % len(CLASS_COLORS)]
        fig.add_trace(go.Scatter(x=X[m & core_mask, 0], y=X[m & core_mask, 1], mode="markers", name=f"cluster {j} core",
                                 marker=dict(size=8, color=color)))
        fig.add_trace(go.Scatter(x=X[m & ~core_mask, 0], y=X[m & ~core_mask, 1], mode="markers", name=f"cluster {j} border",
                                 marker=dict(size=8, color="white", line=dict(color=color, width=2))))
    fig.update_layout(title=title, height=420, showlegend=len(set(labels)) <= 6)
    return fig


db = DBSCAN(eps=eps, min_samples=ms).fit(X)
core = np.zeros(len(X), bool)
core[db.core_sample_indices_] = True
n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
plot(_plot_db(db.labels_, core, f"DBSCAN: {n_cl} clusters · noise {np.mean(db.labels_ == -1):.0%} · "
                                f"ARI vs truth {adjusted_rand_score(truth, db.labels_):.2f}"))

st.markdown("## Animation: ماذا يحدث حين يكبر eps؟")
eps_grid = np.round(np.linspace(0.08, 0.6, 12), 3)


def _frame(i: int) -> None:
    e = eps_grid[i]
    d = DBSCAN(eps=e, min_samples=ms).fit(X)
    c = np.zeros(len(X), bool)
    c[d.core_sample_indices_] = True
    k = len(set(d.labels_)) - (1 if -1 in d.labels_ else 0)
    plot(_plot_db(d.labels_, c, f"eps = {e} → {k} clusters, noise {np.mean(d.labels_ == -1):.0%}"))


stepper(f"db_anim_{data}_{ms}", len(eps_grid), _frame, labels=[f"eps {e}" for e in eps_grid])
intuition("eps صغير: كل شيء ضجيج. eps مناسب: الأشكال الحقيقية تظهر. eps كبير: كل شيء عنقود واحد.")

st.markdown("## اختيار eps: منحنى k-distance")
dist = np.sort(NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)[0][:, -1])
fig = lines(np.arange(len(dist)), {f"distance to {ms}-th neighbour": dist}, title="Sorted k-distance plot: look for the knee",
            xaxis="points sorted", yaxis="distance")
fig.add_hline(y=eps, line=dict(color=PALETTE["coral"], dash="dash"), annotation_text=f"current eps = {eps}")
plot(fig, height=300)
why("اجعل k في منحنى k-distance = min_samples، واختر eps عند «الركبة».", "النقاط يسار الركبة في مناطق كثيفة، ويمينها ضجيج.")

st.markdown("## HDBSCAN: كثافات متفاوتة دون eps")
mcs = st.slider("min_cluster_size", 5, 60, 15, key="db_mcs")
hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X)
km = KMeans(len(set(truth[truth >= 0])), n_init="auto", random_state=0).fit(X)
rows = [{"method": "DBSCAN (current eps)", "clusters": n_cl, "noise %": np.mean(db.labels_ == -1) * 100,
         "ARI": adjusted_rand_score(truth, db.labels_)},
        {"method": f"HDBSCAN (min_cluster_size={mcs})", "clusters": len(set(hd.labels_)) - (1 if -1 in hd.labels_ else 0),
         "noise %": np.mean(hd.labels_ == -1) * 100, "ARI": adjusted_rand_score(truth, hd.labels_)},
        {"method": "K-Means (true k)", "clusters": km.n_clusters, "noise %": 0.0, "ARI": adjusted_rand_score(truth, km.labels_)}]
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
hcore = hd.probabilities_ > 0.5
plot(_plot_db(hd.labels_, hcore, "HDBSCAN clusters (filled = membership probability > 0.5)"))
st.caption("في «varied densities» يفشل eps الواحد لـDBSCAN: إما يفتت العنقود المتفرق أو يدمج كل شيء. HDBSCAN يتكيف مع الكثافة. "
           "ملاحظة إصدار: HDBSCAN في 1.9.1 يحذّر إن لم يُحدَّد copy (ستتغير قيمته الافتراضية)؛ مررنا copy=True صراحة.")
comparison_table([
    {"": "K-Means", "k مسبقًا": "نعم", "الأشكال": "كروية", "الضجيج": "لا", "المعاملات": "k"},
    {"": "DBSCAN", "k مسبقًا": "لا", "الأشكال": "أي شكل", "الضجيج": "نعم", "المعاملات": "eps, min_samples"},
    {"": "HDBSCAN", "k مسبقًا": "لا", "الأشكال": "أي شكل + كثافات متفاوتة", "الضجيج": "نعم", "المعاملات": "min_cluster_size (min_samples)"},
])
hyperparameter_table("DBSCAN")
hyperparameter_table("HDBSCAN")

if at_least("advanced"):
    st.markdown("## متقدم: التعقيد")
    st.markdown("O(n log n) مع فهارس مكانية في الأبعاد المنخفضة، ويتدهور إلى O(n²). eps يعتمد على المقياس ⇒ القياس إلزامي؛ "
                "في الأبعاد العالية تفقد «الكثافة» معناها (لعنة الأبعاد).")
if at_least("research"):
    researcher_note(["Campello et al. (2013): HDBSCAN يبني شجرة عناقيد على كل مستويات الكثافة ويختار الأكثر ثباتًا (EOM).",
                     "DBSCAN كأداة كشف شذوذ: نقاط Noise مرشحة، لكنها تعتمد على eps."])
    st.markdown(cite("ester1996", "campello2013"))
template_checklist({5: "مختبر وAnimation", 7: "Core/Border/Noise", 9: "جداول المعاملات", 19: "k-distance plot",
                    22: "DBSCAN/HDBSCAN", 23: "DBSCAN Lab", 24: "eps/min_samples"})
mistakes(["دون قياس.", "eps واحد لكثافات متفاوتة.", "اعتبار كل Noise شذوذًا مؤكدًا."])
page_footer("dbscan",
            takeaways=["DBSCAN يعرّف العناقيد بالكثافة ويحدد الضجيج.", "اختر eps بمنحنى k-distance.",
                       "HDBSCAN يتعامل مع الكثافات المتفاوتة."])
