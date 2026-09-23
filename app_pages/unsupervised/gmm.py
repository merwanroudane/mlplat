import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from components.algorithm_profile import algorithm_profile
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.plotting import CLASS_COLORS, lines, plot

page_header("gmm")
algorithm_profile("gmm")

formula(r"p(x) = \sum_{k=1}^K \pi_k\,\mathcal N(x;\mu_k,\Sigma_k),\qquad \gamma_{ik} = \frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal N(x_i;\mu_j,\Sigma_j)}",
        title="Gaussian mixture and responsibilities",
        symbols={r"\pi_k": "وزن المكوّن", r"\gamma_{ik}": "مسؤولية المكوّن k عن النقطة i (عضوية ناعمة)"},
        intuition="كل نقطة تنتمي لكل عنقود باحتمال. E-step يحسب المسؤوليات؛ M-step يحدّث π وμ وΣ كمتوسطات مرجّحة بها.")

rng = np.random.default_rng(1)
X = np.vstack([rng.multivariate_normal([0, 0], [[2.0, 1.2], [1.2, 1.0]], 200),
               rng.multivariate_normal([4, 3], [[0.4, 0], [0, 0.4]], 150),
               rng.multivariate_normal([4, -1.5], [[1.5, -0.6], [-0.6, 0.6]], 150)])


def _ellipse(mu, cov, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 80)
    circ = np.c_[np.cos(t), np.sin(t)] * n_std * np.sqrt(vals)
    return circ @ vecs.T + mu


@st.cache_data(show_spinner=False)
def _em_trace(steps: int = 12):
    out = []
    for it in range(1, steps + 1):
        g = GaussianMixture(3, covariance_type="full", max_iter=it, init_params="random_from_data", random_state=3,
                            warm_start=False, tol=0).fit(X)
        out.append((g.means_.copy(), g.covariances_.copy(), g.weights_.copy(), g.predict_proba(X), g.score(X)))
    return out


import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # early iterations have not converged by design
    trace = _em_trace()

st.markdown("## EM Lab: Animation")


def _frame(i: int) -> None:
    means, covs, w, resp, ll = trace[i]
    colors = resp @ np.array([[28, 126, 214], [247, 103, 7], [112, 72, 232]])
    fig = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=6, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors])))
    for k in range(3):
        e = _ellipse(means[k], covs[k])
        fig.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", name=f"component {k} (π={w[k]:.2f})",
                                 line=dict(color=CLASS_COLORS[k], width=3)))
    fig.update_layout(title=f"EM iteration {i + 1} · mean log-likelihood {ll:.3f} (colour = soft responsibilities)", height=440)
    plot(fig)


stepper("gmm_em", len(trace), _frame, labels=[f"iter {i + 1}" for i in range(len(trace))])
plot(lines(np.arange(1, len(trace) + 1), {"log-likelihood": [t[4] for t in trace]}, title="EM never decreases the likelihood",
           xaxis="iteration", markers=True), height=280)

st.markdown("## أنواع التغاير واختيار K بـBIC")
comparison_table([
    {"covariance_type": "full", "الشكل": "قطع ناقص بأي اتجاه", "معاملات لكل مكوّن": "p(p+1)/2"},
    {"covariance_type": "tied", "الشكل": "نفس الشكل لكل المكونات", "معاملات لكل مكوّن": "مشترك"},
    {"covariance_type": "diag", "الشكل": "محاذٍ للمحاور", "معاملات لكل مكوّن": "p"},
    {"covariance_type": "spherical", "الشكل": "دائري", "معاملات لكل مكوّن": "1"},
])


@st.cache_data(show_spinner=False)
def _bic():
    rows = []
    for ct in ("full", "tied", "diag", "spherical"):
        for k in range(1, 8):
            g = GaussianMixture(k, covariance_type=ct, random_state=0).fit(X)
            rows.append({"covariance_type": ct, "K": k, "BIC": g.bic(X)})
    return pd.DataFrame(rows)


b = _bic()
fig = go.Figure()
for i, ct in enumerate(("full", "tied", "diag", "spherical")):
    d = b[b["covariance_type"] == ct]
    fig.add_trace(go.Scatter(x=d["K"], y=d["BIC"], mode="lines+markers", name=ct, line=dict(color=CLASS_COLORS[i])))
best = b.loc[b["BIC"].idxmin()]
fig.update_layout(title=f"BIC (lower is better) · best: K = {int(best['K'])}, {best['covariance_type']}", xaxis_title="K", height=340)
plot(fig)
formula(r"\text{BIC} = -2\log\hat L + d\log n", title="Bayesian information criterion",
        symbols={"d": "عدد المعاملات", "n": "عدد النقاط"}, intuition="الملاءمة مع عقوبة التعقيد؛ يساعد في اختيار K ونوع التغاير.")

st.markdown("## GMM مقابل K-Means")
km = KMeans(3, n_init="auto", random_state=0).fit(X)
gm = GaussianMixture(3, random_state=0).fit(X)
agree = max(np.mean(km.labels_ == np.array(perm)[gm.predict(X)]) for perm in
            [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
st.caption(f"اتفاق التعيينات = {agree:.1%}. الاختلاف يتركز حيث العناقيد مستطيلة: K-Means يفترض كرات متساوية.")
intuition("GMM يعطي أيضًا كثافة p(x): النقاط ذات p(x) منخفضة مرشحة كشذوذ، ويمكن توليد عينات جديدة (نموذج توليدي).")

if at_least("advanced"):
    st.markdown("## متقدم: EM كتعظيم لحد أدنى")
    st.markdown("EM يعظّم حدًا أدنى (ELBO) لـlog-likelihood: E-step يجعل الحد مماسًا، M-step يعظّمه. الضمان: عدم التناقص؛ "
                "لا ضمان للحد الأقصى العالمي ⇒ `n_init` وتهيئات متعددة. التفرد (Σ → 0 حول نقطة) يُمنع بـ`reg_covar`.")
if at_least("research"):
    researcher_note(["Dempster, Laird & Rubin (1977) صاغوا EM عامًا للبيانات غير المكتملة؛ المسؤوليات = «البيانات المفقودة» (المكوّن).",
                     "BayesianGaussianMixture يضع مسبقًا ديريكليه على الأوزان فيُطفئ المكونات غير اللازمة."])
    st.markdown(cite("dempster1977", "esl"))
mistakes(["K من الأرجحية وحدها (تزيد دائمًا).", "full covariance مع n صغير وp كبير.", "تجاهل التهيئات المتعددة."])
page_footer("gmm",
            takeaways=["GMM = تجميع احتمالي ناعم بمكونات غاوسية.", "EM يتناوب بين المسؤوليات والتحديث.",
                       "BIC لاختيار K ونوع التغاير."])
