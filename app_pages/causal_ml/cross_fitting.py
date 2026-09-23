import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from components.animation import stepper
from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils import causal as C
from utils.datasets import make_plr
from utils.plotting import plot

page_header("cross_fitting")

definition("Cross-fitting", "قسّم البيانات إلى K طية. لكل طية k: درّب دوال الإزعاج على الطيات **الأخرى**، وتنبأ بها على الطية k. "
           "بعد التدوير يحصل كل صف على تنبؤ إزعاج «خارج الطية». احسب الدرجة المتعامدة على كل البيانات وحل لـθ.")
mermaid("""
flowchart LR
  D[(Dataset n)] --> F[Split into K folds]
  F --> T1[Fold 1 held out] --> N1[Train ℓ̂, m̂ on folds 2..K] --> P1[Predict on fold 1]
  F --> T2[Fold 2 held out] --> N2[Train ℓ̂, m̂ on folds 1,3..K] --> P2[Predict on fold 2]
  F --> TK[Fold K held out] --> NK[Train on the rest] --> PK[Predict on fold K]
  P1 & P2 & PK --> OOF[Out-of-fold nuisance predictions for all n]
  OOF --> S[Orthogonal score ψ_i] --> E[θ̂ and SE]
""")

st.markdown("## DML Cross-Fitting Lab: Animation")
K = st.slider("عدد الطيات K", 2, 6, 3, key="cf_K")
df = make_plr(n=60, p=3, seed=5)
X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
folds = C.fold_indices(len(y), K, seed=0)
learner = RandomForestRegressor(50, min_samples_leaf=3, random_state=0, n_jobs=1)


@st.cache_data(show_spinner=False, max_entries=8)
def _oof(K: int):
    fl = C.fold_indices(len(y), K, seed=0)
    ell = C.cross_fit_predict(learner, X, y, fl)
    m = C.cross_fit_predict(learner, X, d, fl)
    return ell, m


ell_hat, m_hat = _oof(K)
fold_of = np.empty(len(y), int)
for k, (_, te) in enumerate(folds):
    fold_of[te] = k
steps = []
for k in range(K):
    steps.append(("train", k))
    steps.append(("predict", k))
steps.append(("score", -1))
steps.append(("estimate", -1))


def _frame(i: int) -> None:
    phase, k = steps[i]
    order = np.argsort(fold_of, kind="stable")
    status = np.full(len(y), "not yet", dtype=object)
    done = set()
    for j in range(i + 1):
        ph, kk = steps[j]
        if ph == "predict":
            done.add(kk)
    for kk in done:
        status[fold_of == kk] = "predicted"
    if phase in ("train", "predict"):
        status[fold_of != k] = np.where(status[fold_of != k] == "predicted", "predicted (train now)", "training")
        status[fold_of == k] = "held out → predicted" if phase == "predict" else "held out"
    colors = {"not yet": "#E9ECEF", "training": PALETTE["sky"], "held out": PALETTE["coral"], "held out → predicted": PALETTE["coral"],
              "predicted": PALETTE["teal"], "predicted (train now)": "#74C0FC"}
    fig = go.Figure(go.Bar(x=np.arange(len(y)), y=np.ones(len(y)), marker_color=[colors[s] for s in status[order]],
                           hovertext=[f"row {r}, fold {fold_of[r] + 1}: {status[r]}" for r in order], hoverinfo="text"))
    for kk in range(1, K):
        fig.add_vline(x=np.searchsorted(fold_of[order], kk) - 0.5, line=dict(color="#212529", width=2))
    title = {"train": f"Fold {k + 1}: train ℓ̂ and m̂ on the other {K - 1} folds (blue)",
             "predict": f"Fold {k + 1}: predict nuisances on the held-out fold (orange) — rows never seen by the model",
             "score": "All rows now have out-of-fold predictions: compute residuals Û = Y − ℓ̂, V̂ = D − m̂",
             "estimate": "Solve the orthogonal moment on all n rows: θ̂ = ΣV̂Û / ΣV̂²"}[phase]
    fig.update_layout(title=title, height=190, yaxis=dict(visible=False), xaxis=dict(title="rows grouped by fold"), bargap=0.05,
                      margin=dict(t=40, b=30))
    plot(fig)
    if phase in ("score", "estimate"):
        u, v = y - ell_hat, d - m_hat
        res = C.plr_from_nuisance(y, d, ell_hat, m_hat)
        f2 = go.Figure(go.Scatter(x=v, y=u, mode="markers", marker=dict(color=PALETTE["purple"], size=8)))
        if phase == "estimate":
            xs = np.linspace(v.min(), v.max(), 20)
            f2.add_trace(go.Scatter(x=xs, y=res.theta * xs, mode="lines", line=dict(color=PALETTE["coral"], width=3)))
        f2.update_layout(title=f"Residual-on-residual{f': θ̂ = {res.theta:.3f} (SE {res.se:.3f})' if phase == 'estimate' else ''}",
                         xaxis_title="V̂ = D − m̂(X)", yaxis_title="Û = Y − ℓ̂(X)", showlegend=False, height=320)
        plot(f2)


stepper(f"cf_anim_{K}", len(steps), _frame, labels=[f"{p} fold {k + 1}" if k >= 0 else p for p, k in steps])
intuition("كل صف تنبأ له نموذج **لم يره**. لذلك خطأ الإزعاج في ذلك الصف مستقل عن ضجيجه، فلا يتسرب «الإفراط» إلى θ̂. "
          "والتدوير يجعلنا نستخدم كل البيانات للتقدير (لا نصفها كما في التقسيم البسيط).")

st.markdown("## Cross-Validation مقابل Cross-Fitting")
comparison_table([
    {"": "Cross-validation", "على الطية المحجوزة": "نقيس أداء (خطأ)", "المخرج": "تقدير أداء التعميم", "في DML": "داخل كل طية لضبط المتعلمين"},
    {"": "Cross-fitting", "على الطية المحجوزة": "نتنبأ بالإزعاج", "المخرج": "تنبؤات خارج الطية لكل n ثم θ̂", "في DML": "هيكل التقدير نفسه"},
])
why("لا تخلط بين الاثنين في الإبلاغ.", "«استخدمنا 5-fold» قد تعني CV للضبط أو Cross-fitting للتقدير؛ صرّح بكليهما.")

st.markdown("## DML1 مقابل DML2 وتكرار التقسيم (n_rep)")
formula(r"\text{DML1: } \hat\theta = \frac1K\sum_k \hat\theta_k\qquad \text{DML2: solve } \frac1n\sum_{i=1}^n \psi(W_i;\theta,\hat\eta_{k(i)}) = 0",
        title="Two ways to aggregate across folds",
        intuition="DML1 يحل لكل طية ثم يتوسط؛ DML2 يجمع الدرجات من كل الطيات ثم يحل مرة واحدة — أكثر استقرارًا في العينات الصغيرة "
                  "وهو المستخدم في DoubleML وفي utils/causal.py.")


@st.cache_data(show_spinner="يكرر التقسيم 10 مرات…")
def _nrep():
    d2 = make_plr(n=500, seed=21)
    Xa, ya, da = d2.filter(like="x").to_numpy(), d2["y"].to_numpy(), d2["d"].to_numpy()
    rf = RandomForestRegressor(80, min_samples_leaf=5, random_state=0, n_jobs=1)
    res = C.dml_plr(Xa, ya, da, rf, rf, n_folds=5, n_rep=10, seed=100)
    return res


res = _nrep()
thetas = np.array(res.extra["thetas"])
c1, c2 = st.columns([1.5, 1])
with c1:
    fig = go.Figure(go.Scatter(x=np.arange(1, 11), y=thetas, mode="markers+lines", marker=dict(size=10, color=PALETTE["sky"])))
    fig.add_hline(y=np.median(thetas), line=dict(color=PALETTE["coral"], dash="dash"), annotation_text="median")
    fig.add_hline(y=0.5, line=dict(color=PALETTE["teal"], dash="dot"), annotation_text="θ₀")
    fig.update_layout(title="θ̂ across 10 different random fold assignments", xaxis_title="split repetition", height=300)
    plot(fig)
with c2:
    st.metric("وسيط θ̂ (n_rep = 10)", f"{res.theta:.4f}")
    st.metric("SE مصحح بالوسيط", f"{res.se:.4f}")
    st.metric("مدى التقديرات عبر التقسيمات", f"{thetas.min():.3f} – {thetas.max():.3f}")
st.caption("التقسيم العشوائي نفسه مصدر تباين؛ n_rep > 1 مع التجميع بالوسيط يجعل النتيجة أقل اعتمادًا على «حظ» التقسيم "
           "(Chernozhukov et al., 2018، §3.4). في DoubleML: `n_rep`.")

if at_least("advanced"):
    st.markdown("## متقدم: اختيار K")
    st.markdown("K أكبر ⇒ كل نموذج إزعاج يُدرَّب على نسبة أكبر (K−1)/K من البيانات ⇒ إزعاج أدق، بتكلفة K تدريبًا. 4–5 خيار شائع؛ "
                "مع البيانات المجمّعة يجب أن تحترم الطيات المجموعات (Cluster-robust DML).")
if at_least("research"):
    researcher_note(["Cross-fitting يلغي الحاجة لشروط Donsker على فضاء الإزعاج — أي يسمح بمتعلمين مرنين جدًا.",
                     "مع n_rep > 1 أبلغ عن الوسيط والخطأ المعياري المعدل وعن عدد التكرارات."])
    st.markdown(cite("chernozhukov2018"))
page_link("plr", "التالي: PLR بالتفصيل", ":material/linear_scale:")
mistakes(["تنبؤات إزعاج داخل العينة.", "طيات تكسر المجموعات في بيانات مجمّعة.", "الاعتماد على تقسيم واحد في عينات صغيرة."])
page_footer("cross_fitting",
            takeaways=["كل صف يحصل على تنبؤ إزعاج من نموذج لم يره.", "DML2 يجمع الدرجات ثم يحل مرة واحدة.",
                       "n_rep مع الوسيط يقلل حساسية التقسيم."])
