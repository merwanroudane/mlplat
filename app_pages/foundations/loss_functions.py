import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils import metrics as M
from utils.plotting import plot

page_header("loss_functions")

st.markdown("## Loss مقابل Objective مقابل Metric")
comparison_table([
    {"المفهوم": "Loss ℓ(y, ŷ)", "التعريف": "خطأ ملاحظة واحدة", "مثال": "(y − ŷ)²", "من يستخدمه؟": "خوارزمية التدريب"},
    {"المفهوم": "Objective", "التعريف": "ما يُحسَّن فعلًا: متوسط الخسارة + عقوبة", "مثال": "MSE + α‖w‖²",
     "من يستخدمه؟": "الـSolver"},
    {"المفهوم": "Metric", "التعريف": "ما نبلغ عنه ونقرر به", "مثال": "ROC-AUC، MAE، التكلفة",
     "من يستخدمه؟": "الإنسان/المؤسسة"},
])
definition("لماذا لا نحسّن المقياس مباشرة؟", "كثير من المقاييس غير قابلة للاشتقاق (Accuracy، AUC)؛ فنحسّن خسارة "
           "**بديلة (Surrogate)** ملساء ومحدبة تتماشى معه، ثم نقيس المقياس الحقيقي على التحقق.")

st.markdown("## مختبر الخسائر: غيّر التنبؤ وشاهد العقوبة")
task = st.segmented_control("النوع", ["Regression", "Classification (margin)"], default="Regression", key="lf_task",
                            required=True)
if task == "Regression":
    c1, c2, c3 = st.columns(3)
    y_true = c1.number_input("y الحقيقي", value=3.0, step=0.5, key="lf_y")
    delta = c2.slider("Huber δ", 0.1, 3.0, 1.0, 0.1, key="lf_delta")
    tau = c3.slider("Quantile τ", 0.05, 0.95, 0.5, 0.05, key="lf_tau")
    y_hat = st.slider("التنبؤ ŷ", y_true - 6, y_true + 6, y_true + 2.0, 0.1, key="lf_yhat")
    r_grid = np.linspace(-6, 6, 400)
    losses = {"squared (MSE)": M.loss_squared(r_grid), "absolute (MAE)": M.loss_absolute(r_grid),
              f"Huber δ={delta}": M.loss_huber(r_grid, delta), f"quantile τ={tau}": M.loss_quantile(r_grid, tau)}
    r = y_true - y_hat
    fig = go.Figure()
    for i, (name, v) in enumerate(losses.items()):
        fig.add_trace(go.Scatter(x=r_grid, y=v, name=name, line=dict(color=SEQUENCE[i], width=2.6)))
    fig.add_vline(x=r, line=dict(color="#212529", dash="dot"), annotation_text=f"r = y − ŷ = {r:.1f}")
    fig.update_layout(title="Loss as a function of the residual r = y − ŷ", xaxis_title="residual r",
                      yaxis=dict(range=[0, 12], title="loss"), height=400)
    plot(fig)
    vals = {"squared": float(M.loss_squared(r)), "absolute": float(M.loss_absolute(r)),
            "Huber": float(M.loss_huber(r, delta)), "quantile": float(M.loss_quantile(r, tau))}
    with st.container(horizontal=True):
        for k, v in vals.items():
            st.metric(k, f"{v:.3f}", border=True)
    intuition("التربيعية تعاقب الأخطاء الكبيرة بشدة (حساسة للشواذ). المطلقة خطية (متينة، هدفها الوسيط). Huber تجمع الاثنين. "
              "Quantile غير متماثلة: مع τ = 0.9 يكلّف التنبؤ المنخفض أكثر، فيتجه النموذج نحو المئين 90.")
    st.markdown("**Poisson deviance** (لبيانات العدّ، μ > 0):")
    mu = st.slider("μ (التنبؤ)", 0.1, 10.0, 2.0, 0.1, key="lf_mu")
    yk = np.arange(0, 11)
    fig = go.Figure(go.Bar(x=yk, y=M.poisson_deviance_point(yk, mu), marker_color=PALETTE["purple"]))
    fig.update_layout(title=f"Unit Poisson deviance d(y, μ={mu}) for counts y = 0..10", xaxis_title="observed count y",
                      height=280)
    plot(fig)
else:
    m_grid = np.linspace(-3, 3, 400)
    m = st.slider("الهامش m = y·f(x)  (y ∈ {−1, +1})", -3.0, 3.0, 0.5, 0.1, key="lf_m")
    losses = {"0-1": M.loss_zero_one(m_grid), "hinge (SVM)": M.loss_hinge(m_grid), "logistic (scaled)": M.loss_logistic(m_grid),
              "exponential (AdaBoost)": M.loss_exponential(m_grid)}
    fig = go.Figure()
    for i, (name, v) in enumerate(losses.items()):
        fig.add_trace(go.Scatter(x=m_grid, y=v, name=name, line=dict(color=SEQUENCE[i], width=2.6,
                                                                      shape="hv" if name == "0-1" else "linear")))
    fig.add_vline(x=m, line=dict(color="#212529", dash="dot"), annotation_text=f"m = {m:.1f}")
    fig.update_layout(title="Classification losses as functions of the margin", xaxis_title="margin y·f(x)",
                      yaxis=dict(range=[0, 5], title="loss"), height=400)
    plot(fig)
    with st.container(horizontal=True):
        st.metric("0-1", f"{float(M.loss_zero_one(m)):.0f}", border=True)
        st.metric("hinge", f"{float(M.loss_hinge(m)):.3f}", border=True)
        st.metric("logistic", f"{float(M.loss_logistic(m)):.3f}", border=True)
        st.metric("exponential", f"{float(M.loss_exponential(m)):.3f}", border=True)
    intuition("كل الخسائر البديلة حدود عليا محدبة لخسارة 0-1. Hinge = 0 متى m ≥ 1 (لا يهتم بالنقاط البعيدة عن الحد ⇒ "
              "متجهات داعمة). Logistic لا تصل للصفر أبدًا (كل النقاط تساهم ⇒ احتمالات). Exponential تنفجر مع الأخطاء "
              "الكبيرة ⇒ حساسية AdaBoost للتسميات الخاطئة.")
    st.markdown("**Log loss وBrier** على مقياس الاحتمال:")
    p = np.linspace(0.001, 0.999, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=-np.log(p), name="log loss (y=1)", line=dict(color=PALETTE["sky"], width=2.5)))
    fig.add_trace(go.Scatter(x=p, y=(1 - p) ** 2, name="Brier (y=1)", line=dict(color=PALETTE["coral"], width=2.5)))
    fig.update_layout(xaxis_title="predicted P(y=1) when truth is 1", yaxis=dict(range=[0, 5]), height=300,
                      title="Probability-scale losses: log loss punishes confident mistakes without bound")
    plot(fig)

st.markdown("## كيف تغيّر الخسارة النموذج المتعلَّم؟")
st.caption("بيانات مع 10% قيم شاذة في y. نفس النموذج الخطي بثلاث خسائر.")
outl = st.slider("نسبة الشواذ", 0.0, 0.3, 0.1, 0.05, key="lf_out")


@st.cache_data(show_spinner=False)
def _fits(outl: float):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 120)
    y = 2 * x + 1 + rng.normal(size=120)
    k = int(outl * 120)
    idx = rng.choice(120, k, replace=False)
    y[idx] += rng.uniform(15, 30, k)
    X = x[:, None]
    g = np.linspace(0, 10, 50)[:, None]
    return x, y, g.ravel(), {
        "squared (OLS)": LinearRegression().fit(X, y).predict(g),
        "Huber": HuberRegressor().fit(X, y).predict(g),
        "absolute (median, τ=0.5)": QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(X, y).predict(g),
        "quantile τ=0.9": QuantileRegressor(quantile=0.9, alpha=0, solver="highs").fit(X, y).predict(g),
    }


x, y, g, fits = _fits(outl)
fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=PALETTE["muted"], opacity=0.6)))
for i, (name, pr) in enumerate(fits.items()):
    fig.add_trace(go.Scatter(x=g, y=pr, name=name, line=dict(color=SEQUENCE[i], width=2.8)))
fig.add_trace(go.Scatter(x=g, y=2 * g + 1, name="truth 2x+1", line=dict(color="#212529", dash="dot")))
fig.update_layout(height=400, title="Same model, different loss ⇒ different fitted line")
plot(fig)
why("اختر الخسارة من سؤال الأعمال.",
    "تريد المتوسط؟ خسارة تربيعية. الوسيط والمتانة؟ مطلقة/Huber. مخزونًا يكفي 90% من الأيام؟ Quantile τ=0.9. "
    "احتمالات موثوقة؟ Log loss.")

if at_least("advanced"):
    st.markdown("## متقدم: كل خسارة تستهدف دالة إحصائية")
    comparison_table([
        {"الخسارة": "Squared", "المُقلِّل الأمثل": "E[Y|X] (المتوسط الشرطي)", "توزيع ضمني": "Gaussian"},
        {"الخسارة": "Absolute", "المُقلِّل الأمثل": "الوسيط الشرطي", "توزيع ضمني": "Laplace"},
        {"الخسارة": "Pinball τ", "المُقلِّل الأمثل": "المئين الشرطي τ", "توزيع ضمني": "Asymmetric Laplace"},
        {"الخسارة": "Poisson deviance", "المُقلِّل الأمثل": "E[Y|X] للعدّ مع رابط log", "توزيع ضمني": "Poisson"},
        {"الخسارة": "Log loss", "المُقلِّل الأمثل": "P(Y=1|X) الحقيقية", "توزيع ضمني": "Bernoulli"},
        {"الخسارة": "Hinge", "المُقلِّل الأمثل": "sign(P(Y=1|X) − ½) — لا احتمالات", "توزيع ضمني": "—"},
        {"الخسارة": "Exponential", "المُقلِّل الأمثل": "½ log-odds", "توزيع ضمني": "—"},
    ])
if at_least("research"):
    researcher_note(["Proper scoring rules (Log loss، Brier) تُقلَّل بالاحتمالات الحقيقية؛ Hinge ليست Proper ⇒ SVM لا يعطي "
                     "احتمالات دون معايرة.",
                     "في DML نستخدم خسارة التنبؤ لمتعلمي الإزعاج، لكن الهدف النهائي هو الدرجة المتعامدة، لا الخسارة."])
    st.markdown(cite("huber1964", "esl"))
mistakes(["تقييم نموذج بـAccuracy وتدريبه بـHinge ثم استخدام مخرجاته كاحتمالات.", "استخدام MSE مع شواذ ثقيلة دون تفكير.",
          "الخلط بين دالة الهدف والمقياس المُبلغ عنه."])
page_footer("loss_functions",
            takeaways=["Loss لملاحظة، Objective للتدريب، Metric للقرار.", "كل خسارة تستهدف إحصائية مختلفة (متوسط، وسيط، مئين).",
                       "خسائر التصنيف بدائل محدبة لـ0-1."])
