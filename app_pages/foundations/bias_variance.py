import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from config import RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import true_f_1d
from utils.plotting import lines, plot

page_header("bias_variance")

formula(r"\mathbb E\big[(y_0-\hat f(x_0))^2\big] = \underbrace{\big(f(x_0)-\mathbb E\hat f(x_0)\big)^2}_{\text{Bias}^2} + "
        r"\underbrace{\operatorname{Var}\big(\hat f(x_0)\big)}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{noise}}",
        title="Bias–variance decomposition (squared loss)",
        symbols={r"\mathbb E\hat f(x_0)": "متوسط تنبؤ النموذج عند x₀ عبر عينات تدريب مختلفة",
                 r"\operatorname{Var}(\hat f(x_0))": "تقلب التنبؤ من عينة لأخرى", r"\sigma^2": "ضجيج غير قابل للاختزال"},
        intuition="التحيز: خطأ منهجي لأن النموذج أبسط من الحقيقة. التباين: حساسية لتفاصيل العينة المحددة.",
        example="خط مستقيم على منحنى جيبي: تحيز كبير، تباين صغير. كثير حدود بدرجة 15 على 30 نقطة: العكس.")

st.markdown("## Animation: نفس النموذج على 20 عينة تدريب مختلفة")
st.caption("كل منحنى رمادي = نموذج دُرّب على عينة جديدة من نفس المجتمع. المنحنى البنفسجي = متوسطها. "
           "تحرّك عبر السعة (Play) لترى التحيز يتقلص والتباين يتضخم.")
family = st.segmented_control("عائلة النماذج", ["Polynomial degree", "Tree depth"], default="Polynomial degree",
                              key="bv_family", required=True)
c1, c2 = st.columns(2)
n = c1.slider("n لكل عينة", 15, 200, 30, 5, key="bv_n")
noise = c2.slider("σ", 0.05, 1.0, 0.3, 0.05, key="bv_noise")
caps = list(range(1, 13)) if family == "Polynomial degree" else list(range(1, 11))
grid = np.linspace(0, 1, 150)
truth = true_f_1d(grid)


@st.cache_data(show_spinner="يدرّب 20 نموذجًا لكل مستوى سعة…", max_entries=16)
def _simulate(family: str, n: int, noise: float, reps: int = 20):
    rng = np.random.default_rng(RANDOM_SEED)
    xs = [np.sort(rng.uniform(0, 1, n)) for _ in range(reps)]
    ys = [np.sin(2 * np.pi * x) + rng.normal(scale=noise, size=n) for x in xs]
    caps = list(range(1, 13)) if family == "Polynomial degree" else list(range(1, 11))
    preds = {}
    for c in caps:
        P = []
        for x, y in zip(xs, ys):
            m = (make_pipeline(PolynomialFeatures(c), Ridge(alpha=1e-8)) if family == "Polynomial degree"
                 else DecisionTreeRegressor(max_depth=c, random_state=0))
            P.append(m.fit(x[:, None], y).predict(grid[:, None]))
        preds[c] = np.clip(np.array(P), -4, 4)
    return preds


preds = _simulate(family, n, noise)
bias2 = [float(np.mean((preds[c].mean(0) - truth) ** 2)) for c in caps]
var = [float(np.mean(preds[c].var(0))) for c in caps]


def _frame(i: int) -> None:
    c = caps[i]
    P = preds[c]
    fig = go.Figure()
    for k, p in enumerate(P):
        fig.add_trace(go.Scatter(x=grid, y=p, mode="lines", line=dict(color="rgba(92,103,125,0.28)", width=1),
                                 showlegend=(k == 0), name="individual fits"))
    fig.add_trace(go.Scatter(x=grid, y=P.mean(0), mode="lines", name="average fit E[f̂]",
                             line=dict(color=PALETTE["purple"], width=3.5)))
    fig.add_trace(go.Scatter(x=grid, y=truth, mode="lines", name="truth f", line=dict(color=PALETTE["teal"], width=2.5, dash="dash")))
    fig.update_layout(title=f"{family} = {c} · Bias² = {bias2[i]:.3f} · Variance = {var[i]:.3f}",
                      yaxis=dict(range=[-2.5, 2.5]), height=400, legend=dict(orientation="h", y=1.12))
    left, right = st.columns([1.7, 1])
    with left:
        plot(fig)
    with right:
        fig2 = go.Figure(go.Bar(x=["Bias²", "Variance", "σ²"], y=[bias2[i], var[i], noise ** 2],
                                marker_color=[PALETTE["coral"], PALETTE["sky"], PALETTE["muted"]],
                                text=[f"{v:.3f}" for v in (bias2[i], var[i], noise ** 2)], textposition="auto"))
        fig2.update_layout(title=f"Expected test MSE ≈ {bias2[i] + var[i] + noise ** 2:.3f}", height=400,
                           yaxis=dict(range=[0, max(max(bias2), max(var), noise ** 2) * 1.1]))
        plot(fig2)


stepper(f"bv_{family}_{n}_{noise}", len(caps), _frame, labels=[f"{family.split()[0]} {c}" for c in caps])

fig = lines(caps, {"Bias²": bias2, "Variance": var, "Bias² + Variance + σ²": np.array(bias2) + np.array(var) + noise ** 2},
            title="The trade-off: pick the capacity that minimises the sum", xaxis=family, yaxis="error", markers=True,
            dash={"Bias² + Variance + σ²": "dash"})
fig.update_yaxes(type="log")
plot(fig, height=340)
best = caps[int(np.argmin(np.array(bias2) + np.array(var)))]
st.markdown(f"السعة المثلى هنا ≈ **{best}**. جرّب زيادة n: يتقلص التباين فتتحرك السعة المثلى نحو الأعلى.")

why("اختر السعة بالتحقق لا بالحدس.", "النقطة المثلى تعتمد على n وσ وشكل f؛ وكلها مجهولة في الواقع.")
intuition("أدوات تخفيض التباين: بيانات أكثر، تنظيم، تقليم، Bagging/Random Forest. أدوات تخفيض التحيز: نموذج أكثر مرونة، "
          "خصائص أفضل، Boosting.")

if at_least("advanced"):
    st.markdown("## متقدم: التحيز والتباين في التصنيف")
    st.markdown("مع خسارة 0-1 لا يوجد تفكيك جمعي نظيف؛ لكن الفكرة تبقى: Bagging يخفض التباين (Random Forest)، وBoosting "
                "يخفض التحيز أساسًا. مع Log loss يوجد تفكيك مشابه بالانحرافات (Bregman divergences).")
if at_least("research"):
    researcher_note(["التفكيك نظري: يتطلب معرفة f والقدرة على إعادة سحب عينات. في المحاكاة فقط نستطيع حسابه بدقة كما هنا.",
                     "Double descent (Belkin et al., 2019) يبيّن أن منحنى الخطأ قد ينخفض ثانية بعد عتبة الاستيفاء في "
                     "النماذج شديدة الإفراط في المعاملات."])
    st.markdown(cite("esl", "islp"))
mistakes(["الظن أن نموذجًا بلا تحيز هو الأفضل.", "زيادة السعة كلما انخفض خطأ التدريب.",
          "تقدير التباين من عينة واحدة."])
page_footer("bias_variance",
            takeaways=["الخطأ المتوقع = تحيز² + تباين + ضجيج.", "السعة تنقل الخطأ من التحيز إلى التباين.",
                       "البيانات الإضافية تقلل التباين وتسمح بسعة أكبر."])
