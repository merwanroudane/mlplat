import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("probability")

st.markdown("## المتغير العشوائي والتوقع والتباين")
formula(r"\mathbb E[X] = \sum_x x\,p(x)\ \text{or}\ \int x f(x)\,dx,\qquad \operatorname{Var}(X) = \mathbb E[(X-\mathbb E X)^2]",
        title="Expectation and variance",
        symbols={r"\mathbb E[X]": "المتوسط على المدى الطويل", r"\operatorname{Var}(X)": "متوسط مربع الابتعاد عن المتوسط"},
        intuition="المخاطرة المتوقعة في ML هي توقع الخسارة؛ وخطأ التعميم يتفكك إلى تحيز² + تباين + ضجيج.",
        example="نرد عادل: E[X] = 3.5، Var(X) = 35/12 ≈ 2.92.")
formula(r"\operatorname{Cov}(X,Y)=\mathbb E[(X-\mu_X)(Y-\mu_Y)],\qquad \rho=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}",
        title="Covariance and correlation",
        intuition="مصفوفة التغاير Σ أساس PCA وLDA/QDA والتوزيع الطبيعي متعدد المتغيرات.",
        example="ρ = 0.9 بين خاصيتين ⇒ معلوماتهما متداخلة ⇒ معاملات انحدار غير مستقرة.")

st.markdown("## الاحتمال الشرطي وقاعدة Bayes")
formula(r"P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}", title="Bayes' rule",
        symbols={"P(A)": "الاحتمال المسبق (Prior)", "P(B|A)": "الإمكان (Likelihood)", "P(A|B)": "اللاحق (Posterior)"},
        intuition="يحدّث معتقدنا بعد رؤية دليل. Naive Bayes وLDA/QDA تطبيقات مباشرة.")
st.markdown("### مختبر: اختبار طبي دقيق لمرض نادر")
c1, c2, c3 = st.columns(3)
prev = c1.slider("انتشار المرض P(D)", 0.001, 0.3, 0.01, 0.001, format="%.3f", key="pr_prev")
sens = c2.slider("الحساسية P(+|D)", 0.5, 1.0, 0.95, 0.01, key="pr_sens")
spec = c3.slider("النوعية P(−|¬D)", 0.5, 1.0, 0.95, 0.01, key="pr_spec")
p_pos = sens * prev + (1 - spec) * (1 - prev)
ppv = sens * prev / p_pos
N = 10_000
tp, fp = sens * prev * N, (1 - spec) * (1 - prev) * N
fig = go.Figure(go.Bar(x=["True positives", "False positives"], y=[tp, fp], marker_color=[PALETTE["teal"], PALETTE["coral"]],
                       text=[f"{tp:.0f}", f"{fp:.0f}"], textposition="auto"))
fig.update_layout(title=f"Out of {N:,} people tested — P(disease | positive) = {ppv:.1%}", height=320)
c1, c2 = st.columns([1.5, 1])
with c1:
    plot(fig)
with c2:
    st.metric("P(D | +)", f"{ppv:.1%}")
    st.markdown("مع انتشار منخفض، معظم النتائج الموجبة **كاذبة** رغم دقة الاختبار. هذا نفسه سبب انخفاض Precision في "
                "كشف الاحتيال.")

st.markdown("## الإمكان (Likelihood) وتقدير الإمكان الأعظم")
formula(r"\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n p(y_i\mid x_i;\theta) = \arg\min_\theta -\sum_i\log p(y_i\mid x_i;\theta)",
        title="Maximum likelihood = minimum negative log-likelihood",
        intuition="أغلب دوال الخسارة سالب لوغاريتم إمكان: MSE ⇔ ضجيج طبيعي، Log loss ⇔ Bernoulli، Poisson deviance ⇔ Poisson.",
        example="y ~ N(f(x), σ²) ⇒ −log p = (y − f(x))²/(2σ²) + const ⇒ تعظيم الإمكان = تقليل MSE.")

st.markdown("### مختبر الإمكان: قدّر احتمال عملة")
c1, c2 = st.columns(2)
n = c1.slider("عدد الرميات", 5, 500, 30, 5, key="pr_n")
true_p = c2.slider("الاحتمال الحقيقي", 0.05, 0.95, 0.7, 0.05, key="pr_tp")
heads = int(np.random.default_rng(0).binomial(n, true_p))
grid = np.linspace(0.001, 0.999, 400)
loglik = heads * np.log(grid) + (n - heads) * np.log(1 - grid)
fig = go.Figure(go.Scatter(x=grid, y=loglik - loglik.max(), line=dict(color=PALETTE["purple"], width=3), name="log-likelihood"))
fig.add_vline(x=heads / n, line=dict(color=PALETTE["coral"], dash="dash"), annotation_text=f"MLE = {heads / n:.3f}")
fig.add_vline(x=true_p, line=dict(color=PALETTE["teal"], dash="dot"), annotation_text="true p", annotation_position="bottom right")
fig.update_layout(title=f"{heads} heads out of {n}: log-likelihood (shifted to max 0)", xaxis_title="p",
                  yaxis=dict(range=[-10, 0.5]), height=340)
plot(fig)
intuition("مع زيادة n يصبح منحنى الإمكان أضيق: عدم يقين أقل. انحناء المنحنى عند القمة هو معلومات Fisher.")

st.markdown("## توزيعات ستقابلها في ML")
dist = st.segmented_control("التوزيع", ["Normal", "Bernoulli/Binomial", "Poisson", "Exponential", "Uniform"],
                            default="Normal", key="pr_dist", required=True)
xs = np.linspace(-4, 4, 300)
if dist == "Normal":
    fig = go.Figure(go.Scatter(x=xs, y=stats.norm.pdf(xs), fill="tozeroy", line=dict(color=PALETTE["sky"])))
    note_txt = "ضجيج الانحدار، LDA/QDA، GMM، والتقريب الطبيعي للمقدِّرات (أساس فترات الثقة في DML)."
elif dist == "Bernoulli/Binomial":
    k = np.arange(0, 21)
    fig = go.Figure(go.Bar(x=k, y=stats.binom.pmf(k, 20, 0.3), marker_color=PALETTE["coral"]))
    note_txt = "التصنيف الثنائي؛ Log loss هو سالب لوغاريتم إمكان Bernoulli."
elif dist == "Poisson":
    k = np.arange(0, 21)
    fig = go.Figure(go.Bar(x=k, y=stats.poisson.pmf(k, 4), marker_color=PALETTE["purple"]))
    note_txt = "بيانات العدّ (مطالبات، زيارات)؛ PoissonRegressor وPoisson deviance."
elif dist == "Exponential":
    x = np.linspace(0, 6, 300)
    fig = go.Figure(go.Scatter(x=x, y=stats.expon.pdf(x), fill="tozeroy", line=dict(color=PALETTE["teal"])))
    note_txt = "أزمنة الانتظار؛ أساس Survival analysis."
else:
    x = np.linspace(-0.5, 1.5, 300)
    fig = go.Figure(go.Scatter(x=x, y=stats.uniform.pdf(x), fill="tozeroy", line=dict(color=PALETTE["amber"])))
    note_txt = "Random search في HPO، والتهيئة العشوائية."
fig.update_layout(height=280, showlegend=False, title=dist)
plot(fig)
st.caption(note_txt)

if at_least("advanced"):
    st.markdown("## متقدم: الاستقلال الشرطي والتوقع الشرطي")
    comparison_table([
        {"المفهوم": "X ⫫ Y | Z", "المعنى": "بمعلومية Z لا يضيف X معلومات عن Y", "أين؟": "Naive Bayes، DAGs، عدم الإرباك"},
        {"المفهوم": "E[Y | X]", "المعنى": "أفضل متنبئ تحت MSE", "أين؟": "هدف الانحدار؛ دوال الإزعاج في DML"},
        {"المفهوم": "Law of total expectation", "المعنى": "E[Y] = E[E[Y|X]]", "أين؟": "تعريف ATE: E[τ(X)]"},
    ])
if at_least("research"):
    researcher_note(["دوال الإزعاج في DML هي توقعات شرطية: ℓ₀(X) = E[Y|X] وm₀(X) = E[D|X].",
                     "Proper scoring rules (Log loss، Brier) تُعظَّم بالاحتمالات الحقيقية — أساس المعايرة."])
mistakes(["تجاهل الانتشار عند تفسير النتائج الموجبة (Base-rate fallacy).", "الخلط بين P(A|B) وP(B|A).",
          "الخلط بين الارتباط الصفري والاستقلال."])
page_footer("probability",
            takeaways=["Bayes يربط الانتشار والحساسية بـPrecision.", "معظم الخسائر = سالب لوغاريتم إمكان.",
                       "E[Y|X] هو الهدف الذي يقرّبه كل نموذج انحدار."])
