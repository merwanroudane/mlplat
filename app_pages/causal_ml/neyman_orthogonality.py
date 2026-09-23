import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("neyman_orthogonality")

st.markdown("## الحدس (مبتدئ)")
intuition("تخيل أنك تقيس أثر D على Y بعد «تنظيف» كليهما من X. لو كان تنظيفك غير مثالي قليلًا، فالدرجة المتعامدة مصممة "
          "بحيث **لا يتحرك** التقدير تقريبًا — الخطأ الصغير في التنظيف يسبب خطأ أصغر بكثير (مربعه) في θ̂.")

st.markdown("## الصياغة (بحثي)")
formula(r"\mathbb E\big[\psi(W;\theta_0,\eta_0)\big] = 0", title="Moment condition",
        symbols={"W": "البيانات (Y, D, X)", r"\theta": "المعلمة المستهدفة", r"\eta": "دوال الإزعاج (ℓ, m, g...)",
                 r"\psi": "دالة الدرجة (Score)"},
        intuition="θ₀ هي القيمة التي تجعل متوسط الدرجة صفرًا عند الإزعاج الحقيقي.")
formula(r"\partial_\eta\,\mathbb E\big[\psi(W;\theta_0,\eta_0 + r(\eta-\eta_0))\big]\Big|_{r=0} = 0\quad\forall\eta",
        title="Neyman orthogonality (Gateaux derivative)",
        intuition="المشتقة الاتجاهية لشرط العزم بالنسبة للإزعاج تساوي صفرًا عند القيم الحقيقية ⇒ أخطاء الإزعاج الصغيرة لا تؤثر "
                  "من الرتبة الأولى.")
comparison_table([
    {"الدرجة": "Non-orthogonal (plug-in)", "ψ": "(Y − Dθ − g(X))·D", "المشتقة في اتجاه g": "−E[D·Δg(X)] ≠ 0", "الأثر": "خطأ ĝ ينتقل خطيًا إلى θ̂"},
    {"الدرجة": "Orthogonal (partialling out)", "ψ": "(Y − ℓ(X) − θ(D − m(X)))·(D − m(X))", "المشتقة في اتجاه (ℓ, m)": "0",
     "الأثر": "خطأ من الرتبة الثانية: ‖ℓ̂ − ℓ‖·‖m̂ − m‖"},
])

st.markdown("## Orthogonality Sensitivity Lab")
st.caption("نحسب θ̂ على عينة كبيرة (لإزالة ضجيج العينة) بعد تشويه الإزعاج عمدًا بخطأ δ·h(X)، ونرسم التحيز كدالة لـδ.")
direction = st.segmented_control("اتجاه التشويه", ["في ℓ (أو g)", "في m", "في الاثنين"], default="في الاثنين", key="no_dir",
                                 required=True)
rng = np.random.default_rng(0)
n = 200_000
X = rng.normal(size=n)
m0 = np.tanh(X)
g0 = np.cos(X) + X
theta0 = 0.5
V = rng.normal(size=n)
D = m0 + V
Y = theta0 * D + g0 + rng.normal(size=n)
ell0 = theta0 * m0 + g0
h = np.sin(2 * X) + 0.5 * X  # a fixed error direction
deltas = np.linspace(-0.6, 0.6, 25)
naive, ortho = [], []
for dlt in deltas:
    g_hat = g0 + (dlt * h if direction != "في m" else 0)
    ell_hat = ell0 + (dlt * h if direction != "في m" else 0)
    m_hat = m0 + (dlt * h if direction != "في ℓ (أو g)" else 0)
    naive.append(np.sum(D * (Y - g_hat)) / np.sum(D * D))
    v = D - m_hat
    ortho.append(np.sum(v * (Y - ell_hat)) / np.sum(v * v))
fig = go.Figure()
fig.add_trace(go.Scatter(x=deltas, y=np.array(naive) - theta0, name="non-orthogonal score", line=dict(color=PALETTE["coral"], width=3)))
fig.add_trace(go.Scatter(x=deltas, y=np.array(ortho) - theta0, name="orthogonal score", line=dict(color=PALETTE["teal"], width=3)))
fig.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dot"))
fig.update_layout(title="Bias of θ̂ as the nuisance error δ grows", xaxis_title="nuisance error size δ", yaxis_title="θ̂ − θ₀", height=400)
plot(fig)
st.markdown("الدرجة غير المتعامدة: تحيز **خطي** في δ (ميل غير صفري عند 0). المتعامدة: منحنى **مسطح** عند δ = 0 — التحيز ∝ δ² "
            "(يظهر فقط حين يُشوَّه ℓ وm معًا، وهو حاصل ضرب خطأيهما).")

st.markdown("## التحقق التحليلي لـPLR")
st.latex(r"\partial_r\,\mathbb E\big[(Y-\ell_0-r\Delta_\ell-\theta_0(D-m_0-r\Delta_m))(D-m_0-r\Delta_m)\big]_{r=0}"
         r"= -\mathbb E[\Delta_\ell V] + \theta_0\mathbb E[\Delta_m V] - \mathbb E[\zeta\,\Delta_m] = 0")
st.caption("كل حد يساوي صفرًا لأن V وζ متوسطهما الشرطي صفر بمعلومية X، وΔ دوال في X فقط.")

if at_least("advanced"):
    st.markdown("## درجات متعامدة لنماذج أخرى")
    comparison_table([
        {"النموذج": "PLR (IV-type)", "ψ": "(Y − Dθ − g(X))(D − m(X))"},
        {"النموذج": "PLIV", "ψ": "(Y − ℓ(X) − θ(D − r(X)))(Z − m(X))"},
        {"النموذج": "IRM (ATE, AIPW)", "ψ": "g(1,X) − g(0,X) + D(Y − g(1,X))/m(X) − (1−D)(Y − g(0,X))/(1 − m(X)) − θ"},
        {"النموذج": "IIVM (LATE)", "ψ": "نسبة درجتي AIPW للنتيجة والمعالجة على الأداة"},
    ])
if at_least("research"):
    researcher_note(["الدرجات المتعامدة ترتبط بـEfficient influence functions في النظرية شبه المعلمية.",
                     "التعامد يعالج Regularization bias؛ Cross-fitting يعالج Overfitting bias؛ نحتاج الاثنين."])
    st.markdown(cite("chernozhukov2018"))
page_link("cross_fitting", "التالي: Cross-Fitting", ":material/sync_alt:")
mistakes(["الظن أن أي درجة تعطي نفس النتيجة.", "استخدام درجة متعامدة بإزعاج داخل العينة."])
page_footer("neyman_orthogonality",
            takeaways=["التعامد: مشتقة شرط العزم بالنسبة للإزعاج = 0.", "يحول أخطاء الإزعاج إلى تحيز من الرتبة الثانية.",
                       "لذلك يكفي إزعاج بمعدل o(n^{-1/4})."])
