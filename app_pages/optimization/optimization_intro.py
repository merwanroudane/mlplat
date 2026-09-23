import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.optimization import make_linear_data, mse_loss_surface, ols_closed_form
from utils.plotting import contour_surface, plot

page_header("optimization_intro")

st.markdown("## التعلّم = تحسين")
formula(r"\hat{\boldsymbol\theta} = \arg\min_{\boldsymbol\theta}\ J(\boldsymbol\theta) = "
        r"\frac1n\sum_{i=1}^n \ell\big(y_i, f(x_i;\boldsymbol\theta)\big) + \lambda\,\Omega(\boldsymbol\theta)",
        title="Training objective",
        symbols={r"\boldsymbol\theta": "المعاملات المتعلَّمة", "J": "دالة الهدف", r"\Omega": "عقوبة التنظيم"},
        intuition="كل خوارزمية تعلّم تقريبًا = (نموذج f) + (خسارة ℓ) + (تنظيم Ω) + (طريقة لإيجاد θ̂).")

st.markdown("## الحل المغلق مقابل التكراري")
comparison_table([
    {"النوع": "Closed form", "مثال": "OLS: β̂ = (XᵀX)⁻¹Xᵀy؛ Ridge؛ LDA", "المزايا": "دقيق، بلا معدل تعلّم",
     "العيوب": "O(p³)؛ لا يوجد لمعظم النماذج"},
    {"النوع": "Iterative (first order)", "مثال": "GD، SGD، Adam", "المزايا": "رخيص لكل خطوة، يتوسع",
     "العيوب": "يحتاج معدل تعلّم وتكرارات"},
    {"النوع": "Iterative (second order)", "مثال": "Newton، L-BFGS، newton-cholesky", "المزايا": "خطوات قليلة",
     "العيوب": "الهيسيان مكلف"},
    {"النوع": "Coordinate descent", "مثال": "Lasso، Elastic Net", "المزايا": "ممتاز مع العقوبات غير الملساء",
     "العيوب": "تسلسلي"},
    {"النوع": "Greedy / combinatorial", "مثال": "أشجار القرار (CART)", "المزايا": "سريع", "العيوب": "ليس الأمثل عالميًا"},
])

st.markdown("## سطح الخسارة لانحدار خطي بمعاملين")
st.caption("MSE(w₁, w₂) للنموذج ŷ = w₁x + w₂. الحد الأدنى = حل المعادلات الطبيعية.")
noise = st.slider("الضجيج", 0.1, 2.0, 0.5, 0.1, key="oi_noise")
X, y = make_linear_data(200, noise=noise, seed=0)
beta = ols_closed_form(X, y)
f = mse_loss_surface(X, y)
fig = contour_surface(f, (beta[0] - 3, beta[0] + 3), (beta[1] - 3, beta[1] + 3))
fig.add_trace(go.Scatter(x=[beta[0]], y=[beta[1]], mode="markers+text", text=["closed-form minimum"],
                         textposition="top center", marker=dict(size=14, symbol="star", color=PALETTE["coral"])))
fig.update_layout(title=f"MSE surface · β̂ = ({beta[0]:.3f}, {beta[1]:.3f}) · true (2, −1)")
plot(fig)
intuition("خطوط الكنتور قطوع ناقصة: MSE للانحدار الخطي دالة تربيعية محدبة ⇒ حد أدنى واحد عالمي. الاستطالة تعكس "
          "رقم التكييف (Condition number) لـXᵀX.")

st.markdown("## التحدّب")
definition("Convex function", "$f(\\lambda a + (1-\\lambda) b) \\le \\lambda f(a) + (1-\\lambda) f(b)$ لكل $\\lambda\\in[0,1]$: "
           "الوتر فوق المنحنى. للدوال الملساء: الهيسيان موجب شبه تحديد.")
c1, c2 = st.columns(2)
xs = np.linspace(-2.2, 2.2, 300)
with c1:
    fig = go.Figure(go.Scatter(x=xs, y=xs ** 2 + 0.3 * xs, line=dict(color=PALETTE["sky"], width=3)))
    fig.add_trace(go.Scatter(x=[-1.8, 1.5], y=[(-1.8) ** 2 - 0.54, 1.5 ** 2 + 0.45], mode="lines+markers",
                             line=dict(color=PALETTE["coral"], dash="dash")))
    fig.update_layout(title="Convex: one global minimum", showlegend=False, height=300)
    plot(fig)
with c2:
    fig = go.Figure(go.Scatter(x=xs, y=(xs ** 2 - 1) ** 2 + 0.3 * xs, line=dict(color=PALETTE["purple"], width=3)))
    fig.update_layout(title="Non-convex: local minima depend on the start", showlegend=False, height=300)
    plot(fig)
comparison_table([
    {"النموذج": "Linear / Ridge / Lasso / Logistic / SVM", "محدب؟": "نعم", "أثر ذلك": "أي Solver يتقارب إلى الحل نفسه"},
    {"النموذج": "Neural networks", "محدب؟": "لا", "أثر ذلك": "البداية والعشوائية تغيّر الحل"},
    {"النموذج": "K-Means", "محدب؟": "لا", "أثر ذلك": "n_init: عدة بدايات"},
    {"النموذج": "GMM (likelihood)", "محدب؟": "لا", "أثر ذلك": "EM يصل لحد محلي"},
    {"النموذج": "Decision trees", "محدب؟": "مسألة توافقية", "أثر ذلك": "بحث جشع، لا ضمان أمثلية"},
])
page_link("gradient_descent", "التالي: مختبر Gradient Descent", ":material/south_east:")

if at_least("advanced"):
    st.markdown("## متقدم: شروط التقارب")
    st.markdown("إذا كانت $J$ محدبة ومشتقتها **L-Lipschitz** (L-smooth)، فإن GD بخطوة $\\eta \\le 1/L$ يحقق "
                "$J(\\theta_k) - J^* \\le \\frac{\\|\\theta_0-\\theta^*\\|^2}{2\\eta k}$ = O(1/k). وإذا كانت أيضًا "
                "**μ-strongly convex** فالتقارب خطي (هندسي) بمعدل $(1-\\mu/L)^k$؛ النسبة $L/\\mu$ = رقم التكييف.")
if at_least("research"):
    researcher_note(["لأن مسائل GLM محدبة، فاختلاف الـSolver لا يغيّر الحل (بعد التقارب) لكنه يغيّر الزمن؛ عدم التقارب "
                     "(ConvergenceWarning) يغيّر النتائج فعلًا — أبلغ عنه.",
                     "في DML، معادلة الدرجة لـPLR خطية في θ ⇒ حل مغلق؛ في نماذج أخرى (IV-type غير خطي) نحتاج حلًا عدديًا."])
mistakes(["توقع أن الشبكات العصبية تعطي النتيجة نفسها في كل تشغيل.", "تجاهل ConvergenceWarning.",
          "الظن أن الحل المغلق دائمًا أفضل (O(p³) مع p كبير)."])
page_footer("optimization_intro",
            takeaways=["التدريب = تقليل دالة هدف.", "الحل المغلق نادر؛ معظم النماذج تكرارية.",
                       "التحدب يضمن حدًا أدنى عالميًا ويجعل الـSolver تفصيلًا حسابيًا."])
