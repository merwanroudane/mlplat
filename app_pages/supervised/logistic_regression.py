import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.decision_boundary import boundary_chart
from components.formulas import formula
from components.parameter_lab import parameter_playground
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d, xy
from utils.models import LogisticRegressionGD, sigmoid
from utils.plotting import lines, plot

page_header("logistic_regression")
algorithm_profile("logistic")

st.markdown("## من الاحتمال إلى Odds إلى Log-odds")
formula(r"\log\frac{p(x)}{1-p(x)} = \beta_0 + \beta^\top x\quad\Longleftrightarrow\quad p(x) = \sigma(\beta_0+\beta^\top x) = \frac{1}{1+e^{-(\beta_0+\beta^\top x)}}",
        title="Logistic model",
        symbols={"p(x)": "P(y = 1 | x)", r"\frac{p}{1-p}": "Odds: احتمال الحدوث مقسومًا على عدمه", r"\sigma": "Sigmoid"},
        intuition="نمذجة الاحتمال مباشرة خطيًا قد تعطي قيمًا خارج [0, 1]؛ نمذجة log-odds خطيًا ثم تطبيق Sigmoid تحل ذلك.",
        example="β₀ = −2، β₁ = 0.8، x = 3 ⇒ z = 0.4 ⇒ p = σ(0.4) = 0.599؛ زيادة x بوحدة تضرب Odds بـe^0.8 = 2.23.")
c1, c2 = st.columns([1.3, 1])
with c1:
    z = np.linspace(-7, 7, 300)
    fig = go.Figure(go.Scatter(x=z, y=sigmoid(z), line=dict(color=PALETTE["purple"], width=3), name="σ(z)"))
    fig.add_hline(y=0.5, line=dict(dash="dot", color=PALETTE["muted"]))
    fig.update_layout(title="Sigmoid maps log-odds z to a probability", xaxis_title="z = β₀ + βᵀx", yaxis_title="p", height=300)
    plot(fig)
with c2:
    p = st.slider("احتمال p", 0.01, 0.99, 0.8, 0.01, key="lg_p")
    st.metric("Odds = p/(1−p)", f"{p / (1 - p):.3f}")
    st.metric("Log-odds", f"{np.log(p / (1 - p)):.3f}")

st.markdown("## الخسارة والتدرج")
formula(r"\mathcal L(\beta) = -\frac1n\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big],\qquad \nabla\mathcal L = \frac1n X^\top(p-y)",
        title="Log loss (binary cross-entropy) and its gradient",
        intuition="الخسارة محدبة ⇒ حل وحيد. التدرج أنيق: الخطأ (p − y) مضروبًا في الخصائص.",
        example="y = 1 وp = 0.9 ⇒ الخسارة 0.105؛ y = 1 وp = 0.1 ⇒ 2.30: الثقة الخاطئة تُعاقَب بشدة.")

st.markdown("## مختبر حدّ القرار اللوجستي")
c1, c2, c3 = st.columns(3)
kind = c1.segmented_control("البيانات", ["linear", "moons", "circles"], default="linear", key="lg_kind", required=True)
C = c2.select_slider("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0, key="lg_C")
poly = c3.toggle("أضف خصائص تربيعية (x², xy)", value=False, key="lg_poly")
Xb, yb = toy_2d(kind, n=300, noise=0.25)
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402
steps = [PolynomialFeatures(2, include_bias=False)] if poly else []
model = make_pipeline(*steps, StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Xb, yb)
boundary_chart(model, Xb, yb, title=f"Logistic regression (C={C}{', quadratic features' if poly else ''}) — "
                                    f"train accuracy {model.score(Xb, yb):.3f}")
coef = model[-1].coef_[0]
st.caption(f"‖β‖ = {np.linalg.norm(coef):.2f} — C صغير ينكمش المعاملات ⇒ احتمالات أقرب إلى 0.5 وحدود «أنعم». "
           "النموذج خطي في الخصائص؛ الخصائص التربيعية تجعل الحد قطعًا مخروطيًا.")

st.markdown("## المعاملات الفائقة والتغيير المهم في scikit-learn 1.8+")
hyperparameter_table("LogisticRegression")
comparison_table([
    {"العقوبة المطلوبة": "L2 (افتراضي)", "الطريقة الحديثة": "l1_ratio=0.0 (افتراضي)", "بدلًا من": "penalty='l2'",
     "Solvers": "lbfgs, newton-cg, newton-cholesky, sag, saga, liblinear"},
    {"العقوبة المطلوبة": "L1", "الطريقة الحديثة": "l1_ratio=1.0 + solver='saga' أو 'liblinear'", "بدلًا من": "penalty='l1'",
     "Solvers": "saga، liblinear (ثنائي)"},
    {"العقوبة المطلوبة": "Elastic Net", "الطريقة الحديثة": "0 < l1_ratio < 1 + solver='saga'", "بدلًا من": "penalty='elasticnet'",
     "Solvers": "saga"},
    {"العقوبة المطلوبة": "بلا عقوبة", "الطريقة الحديثة": "C=np.inf", "بدلًا من": "penalty=None",
     "Solvers": "lbfgs, newton-cg, newton-cholesky, sag, saga"},
])
st.caption("متحقَّق منه مباشرة من scikit-learn 1.9.1: تمرير penalty يطلق تحذيرًا نصه «'penalty' was deprecated in version 1.8 and "
           "will be removed in 1.10». "
           "و`n_jobs` في LogisticRegression مهجور أيضًا (يُزال في 1.10).")

st.markdown("## من الصفر مقابل scikit-learn مقابل statsmodels")
Xc, yc = xy("classification")
Xcs = StandardScaler().fit_transform(Xc)


def _code(p):
    return ("# from scratch (utils/models.py): minimises C·Σ logloss + ½‖w‖²  (same objective as sklearn)\n"
            f"scratch = LogisticRegressionGD(C={p['C']}, lr=0.5, n_iter=3000).fit(X, y)\n"
            f"sk = LogisticRegression(C={p['C']}).fit(X, y)          # lbfgs, l1_ratio=0 (L2)\n"
            "sm_fit = sm.Logit(y, sm.add_constant(X)).fit()        # unpenalised MLE with standard errors")


def _run(C):
    scratch = LogisticRegressionGD(C=C, lr=0.5, n_iter=3000).fit(Xcs, yc.to_numpy())
    sk = LogisticRegression(C=C, max_iter=5000).fit(Xcs, yc)
    smf = sm.Logit(yc.to_numpy(), sm.add_constant(Xcs)).fit(disp=0, method="newton", maxiter=200)
    tab = pd.DataFrame({"scratch GD": np.r_[scratch.intercept_, scratch.coef_], "sklearn": np.r_[sk.intercept_, sk.coef_[0]],
                        "statsmodels (C=∞)": smf.params, "odds ratio (sm)": np.exp(smf.params), "p-value (sm)": smf.pvalues},
                       index=["intercept"] + [f"f{i}" for i in range(Xcs.shape[1])])
    curve = lines(np.arange(len(scratch.loss_curve_)), {"scratch log loss": scratch.loss_curve_},
                  title="Scratch GD training curve", xaxis="iteration", yaxis="log loss")
    return [tab.round(4), curve]


code_lab("logit_code", "Logistic regression three ways", _code, _run,
         lambda: {"C": st.select_slider("C", [0.01, 0.1, 1.0, 10.0], value=1.0, key="lgc_C")},
         explanation="scratch وsklearn يحلان الهدف المنظَّم نفسه (C·Σloss + ½‖w‖²)، فيتطابقان تقريبًا؛ statsmodels بلا تنظيم "
                     "فيقترب منهما حين C كبير. statsmodels يعطي أخطاء معيارية وp-values للاستدلال.")

st.markdown("## ساحة المعاملات")
parameter_playground("logistic", key="lg_pg", dataset="moons")
why("قس الخصائص قبل LogisticRegression المنظَّم.",
    "العقوبة ½‖w‖² تعامل كل المعاملات بالتساوي؛ خاصية بوحدات كبيرة تحتاج معاملًا صغيرًا فلا تُعاقَب، والعكس.")
intuition("الفصل التام (خط يفصل الفئتين كليًا) يدفع المعاملات إلى ∞ في MLE غير المنظَّم؛ التنظيم يبقيها منتهية. "
          "لذلك C=∞ مع بيانات قابلة للفصل يطلق ConvergenceWarning.")

if at_least("advanced"):
    st.markdown("## متقدم: التعقيد والتشخيص")
    st.markdown("- **التعقيد:** lbfgs O(np) لكل تكرار؛ newton-cholesky O(np² + p³) لكل تكرار.\n"
                "- **التشخيص:** Calibration curve، ConvergenceWarning، معاملات ضخمة (فصل تام)، التعدد الخطي.\n"
                "- **Class weight:** `class_weight='balanced'` يرفع وزن الفئة النادرة لكنه يُفسد معايرة الاحتمالات؛ "
                "البديل الأنظف: درّب عاديًا واضبط العتبة.")
if at_least("research"):
    researcher_note(["Odds ratios غير قابلة للطي (Non-collapsible): تتغير عند إضافة متغير غير مُربك؛ احذر مقارنتها بين نماذج.",
                     "للاستدلال استخدم statsmodels (MLE غير منظَّم) أو إجراءات Post-selection؛ معاملات sklearn المنظَّمة متحيزة.",
                     "في DML-IRM نموذج الميل m(X) غالبًا Logistic منظَّم؛ جودة المعايرة تؤثر في الأوزان 1/m."])
    st.markdown(cite("esl", "islp"))

template_checklist({3: "صيغة log-odds", 5: "مختبر حد القرار", 6: "Log loss", 7: "التدرج + الحلول", 8: "coef_ / intercept_",
                    9: "جدول المعاملات الفائقة + جدول penalty/l1_ratio", 20: "Odds ratios (statsmodels)",
                    21: "LogisticRegressionGD", 22: "LogisticRegression", 23: "مختبر الحد", 24: "ساحة المعاملات"})
mistakes(["استخدام penalty المهجور في كود جديد.", "نسيان القياس مع التنظيم.", "تفسير المعاملات المنظَّمة كتقديرات غير متحيزة.",
          "اعتبار 0.5 عتبة مقدسة."])
page_footer("logistic_regression",
            takeaways=["Logistic = نموذج خطي لـlog-odds + Sigmoid.", "Log loss محدبة؛ C يتحكم في التنظيم.",
                       "منذ 1.8: استخدم l1_ratio وC بدل penalty."])
