import numpy as np
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile
from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.diagrams import mermaid
from components.formulas import formula
from components.parameter_lab import parameter_playground
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import toy_2d
from utils.plotting import lines, plot

page_header("dl_bridge")
algorithm_profile("mlp")

st.markdown("## من Perceptron إلى MLP")
formula(r"h = \phi(W_1x + b_1),\qquad \hat y = \sigma(W_2h + b_2)", title="One hidden layer (forward pass)",
        symbols={"W, b": "أوزان وانحيازات", r"\phi": "دالة تنشيط غير خطية (ReLU، tanh)", "h": "تمثيل مخفي"},
        intuition="كل طبقة = نموذج خطي + لاخطية. بدون التنشيط، طبقات كثيرة = نموذج خطي واحد. مع التنشيط = مقرّب عام للدوال.")
mermaid("""
flowchart LR
  x1((x1)) --> h1((h1)) & h2((h2)) & h3((h3))
  x2((x2)) --> h1 & h2 & h3
  h1 & h2 & h3 --> y((ŷ))
""")
comparison_table([
    {"المفهوم": "Activation", "المعنى": "ReLU = max(0, z)، tanh، sigmoid"},
    {"المفهوم": "Forward pass", "المعنى": "حساب ŷ طبقة بطبقة"},
    {"المفهوم": "Backpropagation", "المعنى": "قاعدة السلسلة من الخسارة إلى كل وزن"},
    {"المفهوم": "Epoch", "المعنى": "مرور كامل على بيانات التدريب"},
    {"المفهوم": "Batch", "المعنى": "دفعة الأمثلة لكل تحديث (Mini-batch SGD/Adam)"},
])

st.markdown("## تمرير أمامي يدوي")
c1, c2 = st.columns(2)
x = np.array([c1.slider("x1", -2.0, 2.0, 1.0, 0.1, key="dl_x1"), c2.slider("x2", -2.0, 2.0, -0.5, 0.1, key="dl_x2")])
W1 = np.array([[1.0, -1.0], [0.5, 1.5], [-1.2, 0.8]])
b1 = np.array([0.0, -0.5, 0.2])
W2 = np.array([1.5, -2.0, 1.0])
z1 = W1 @ x + b1
h = np.maximum(0, z1)
z2 = W2 @ h + 0.1
st.code(f"z1 = W1·x + b1 = {np.round(z1, 3)}\nh  = ReLU(z1)   = {np.round(h, 3)}\nz2 = W2·h + b2  = {z2:.3f}\nŷ  = σ(z2)      = {1 / (1 + np.exp(-z2)):.3f}",
        language="text")
formula(r"\frac{\partial\mathcal L}{\partial W_1} = \underbrace{(\hat y - y)}_{\partial\mathcal L/\partial z_2}\,W_2^\top \odot \phi'(z_1)\ x^\top",
        title="Backpropagation = chain rule", intuition="الخطأ عند المخرج يُرسل للخلف عبر الأوزان، ويُضرب بمشتقة التنشيط في كل طبقة.")

st.markdown("## MLP Lab")
c1, c2, c3 = st.columns(3)
kind = c1.segmented_control("البيانات", ["moons", "circles", "xor"], default="circles", key="dl_kind", required=True)
units = c2.select_slider("وحدات مخفية", [1, 2, 4, 8, 32], value=8, key="dl_units")
act = c3.segmented_control("التنشيط", ["relu", "tanh", "identity"], default="relu", key="dl_act", required=True)
X, y = toy_2d(kind, n=300, noise=0.2)


@st.cache_resource(show_spinner="يدرّب MLP…", max_entries=32)
def _fit(kind, units, act):
    m = make_pipeline(StandardScaler(), MLPClassifier((units,), activation=act, max_iter=2000, random_state=0))
    return m.fit(X, y)


model = _fit(kind, units, act)
boundary_chart(model, X, y, title=f"MLP ({units} hidden units, {act}) · train accuracy {model.score(X, y):.3f}")
if act == "identity":
    st.info("مع تنشيط identity تصبح الشبكة خطية مهما كان عدد الوحدات — لا تستطيع فصل الدوائر.")
plot(lines(np.arange(len(model[-1].loss_curve_)), {"training loss": model[-1].loss_curve_}, title="Loss per epoch (Adam)",
           xaxis="epoch", yaxis="log loss"), height=280)
st.markdown("### ساحة المعاملات للشبكة")
parameter_playground("mlp", key="dl_pg", dataset="moons")
intuition("scikit-learn MLP مناسب للتعليم وللبيانات الجدولية الصغيرة؛ للصور والنصوص والتسلسلات نحتاج مكتبات تعلّم عميق وGPU.")
st.info("**التعلّم العميق سيكون في منصة مستقلة لاحقًا** (CNNs، RNNs/Transformers، Embeddings، التدريب على GPU). هذه الوحدة جسر فقط.",
        icon=":material/neurology:")
if at_least("research"):
    researcher_note(["Rumelhart, Hinton & Williams (1986) قدّموا Backpropagation للشبكات متعددة الطبقات.",
                     "الشبكات العميقة ليست دائمًا الأفضل للبيانات الجدولية؛ التعزيز الشجري يبقى منافسًا قويًا."])
    st.markdown(cite("rumelhart1986", "kingma2015"))
mistakes(["MLP دون قياس.", "شبكة كبيرة على بيانات صغيرة دون تنظيم (alpha).", "تجاهل ConvergenceWarning."])
page_footer("dl_bridge",
            takeaways=["MLP = طبقات خطية + تنشيط غير خطي.", "Backpropagation = قاعدة السلسلة.", "التعمق الحقيقي في منصة Deep Learning لاحقًا."])
