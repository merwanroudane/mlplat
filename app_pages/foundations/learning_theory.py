import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import lines, plot

page_header("learning_theory")

st.markdown("## لماذا يمكن التعميم أصلًا؟")
intuition("إن كان فضاء الفرضيات صغيرًا مقارنة بعدد الأمثلة، فمن غير المرجح أن تبدو فرضية سيئة جيدة على التدريب صدفةً. "
          "كلما كبر الفضاء (سعة أكبر) احتجنا أمثلة أكثر للثقة بأن الأداء على التدريب يعكس الحقيقة.")
definition("Hypothesis space $\\mathcal H$", "مجموعة الدوال التي تستطيع الخوارزمية اختيارها (كل الخطوط، كل الأشجار بعمق ≤ 3...).")

st.markdown("## حد لفضاء منتهٍ (Hoeffding + Union bound)")
formula(r"P\Big(\exists h\in\mathcal H:\ |R(h)-\hat R_n(h)|>\epsilon\Big) \le 2|\mathcal H|\,e^{-2n\epsilon^2}",
        title="Uniform convergence for a finite hypothesis class (0-1 loss)",
        symbols={r"|\mathcal H|": "عدد الفرضيات", "n": "حجم العينة", r"\epsilon": "الفجوة المسموحة بين التدريب والحقيقة"},
        intuition="الفجوة تتقلص بمعدل 1/√n وتنمو مع log|H| فقط.",
        example="|H| = 10⁶، n = 10,000، الثقة 95% ⇒ ε ≈ √((log(2·10⁶) + log 20)/(2n)) ≈ 0.029.")
c1, c2 = st.columns(2)
logH = c1.slider("log₁₀|H|", 1, 30, 6, key="lt_logH")
delta = c2.slider("δ (احتمال الفشل)", 0.01, 0.2, 0.05, 0.01, key="lt_delta")
ns = np.logspace(2, 6, 60)
eps = np.sqrt((logH * np.log(10) + np.log(2 / delta)) / (2 * ns))
fig = lines(ns, {"ε bound": eps}, title=f"Generalisation gap bound vs n (|H| = 10^{logH}, δ = {delta})", xaxis="n",
            yaxis="ε", log_x=True)
plot(fig, height=320)

st.markdown("## VC dimension: قياس السعة لفضاء لا نهائي")
definition("Shattering / VC dimension", "يُمزِّق $\\mathcal H$ مجموعة نقاط إن استطاع تحقيق **كل** التسميات الممكنة $2^m$ "
           "عليها. بُعد VC هو أكبر m يمكن تمزيقه.")
comparison_table([
    {"الفضاء": "عتبات على خط (x > t)", "VC dimension": "1", "ملاحظة": "نقطتان: لا يستطيع (1, 0) بالترتيب"},
    {"الفضاء": "مصنفات خطية في ℝ²", "VC dimension": "3", "ملاحظة": "3 نقاط غير متسامتة تُمزَّق، 4 لا (XOR)"},
    {"الفضاء": "مصنفات خطية في ℝᵖ", "VC dimension": "p + 1", "ملاحظة": "السعة تنمو مع الأبعاد"},
    {"الفضاء": "kNN مع k = 1", "VC dimension": "∞", "ملاحظة": "يحفظ أي تسمية"},
    {"الفضاء": "sin(ωx) على خط", "VC dimension": "∞", "ملاحظة": "معامل واحد وسعة لا نهائية: عدد المعاملات ≠ السعة"},
])
st.markdown("### جرّب: هل يمزّق مصنف خطي 4 نقاط؟")
labels = st.pills("تسمية النقاط الأربع (A, B, C, D)", ["A", "B", "C", "D"], selection_mode="multi", default=["A", "D"],
                  key="lt_labels")
pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([1 if n in (labels or []) else 0 for n in "ABCD"])
from sklearn.svm import SVC  # noqa: E402

separable = len(np.unique(y)) < 2 or SVC(kernel="linear", C=1e6).fit(pts, y).score(pts, y) == 1.0
fig = go.Figure()
for i, name in enumerate("ABCD"):
    fig.add_trace(go.Scatter(x=[pts[i, 0]], y=[pts[i, 1]], mode="markers+text", text=[name], textposition="top center",
                             marker=dict(size=22, color=PALETTE["coral"] if y[i] else PALETTE["sky"],
                                         symbol="diamond" if y[i] else "circle"), showlegend=False))
fig.update_layout(height=280, xaxis=dict(range=[-0.5, 1.5]), yaxis=dict(range=[-0.5, 1.5], scaleanchor="x"),
                  title="Linearly separable ✓" if separable else "NOT linearly separable ✗ (this labelling defeats every line)")
plot(fig)
st.caption("اختر A وD فقط (XOR): لا يوجد خط يفصلهما. لذا بُعد VC للمصنفات الخطية في ℝ² هو 3 لا 4.")

formula(r"R(h) \le \hat R_n(h) + \sqrt{\frac{d\big(\log\frac{2n}{d}+1\big)+\log\frac4\delta}{n}}",
        title="VC generalisation bound (one common form; constants vary by textbook)",
        symbols={"d": "VC dimension"},
        intuition="الفجوة تنمو تقريبًا مع √(d/n): السعة الأعلى تحتاج n أكبر بالتناسب.")

st.markdown("## Structural risk minimization والهوامش")
st.markdown("بدل تقليل الخطأ التجريبي وحده، نقلل **الخطأ + عقوبة السعة**؛ هذا ما يفعله التنظيم عمليًا. في SVM، تعظيم الهامش "
            "يقيّد السعة الفعلية بصرف النظر عن عدد الأبعاد — لذلك تعمل SVM جيدًا في الأبعاد العالية.")

st.markdown("## No Free Lunch")
intuition("إذا ساوينا بين كل المسائل الممكنة، فلا توجد خوارزمية أفضل من غيرها في المتوسط (Wolpert, 1996). المعنى العملي: "
          "التفوق يأتي دائمًا من **افتراضات مسبقة** تناسب البنية الحقيقية للمسألة — نعومة، ندرة، تفاعلات. لذلك لا يعلن "
          "مساعد اختيار الخوارزمية «فائزًا مطلقًا».")

if at_least("advanced"):
    st.markdown("## متقدم: Rademacher complexity")
    st.latex(r"\mathfrak R_n(\mathcal H) = \mathbb E_{\sigma,S}\Big[\sup_{h\in\mathcal H}\frac1n\sum_i \sigma_i h(x_i)\Big]")
    st.markdown("تقيس قدرة الفضاء على ملاءمة ضجيج عشوائي ±1؛ تعطي حدودًا تعتمد على البيانات وأدق من VC غالبًا.")
if at_least("research"):
    researcher_note(["حدود VC فضفاضة جدًا للشبكات العميقة؛ فهم تعميمها موضوع بحث مفتوح (Implicit regularization, flat minima).",
                     "Cross-fitting في DML يتجنب شروط Donsker (قيود على تعقيد فضاء الإزعاج)؛ هذا بالضبط ما يسمح باستخدام "
                     "متعلمات مرنة جدًا كالغابات والتعزيز."])
    st.markdown(cite("uml", "wolpert1996", "esl"))
mistakes(["الخلط بين عدد المعاملات والسعة.", "قراءة الحدود النظرية كتنبؤ دقيق بالأداء.",
          "استنتاج أن خوارزمية «الأفضل دائمًا» من نتيجة مسابقة."])
page_footer("learning_theory",
            takeaways=["الفجوة بين التدريب والحقيقة تتقلص مع n وتنمو مع السعة.",
                       "VC dimension يقيس السعة بقدرة التمزيق.", "No Free Lunch: الأداء يأتي من افتراضات تناسب المسألة."])
