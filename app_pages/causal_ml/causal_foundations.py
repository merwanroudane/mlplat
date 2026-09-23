import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from components.callouts import causal_caution, definition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import bars, overlay_hist, plot

page_header("causal_foundations")

st.markdown("## المفردات")
comparison_table([
    {"المفهوم": "Treatment D", "المعنى": "ما نتدخل فيه (ثنائي، متعدد، مستمر)", "مثال": "برنامج تدريب"},
    {"المفهوم": "Outcome Y", "المعنى": "ما نريد معرفة تأثره", "مثال": "الأجر بعد سنة"},
    {"المفهوم": "Covariates X", "المعنى": "خصائص مُقاسة قبل المعالجة", "مثال": "العمر، التعليم، الخبرة"},
    {"المفهوم": "Confounder", "المعنى": "يؤثر في D وY معًا", "مثال": "الدافعية (إن قيست)"},
    {"المفهوم": "Mediator", "المعنى": "على المسار D → M → Y", "مثال": "الشهادة المكتسبة"},
    {"المفهوم": "Collider", "المعنى": "يتأثر بـD وY معًا — لا تضبطه", "مثال": "«قُبل في وظيفة» حين يعتمد على التدريب والأجر"},
])

st.markdown("## النتائج الكامنة والواقع المضاد")
definition("Potential outcomes (Neyman–Rubin)", "لكل وحدة نتيجتان: $Y_i(1)$ لو عولجت و$Y_i(0)$ لو لم تعالَج. الأثر الفردي "
           "$\\tau_i = Y_i(1) - Y_i(0)$. نلاحظ دائمًا واحدة فقط: $Y_i = D_iY_i(1) + (1-D_i)Y_i(0)$ — **المسألة الأساسية للاستدلال السببي**.")
rng = np.random.default_rng(4)
tab = pd.DataFrame({"unit": range(1, 7), "D": [1, 0, 1, 0, 1, 0]})
tab["Y(0)"] = np.round(rng.normal(10, 2, 6), 1)
tab["Y(1)"] = np.round(tab["Y(0)"] + rng.normal(2, 1, 6), 1)
show = st.toggle("اكشف الواقع المضاد (ما لا نراه أبدًا)", value=False, key="cf_reveal")
disp = tab.copy()
disp["τ = Y(1) − Y(0)"] = (tab["Y(1)"] - tab["Y(0)"]).round(1)
if not show:
    disp.loc[disp["D"] == 1, "Y(0)"] = np.nan
    disp.loc[disp["D"] == 0, "Y(1)"] = np.nan
    disp["τ = Y(1) − Y(0)"] = np.nan
st.dataframe(disp, hide_index=True, width="stretch")
st.caption("الخلايا الفارغة (NaN) هي الوقائع المضادة. الاستدلال السببي = تقدير متوسطات ما لا نراه باستخدام افتراضات.")

formula(r"\text{ATE} = \mathbb E[Y(1)-Y(0)],\quad \text{ATT} = \mathbb E[Y(1)-Y(0)\mid D=1],\quad \tau(x)=\text{CATE}(x) = \mathbb E[Y(1)-Y(0)\mid X=x]",
        title="Target parameters",
        intuition="ATE: لو عالجنا الجميع مقابل لا أحد. ATT: الأثر على من عولجوا فعلًا. CATE: الأثر لمن خصائصه x.",
        example="برنامج تدريب اختياري: ATT (أثره على المشاركين) قد يختلف عن ATE (أثره لو فُرض على الجميع).")

st.markdown("## الافتراضات التعريفية")
comparison_table([
    {"الافتراض": "Unconfoundedness / ignorability", "الصيغة": "(Y(1), Y(0)) ⫫ D | X", "المعنى": "كل المربكات مقاسة في X",
     "هل يُختبر؟": "لا — حجة تصميمية + تحليل حساسية"},
    {"الافتراض": "Overlap / positivity", "الصيغة": "0 < P(D=1 | X) < 1", "المعنى": "لكل x يوجد معالَجون وغير معالَجين",
     "هل يُختبر؟": "جزئيًا — توزيع درجة الميل"},
    {"الافتراض": "SUTVA", "الصيغة": "Yᵢ يعتمد على Dᵢ فقط", "المعنى": "لا تداخل بين الوحدات ونسخة واحدة من المعالجة",
     "هل يُختبر؟": "تصميمي"},
    {"الافتراض": "Consistency", "الصيغة": "D = d ⇒ Y = Y(d)", "المعنى": "النتيجة الملاحظة = الكامنة للمعالجة المستلمة", "هل يُختبر؟": "تصميمي"},
])
formula(r"\text{ATE} = \mathbb E_X\big[\mathbb E[Y\mid D=1,X] - \mathbb E[Y\mid D=0,X]\big]", title="Identification by adjustment",
        intuition="تحت الافتراضات، نحوّل الكمية السببية غير الملاحظة إلى كمية إحصائية قابلة للتقدير — هنا تدخل ML لتقدير "
                  "E[Y | D, X] بمرونة.")

st.markdown("## Confounding Lab")
st.caption("DGP: القدرة X تزيد احتمال التدريب D وتزيد الأجر Y. الأثر السببي الحقيقي للتدريب = τ.")
c1, c2, c3 = st.columns(3)
tau = c1.slider("الأثر الحقيقي τ", -2.0, 4.0, 1.0, 0.25, key="cfl_tau")
conf = c2.slider("قوة الإرباك", 0.0, 3.0, 1.5, 0.1, key="cfl_conf")
n = c3.select_slider("n", [200, 500, 2000, 10000], value=2000, key="cfl_n")
r = np.random.default_rng(0)
Xa = r.normal(size=n)
ps = 1 / (1 + np.exp(-conf * Xa))
D = (r.random(n) < ps).astype(int)
Y = tau * D + 2 * conf * Xa + r.normal(size=n)
naive = Y[D == 1].mean() - Y[D == 0].mean()
adj = LinearRegression().fit(np.c_[D, Xa], Y).coef_[0]
c1, c2 = st.columns([1.3, 1])
with c1:
    plot(bars(["truth τ", "naive difference in means", "adjusted for X (OLS)"], [tau, naive, adj], title="Estimates of the treatment effect",
              color=[PALETTE["teal"], PALETTE["coral"], PALETTE["sky"]]), height=320)
with c2:
    plot(overlay_hist({"X | treated": Xa[D == 1], "X | control": Xa[D == 0]}, title="Covariate imbalance", nbins=30), height=320)
st.markdown(f"الفرق الساذج = **{naive:.2f}** (متحيز بـ{naive - tau:+.2f})؛ بعد ضبط X ≈ **{adj:.2f}**. المعالَجون مختلفون أصلًا في X.")
mermaid("""
flowchart LR
  X((X: ability)) --> D[D: training]
  X --> Y[Y: wage]
  D -->|τ| Y
  style X fill:#FFF3BF,stroke:#F59F00
""")
causal_caution("الضبط يعمل هنا لأن المربك الوحيد **مُقاس**. في الواقع، المربكات غير المقاسة تجعل أي تقدير (مهما كان ML متقدمًا) "
               "متحيزًا. DML يحسّن **التقدير** تحت عدم الإرباك، ولا يصنع عدم الإرباك.")
why("ارسم DAG قبل أي نموذج.", "يحدد أي المتغيرات تضبطها (المربكات) وأيها لا تضبطها (الوسطاء والمتصادمات ونواتج المعالجة).")

st.markdown("## التداخل (Overlap)")
lim = st.slider("قوة الإرباك (لرؤية فشل التداخل)", 0.5, 6.0, 1.5, 0.5, key="cfl_ov")
psv = 1 / (1 + np.exp(-lim * r.normal(size=3000)))
Dv = (r.random(3000) < psv).astype(int)
fig = go.Figure()
fig.add_trace(go.Histogram(x=psv[Dv == 1], nbinsx=40, name="treated", opacity=0.6, marker_color=PALETTE["coral"]))
fig.add_trace(go.Histogram(x=psv[Dv == 0], nbinsx=40, name="control", opacity=0.6, marker_color=PALETTE["sky"]))
fig.update_layout(barmode="overlay", title="Propensity score distributions by group", xaxis_title="P(D=1 | X)", height=300)
plot(fig)
st.caption(f"نسبة الوحدات بدرجة ميل خارج [0.02, 0.98] = {np.mean((psv < 0.02) | (psv > 0.98)):.1%}. مع تداخل ضعيف لا توجد وحدات "
           "مقارنة لبعض قيم X؛ الأوزان 1/m(X) تنفجر ويصبح التقدير هشًا (وحدة IRM).")

page_link("dml_core", "التالي: الفكرة الجوهرية في Double ML", ":material/lightbulb:")
if at_least("advanced"):
    st.markdown("## متقدم: Bad controls")
    st.markdown("- **Mediator:** ضبطه يزيل جزءًا من الأثر الكلي.\n- **Collider:** ضبطه يفتح مسارًا مربكًا جديدًا (Selection bias).\n"
                "- **Post-treatment variables:** أي شيء يُقاس بعد D مشبوه.\n- **Instrument كضابط:** يزيد التباين وقد يضخم التحيز.")
if at_least("research"):
    researcher_note(["Imbens & Rubin (2015) للإطار الكامل؛ Rosenbaum & Rubin (1983) لدور درجة الميل.",
                     "تحليل الحساسية للمربكات غير المقاسة ليس اختياريًا في الأبحاث الرصدية (وحدة امتدادات DML)."])
    st.markdown(cite("imbens_rubin", "rosenbaum1983"))
mistakes(["الفرق في المتوسطات كأثر سببي في بيانات رصدية.", "ضبط متغيرات بعد المعالجة.", "تجاهل ضعف التداخل.",
          "الاعتقاد بأن ML قوي يعوّض المربكات غير المقاسة."])
page_footer("causal_foundations",
            takeaways=["الأثر السببي مقارنة بين نتائج كامنة نرى واحدة منها فقط.",
                       "عدم الإرباك والتداخل وSUTVA تحول السؤال السببي إلى إحصائي.", "الجودة التنبؤية لا تثبت السببية."])
