import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import missing_notice, optional
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import load_dataset
from utils.plotting import interval_plot, plot

page_header("panel_did")

st.markdown("## بنية Panel")
df = load_dataset("panel")
dataset_card("panel")
comparison_table([
    {"المفهوم": "id / time", "المعنى": "كل صف = وحدة i في فترة t"},
    {"المفهوم": "Treatment timing (g)", "المعنى": "فترة أول معالجة؛ تبنٍّ متدرّج (Staggered) حين تختلف g بين المجموعات"},
    {"المفهوم": "Never-treated", "المعنى": "وحدات لا تُعالَج أبدًا (g = ∞)؛ مجموعة مقارنة نظيفة"},
    {"المفهوم": "Not-yet-treated", "المعنى": "وحدات ستعالج لاحقًا؛ مقارنة بديلة قبل معالجتها"},
    {"المفهوم": "Anticipation", "المعنى": "تغيّر النتيجة قبل المعالجة (توقعًا لها) ⇒ يجب إزاحة فترة الأساس"},
    {"المفهوم": "Panel vs repeated cross-sections", "المعنى": "نفس الوحدات عبر الزمن مقابل عينات جديدة كل فترة (DoubleMLDIDCS)"},
])
means = df.groupby(["g", "t"])["y"].mean().reset_index()
fig = go.Figure()
for i, g in enumerate(sorted(df["g"].unique())):
    m = means[means.g == g]
    label = "never treated" if g == 0 else f"first treated in t = {g}"
    fig.add_trace(go.Scatter(x=m["t"], y=m["y"], mode="lines+markers", name=label, line=dict(color=SEQUENCE[i], width=3)))
    if g:
        fig.add_vline(x=g - 0.5, line=dict(color=SEQUENCE[i], dash="dot"))
fig.update_layout(title="Average outcome by adoption cohort (parallel before treatment, diverging after)", xaxis_title="period t",
                  yaxis_title="mean y", height=380)
plot(fig)

st.markdown("## ATT(g, t) لـCallaway & Sant'Anna")
formula(r"ATT(g,t) = \mathbb E[Y_t(g) - Y_t(\infty)\mid G=g]", title="Group-time average treatment effect",
        symbols={"G = g": "المجموعة التي بدأت معالجتها في g", r"Y_t(\infty)": "النتيجة الكامنة لو لم تُعالَج أبدًا"},
        intuition="بدل معامل TWFE واحد (قد يتحيز مع آثار ديناميكية متدرّجة)، قدّر أثرًا لكل مجموعة وفترة، ثم اجمعها (حدث، مجموعة، زمن).")
formula(r"ATT(g,t) = \mathbb E\big[Y_t - Y_{g-1}\mid G=g\big] - \mathbb E\big[Y_t - Y_{g-1}\mid C\big]\ \ \text{(conditional on X via DR-DiD)}",
        title="DiD identification with a clean comparison group C",
        intuition="تغيّر المعالَجين منذ ما قبل المعالجة ناقص تغيّر المقارنة — مع ضبط X بدرجة مزدوجة المتانة (Sant'Anna & Zhao, 2020) "
                  "ومتعلمي ML (Chang, 2020).")
definition("Conditional parallel trends", "لولا المعالجة، لكان متوسط تغيّر النتيجة للمجموعة g مساويًا لمتوسط تغيّر المقارنة، بمعلومية X.")
warning("TWFE التقليدي (y ~ D + unit FE + time FE) مع تبنٍّ متدرّج وآثار متغيرة زمنيًا قد يستخدم المعالَجين سابقًا كضوابط "
        "ويعطي أوزانًا سالبة — لذلك الأدبيات الحديثة تقدّر ATT(g, t) مباشرة.")

st.markdown("## Staggered DiD Lab: DoubleMLDIDMulti")
dml = optional("doubleml")
if dml is None:
    missing_notice("doubleml")
else:
    c1, c2 = st.columns(2)
    control = c1.segmented_control("مجموعة المقارنة", ["never_treated", "not_yet_treated"], default="never_treated", key="did_ctrl",
                                   required=True)
    antic = c2.slider("anticipation_periods", 0, 1, 0, key="did_ant")

    @st.cache_data(show_spinner="DoubleMLDIDMulti: ATT(g, t) لكل مجموعة وفترة…", max_entries=8)
    def _did(control: str, antic: int):
        from doubleml.did import DoubleMLDIDMulti
        p = df.copy()
        p["g"] = p["g"].astype(float).replace(0, np.inf)   # never-treated coded as infinity
        data = dml.DoubleMLPanelData(p.drop(columns=["d", "true_att"]), y_col="y", d_cols="g", t_col="t", id_col="id",
                                     x_cols=["x1", "x2"])
        m = DoubleMLDIDMulti(data, ml_g=LinearRegression(), ml_m=LogisticRegression(), control_group=control,
                             anticipation_periods=antic, n_folds=5)
        m.fit()
        es = m.aggregate("eventstudy")
        return m.summary, es.aggregated_frameworks.summary

    gt, es = _did(control, antic)
    es = es.reset_index().rename(columns={"index": "event time"})
    et = es["event time"].astype(float)
    fig = interval_plot([f"e = {int(e)}" for e in et], es["coef"], es["2.5 %"], es["97.5 %"], title="Event study (aggregated ATT by time since treatment)",
                        xaxis="effect")
    fig.add_trace(go.Scatter(x=[1 + 0.5 * e if e >= 0 else 0 for e in et], y=[f"e = {int(e)}" for e in et], mode="markers",
                             name="truth", marker=dict(symbol="diamond-open", size=14, color="#212529")))
    plot(fig)
    st.caption("الحقيقة في DGP: ATT = 1 + 0.5·e للفترات e ≥ 0 و0 قبل المعالجة (الماسات). الفترات السالبة اختبار مسبق للاتجاهات المتوازية.")
    with st.expander("جدول ATT(g, t) الكامل"):
        st.dataframe(gt.round(3), width="stretch")
    st.code('''p["g"] = p["g"].replace(0, np.inf)                               # never treated = inf
data = dml.DoubleMLPanelData(p, y_col="y", d_cols="g", t_col="t", id_col="id", x_cols=["x1", "x2"])
did = DoubleMLDIDMulti(data, ml_g=LinearRegression(), ml_m=LogisticRegression(),
                       control_group="never_treated", anticipation_periods=0, n_folds=5)
did.fit(); did.aggregate("eventstudy").plot_effects()''', language="python")
    st.caption("متحقَّق في DoubleML 0.11.4: DoubleMLPanelData(data, y_col, d_cols, t_col, id_col, x_cols, ...)، و"
               "DoubleMLDIDMulti(..., gt_combinations='standard', control_group='never_treated', anticipation_periods=0, "
               "panel=True, ...)، وaggregate('eventstudy'|'group'|'time').")
why("اعرض دائمًا الفترات السابقة للمعالجة.", "معاملات قبل المعالجة قريبة من الصفر تدعم (لا تثبت) الاتجاهات المتوازية.")
intuition("هذا مثال تعليمي بمجموعة بيانات اصطناعية خاصة بالمنصة (لا نسخ لأمثلة التوثيق). ML هنا يضبط X في درجة DR-DiD؛ "
          "التعريف السببي يأتي من الاتجاهات المتوازية الشرطية لا من ML.")

if at_least("advanced"):
    st.markdown("## متقدم: خيارات التصميم")
    comparison_table([
        {"الخيار": "control_group", "المعنى": "never_treated أنظف؛ not_yet_treated يستخدم بيانات أكثر بافتراض أقوى"},
        {"الخيار": "anticipation_periods", "المعنى": "إزاحة فترة الأساس k فترات قبل g"},
        {"الخيار": "panel=True/False", "المعنى": "بيانات Panel أم مقاطع متكررة (DoubleMLDIDCS)"},
        {"الخيار": "aggregate()", "المعنى": "eventstudy (حسب زمن التعرض)، group (حسب المجموعة)، time (حسب الفترة)"},
    ])
if at_least("research"):
    researcher_note(["Callaway & Sant'Anna (2021) لـATT(g, t)؛ Sant'Anna & Zhao (2020) لـDR-DiD؛ Chang (2020) لـDML-DiD.",
                     "لا نستخدم مصطلح «Dynamic DML» دون مرجع: هنا الأثر الديناميكي = ATT حسب زمن التعرض في إطار موثق.",
                     "تحليل حساسية للاتجاهات المتوازية (Rambachan & Roth) امتداد مهم للإبلاغ."])
    st.markdown(cite("callaway2021", "santanna2020", "chang2020"))
mistakes(["TWFE مع تبنٍّ متدرّج وآثار ديناميكية.", "تجاهل الفترات السابقة.", "ضبط متغيرات تتأثر بالمعالجة.",
          "الخلط بين Panel والمقاطع المتكررة."])
page_footer("panel_did",
            takeaways=["ATT(g, t) يتجنب مشكلات TWFE مع التبني المتدرّج.", "DML يضبط X بمرونة داخل درجة DR-DiD.",
                       "الافتراض الحاسم: الاتجاهات المتوازية الشرطية."])
