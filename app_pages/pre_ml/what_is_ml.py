import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import flow, mermaid
from content.references import cite
from core.page import dsplat_link, page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("what_is_ml")

st.markdown("## التعريف")
definition("Machine Learning (Mitchell, 1997)",
           "يُقال إن برنامجًا **يتعلّم** من خبرة **E** بخصوص مجموعة مهام **T** ومقياس أداء **P**، إذا تحسّن أداؤه في "
           "المهام T، كما يقيسه P، مع ازدياد الخبرة E.")
st.markdown(
    "- **T (المهمة):** مثلًا «تحديد ما إذا كانت معاملة بطاقة احتيالية».\n"
    "- **E (الخبرة):** معاملات سابقة معروف تصنيفها.\n"
    "- **P (الأداء):** مثلًا Recall للاحتيال عند نسبة إنذارات كاذبة مقبولة.\n\n"
    "إذا لم تستطع تحديد T وE وP بجملة واحدة لكل منها، فالمسألة لم تُعرَّف بعد — وهذا أول سبب لفشل المشاريع."
)

st.markdown("## البرمجة التقليدية مقابل تعلّم الآلة")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**البرمجة التقليدية**")
    mermaid("flowchart LR\n  R[قواعد يكتبها الإنسان] --> P[البرنامج]\n  D[بيانات] --> P\n  P --> O[مخرجات]")
with c2:
    st.markdown("**تعلّم الآلة**")
    mermaid("flowchart LR\n  D[بيانات] --> L[خوارزمية تعلّم]\n  Y[مخرجات معروفة] --> L\n  L --> M[نموذج = قواعد متعلَّمة]")
intuition("في تعلّم الآلة لا نكتب القاعدة «إذا كان المبلغ > 5000 وفي بلد جديد فهو احتيال»؛ بل نعطي الخوارزمية أمثلة، "
          "فتستنتج قاعدة (غالبًا أعقد وأدق) تعمّم على حالات لم ترها.")

st.markdown("## مثال حي: التعلّم = التحسّن مع الخبرة")
st.caption("نموذج Logistic Regression يتعلم من عدد متزايد من الأمثلة؛ لاحظ كيف يتحسن الأداء على بيانات لم يرها.")


@st.cache_data(show_spinner=False)
def _learning_with_experience():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from utils.datasets import xy
    X, y = xy("classification")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    sizes = [10, 20, 40, 80, 160, 320, 600]
    rng = np.random.default_rng(0)
    accs = []
    for n in sizes:
        idx = rng.choice(len(X_tr), n, replace=False)
        m = LogisticRegression(max_iter=2000).fit(X_tr.iloc[idx], y_tr.iloc[idx])
        accs.append(m.score(X_te, y_te))
    return sizes, accs


sizes, accs = _learning_with_experience()
fig = go.Figure(go.Scatter(x=sizes, y=accs, mode="lines+markers", line=dict(color=PALETTE["purple"], width=3),
                           marker=dict(size=9), name="test accuracy"))
fig.add_hline(y=0.5, line=dict(dash="dot", color=PALETTE["muted"]), annotation_text="chance level")
fig.update_layout(title="P improves with E (Mitchell's definition in action)", xaxis_title="E: number of training examples",
                  yaxis_title="P: accuracy on unseen data", xaxis_type="log", height=360)
plot(fig)

st.markdown("## ML والإحصاء وعلم البيانات والذكاء الاصطناعي")
comparison_table([
    {"المجال": "Statistics", "السؤال النموذجي": "ما العلاقة؟ ما عدم اليقين؟", "المخرج": "تقديرات + فترات ثقة + اختبارات",
     "معيار النجاح": "صحة الاستدلال تحت افتراضات"},
    {"المجال": "Machine Learning", "السؤال النموذجي": "ما التنبؤ الأفضل لحالة جديدة؟", "المخرج": "نموذج تنبؤي",
     "معيار النجاح": "أداء التعميم خارج العينة"},
    {"المجال": "Causal ML", "السؤال النموذجي": "ماذا يحدث لو غيّرنا X؟", "المخرج": "أثر سببي + عدم اليقين",
     "معيار النجاح": "صحة التعريف (Identification) + استدلال صحيح"},
    {"المجال": "Data Science", "السؤال النموذجي": "كل ما سبق + البيانات والتواصل", "المخرج": "قرارات مدعومة بالبيانات",
     "معيار النجاح": "أثر في القرار الحقيقي"},
    {"المجال": "AI", "السؤال النموذجي": "سلوك ذكي عام", "المخرج": "أنظمة", "معيار النجاح": "أداء المهمة"},
])

st.markdown("## دورة حياة مشروع تعلّم الآلة")
flow(["Problem & metric", "Data readiness", "Split", "Baseline", "Pipeline + model", "Validation & HPO",
      "Test once", "Interpret", "Deploy", "Monitor"], direction="LR")
why("ابدأ دائمًا من المسألة والمقياس لا من الخوارزمية.",
    "الخوارزمية الأفضل لمقياس خاطئ تُنتج نموذجًا ممتازًا لمسألة لا تهم أحدًا. وتقدير أداء صادق يتطلب أن تكون "
    "مجموعة الاختبار قد عُزلت قبل أي قرار نمذجة.")
dsplat_link("جودة البيانات وتنظيفها")

if at_least("advanced"):
    st.markdown("## متقدم: صياغة التعلّم رياضيًا")
    st.markdown("نفترض أن الأزواج $(X, Y)$ تُسحب مستقلة من توزيع مجهول $P$. نبحث عن دالة $f$ من فضاء فرضيات "
                "$\\mathcal{F}$ تجعل **المخاطرة المتوقعة** صغيرة:")
    st.latex(r"R(f) = \mathbb{E}_{(X,Y)\sim P}\big[\ell(Y, f(X))\big]")
    st.markdown("لكننا لا نعرف $P$، فنقلل **المخاطرة التجريبية** على العينة (ERM):")
    st.latex(r"\hat f = \arg\min_{f\in\mathcal F} \frac1n\sum_{i=1}^n \ell(y_i, f(x_i)) \;+\; \lambda\,\Omega(f)")
    st.markdown("كل ما في هذا المقرر تقريبًا هو اختيارات لـ: $\\ell$ (الخسارة)، $\\mathcal F$ (الخوارزمية)، "
                "$\\Omega$ (التنظيم)، وطريقة الحل (التحسين)، وطريقة تقدير $R(\\hat f)$ (التحقق).")
if at_least("research"):
    researcher_note([
        "التعريف التنبؤي للنجاح (خطأ خارج العينة) يختلف عن معيار الاستدلال الإحصائي (صحة التغطية/الحجم)؛ "
        "انظر Shmueli (2010) وMullainathan & Spiess (2017).",
        "افتراض i.i.d. هو ما يجعل الأداء على عينة اختبار تقديرًا غير متحيز للمخاطرة؛ يسقط مع الانجراف والمجموعات والزمن.",
        "في أوراقك البحثية: عرّف T وE وP صراحة، وصِف مصدر البيانات والتقسيم قبل عرض أي نتيجة.",
    ])
    st.markdown(cite("mitchell1997", "shmueli2010", "mullainathan2017", "esl"))

real_world(["هل يوجد قرار سيتغير بناءً على التنبؤ؟", "هل يمكن قياس النجاح بمقياس واحد متفق عليه؟",
            "هل الأمثلة التاريخية تمثل الحالات المستقبلية؟", "ما تكلفة الخطأ بنوعيه؟"])

page_footer("what_is_ml",
            takeaways=["ML = تحسّن الأداء P في المهمة T مع الخبرة E.",
                       "النموذج قواعد متعلَّمة من أمثلة، يُحكم عليه بأدائه على بيانات جديدة.",
                       "ابدأ من المسألة والمقياس، ثم البيانات، ثم الخوارزمية."],
            mistakes=["البدء بالخوارزمية قبل تعريف المسألة.", "قياس النجاح على بيانات التدريب.",
                      "الخلط بين التنبؤ الجيد والفهم السببي."])
