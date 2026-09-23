import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.code_lab import explain_code
from components.diagrams import mermaid
from config import MAX_MC_REPS
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import hist, plot

page_header("data_splits")

st.markdown("## ثلاثة أدوار لا تُخلط")
mermaid("""
flowchart LR
  D[(All data)] --> TR[Train · fit parameters]
  D --> VA[Validation · choose model & hyperparameters]
  D --> TE[Test · one final estimate]
  TR --> VA --> TE
  style TE fill:#FFF4E6,stroke:#F76707
""")
comparison_table([
    {"الجزء": "Train", "الدور": "تعلّم المعاملات", "من يراه؟": "fit", "تحذير": "الأداء هنا متفائل دائمًا"},
    {"الجزء": "Validation", "الدور": "اختيار النموذج والمعاملات الفائقة والعتبة", "من يراه؟": "أنت أثناء التطوير",
     "تحذير": "كثرة المقارنات تجعله متفائلًا أيضًا"},
    {"الجزء": "Test", "الدور": "تقدير نهائي لأداء الإجراء كاملًا", "من يراه؟": "مرة واحدة في النهاية",
     "تحذير": "أي قرار بعد رؤيته يلوثه"},
])
intuition("Test مثل الامتحان النهائي: إن رأيت أسئلته أثناء المذاكرة فدرجتك لم تعد تقيس فهمك.")

st.markdown("## train_test_split قطعة قطعة")
explain_code("X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)", [
    ("test_size=0.2", "20% للاختبار"),
    ("stratify=y", "يحافظ على نسب الفئات في الجزأين (مهم مع عدم التوازن)"),
    ("random_state=42", "تقسيم قابل لإعادة الإنتاج"),
    ("shuffle=True (افتراضي)", "خلط قبل التقسيم — خطأ للبيانات الزمنية"),
], output="أربع مصفوفات", interpretation="للبيانات الزمنية استخدم تقسيمًا حسب التاريخ، وللمجموعات GroupShuffleSplit.")

st.markdown("## مختبر: كم تتقلب درجة الاختبار؟")
st.caption("نكرر التقسيم العشوائي مرات عديدة ونرسم توزيع دقة الاختبار. مجموعة اختبار صغيرة = تقدير متقلب.")
X, y = xy("classification")
c1, c2, c3 = st.columns(3)
test_size = c1.slider("حجم الاختبار (نسبة)", 0.05, 0.5, 0.2, 0.05, key="ds_ts")
reps = c2.slider("عدد التكرارات", 20, MAX_MC_REPS, 100, 10, key="ds_reps")
strat = c3.toggle("stratify", value=True, key="ds_strat")


@st.cache_data(show_spinner="يكرر التقسيم…", max_entries=16)
def _split_variability(test_size: float, reps: int, strat: bool):
    accs = []
    for r in range(reps):
        a, b, c, d = train_test_split(X, y, test_size=test_size, random_state=r, stratify=y if strat else None)
        accs.append(LogisticRegression(max_iter=2000).fit(a, c).score(b, d))
    return np.array(accs)


accs = _split_variability(test_size, reps, strat)
n_test = int(round(len(y) * test_size))
c1, c2 = st.columns([2, 1])
with c1:
    fig = hist(accs, title=f"Test accuracy over {reps} random splits (n_test ≈ {n_test})", nbins=25, color=PALETTE["purple"])
    fig.add_vline(x=accs.mean(), line=dict(color=PALETTE["coral"], dash="dash"), annotation_text="mean")
    plot(fig, height=340)
with c2:
    st.metric("المتوسط", f"{accs.mean():.3f}")
    st.metric("الانحراف المعياري", f"{accs.std(ddof=1):.3f}")
    st.metric("المدى", f"{accs.min():.3f} – {accs.max():.3f}")
    se = np.sqrt(accs.mean() * (1 - accs.mean()) / n_test)
    st.caption(f"تقريب ثنائي الحد للخطأ المعياري لدقة على {n_test} حالة: √(p(1−p)/n) ≈ {se:.3f}")
why("لا تعلن فرقًا بين نموذجين أصغر من تقلّب التقسيم.",
    "فرق 1% بين نموذجين على مجموعة اختبار من 200 حالة يقع غالبًا داخل الضجيج. استخدم CV أو فترات Bootstrap.")
page_link("cross_validation", "الحل: التحقق المتقاطع", ":material/view_week:")

st.markdown("## أي تقسيم يناسب بياناتك؟")
comparison_table([
    {"البنية": "مستقلة ومتجانسة", "التقسيم": "عشوائي (+ stratify للتصنيف)", "الأداة": "train_test_split"},
    {"البنية": "كيانات متكررة (مرضى، عملاء)", "التقسيم": "حسب الكيان", "الأداة": "GroupShuffleSplit / GroupKFold"},
    {"البنية": "زمنية/تنبؤ بالمستقبل", "التقسيم": "الماضي للتدريب، المستقبل للاختبار", "الأداة": "تقطيع بالتاريخ / TimeSeriesSplit"},
    {"البنية": "Panel", "التقسيم": "حسب الكيان أو الزمن أو كليهما حسب سؤال التعميم", "الأداة": "GroupKFold + زمن"},
])

st.markdown("## التقسيم الزمني: تصوير")
days = np.arange(100)
fig = go.Figure()
fig.add_trace(go.Bar(x=days[:70], y=np.ones(70), marker_color=PALETTE["sky"], name="train (past)"))
fig.add_trace(go.Bar(x=days[70:85], y=np.ones(15), marker_color=PALETTE["purple"], name="validation"))
fig.add_trace(go.Bar(x=days[85:], y=np.ones(15), marker_color=PALETTE["coral"], name="test (future)"))
fig.update_layout(barmode="stack", height=180, yaxis=dict(visible=False), xaxis_title="time", bargap=0,
                  legend=dict(orientation="h", y=1.3), margin=dict(t=30))
plot(fig)

if at_least("advanced"):
    st.markdown("## متقدم: ماذا يقدّر Test بالضبط؟")
    st.markdown("إذا دُرّب النموذج النهائي على Train+Validation بعد الاختيار، فدرجة Test تقدّر أداء **هذا النموذج**. أما إن "
                "استُخدم Test في الاختيار ولو مرة، فتصبح متحيزة للأعلى (Optimistic bias)، ويزداد التحيز مع عدد النماذج "
                "المقارنة. الحل: Nested CV أو Test معزول فعليًا.")
if at_least("research"):
    researcher_note(["أبلغ عن حجم الاختبار وفترة ثقة للمقياس (Bootstrap على الاختبار)، لا عن رقم نقطي فقط.",
                     "في المسابقات، Leaderboard العام يُستهلك بتكرار الإرسال: هذا Test reuse بالضبط."])
mistakes(["اختيار النموذج أو العتبة على Test.", "تقسيم عشوائي لبيانات زمنية.", "Test صغير جدًا بلا فترة ثقة.",
          "تنظيف أو تحويل كل البيانات قبل التقسيم."])
page_footer("data_splits",
            takeaways=["Train للتعلّم، Validation للاختيار، Test للتقدير النهائي مرة واحدة.",
                       "درجة Test نفسها متغير عشوائي؛ حجمه يحدد دقته.", "التقسيم يجب أن يحاكي الاستخدام الحقيقي."])
