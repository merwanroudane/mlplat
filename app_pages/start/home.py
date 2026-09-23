import html

import streamlit as st

from components.cards import card_grid
from components.diagrams import mermaid
from config import (APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN, APP_SUBTITLE_AR,
                    DATA_SCIENCE_PLATFORM_URL, VERIFIED_ON)
from core.curriculum import GROUPS, LEARNING_PATHS, MODULES, ROADMAP, get_module, modules_in_group, tracked_modules
from core.page import footer
from core.state import is_complete, mark_visited, progress_pct

mark_visited("home")

st.html(
    f"""
<div class="ml-hero">
  <div class="ml-kicker">مقرر جامعي تفاعلي · Interactive textbook + laboratory + research companion</div>
  <h1>{html.escape(APP_NAME_AR)}</h1>
  <div class="ml-en">{html.escape(APP_NAME_EN)}</div>
  <p>{html.escape(APP_SUBTITLE_AR)}</p>
  <div class="ml-author">إعداد وتطوير: <b>{html.escape(APP_AUTHOR_AR)}</b> · <span class="ml-en">{html.escape(APP_AUTHOR_EN)}</span></div>
</div>
"""
)

# ---------------------------------------------------------------- actions
current = st.session_state.get("current_module")
if not current or current == "home" or get_module(current).kind in ("resource", "start"):
    pending = [m for m in tracked_modules() if not is_complete(m.id)]
    current = pending[0].id if pending else "breiman_two_cultures"
with st.container(horizontal=True):
    if st.button("ابدأ من البداية", type="primary", icon=":material/rocket_launch:"):
        st.switch_page(get_module("breiman_two_cultures").file)
    if st.button(f"تابع: {get_module(current).title_ar}", icon=":material/play_arrow:"):
        st.switch_page(get_module(current).file)
    if st.button("مساعد اختيار الخوارزمية", icon=":material/assistant_direction:"):
        st.switch_page(get_module("algorithm_selector").file)
    if st.button("مسار DML", icon=":material/device_hub:"):
        st.switch_page(get_module("causal_foundations").file)
    if st.button("التصنيف غير المتوازن", icon=":material/balance:"):
        st.switch_page(get_module("imbalanced").file)

# ----------------------------------------------------------------- numbers
n_lessons = sum(1 for m in MODULES if m.kind == "lesson")
n_labs = sum(1 for m in MODULES if m.lab)
with st.container(horizontal=True):
    st.metric("وحدات تعليمية", n_lessons, border=True)
    st.metric("مختبرات تفاعلية", n_labs, border=True)
    st.metric("مجموعات المنهج", len(GROUPS), border=True)
    st.metric("وحدات Causal ML / DML", len(modules_in_group("causal_ml")), border=True)
    st.metric("تقدّمك", f"{progress_pct():.0f}%", border=True)

# ------------------------------------------------------- Breiman opening
with st.container(border=True):
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        st.markdown("#### :material/history_edu: نبدأ بسؤال فلسفي: ما الذي نفعله حين نبني نموذجًا؟")
        st.markdown("أول مجموعة في المنهج مخصصة لفلسفة **Leo Breiman**: الثقافتان في النمذجة الإحصائية، ودروس Rashomon وOccam "
                    "وBellman، وإرثه من CART إلى Random Forests — مع مختبرات تعيد إنتاج حججه.")
    with c2:
        st.page_link(get_module("breiman_two_cultures").file, label="الثقافتان", icon=":material/history_edu:")
        st.page_link(get_module("breiman_three_lessons").file, label="Rashomon · Occam · Bellman",
                     icon=":material/diversity_3:")

# --------------------------------------------------------- DSplat bridge
with st.container(key="ml-dsplat-home"):
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        st.markdown("#### :material/cleaning_services: هل بياناتك غير جاهزة؟")
        st.markdown("تفترض هذه المنصة أنك تعرف أساسيات جودة البيانات والتنظيف والقيم المفقودة والشاذة والترميز والقياس "
                    "والتحليل الاستكشافي. إن احتجت مراجعتها فابدأ من **منصة علم البيانات المرافقة (DSplat)**، ثم عد إلى "
                    "وحدة **جاهزية البيانات لتعلّم الآلة** هنا.")
    with c2:
        st.link_button("افتح منصة علم البيانات", DATA_SCIENCE_PLATFORM_URL, icon=":material/open_in_new:",
                       type="primary", width="stretch")
        st.page_link(get_module("ml_readiness").file, label="جاهزية البيانات لـML", icon=":material/fact_check:")

# ----------------------------------------------------------------- roadmap
st.markdown("## خريطة الرحلة · Roadmap")
done_ids = st.session_state.get("completed", set())
chips = []
for i, (en, ar, mid) in enumerate(ROADMAP):
    cls = "ml-step done" if mid in done_ids else "ml-step"
    chips.append(f'<span class="{cls}">{html.escape(en)} · {html.escape(ar)}</span>')
    if i < len(ROADMAP) - 1:
        chips.append('<span class="ml-arrow">→</span>')
st.html(f'<div class="ml-roadmap">{"".join(chips)}</div>')

step = st.pills("اختر محطة من الخريطة لمعرفة ما تغطيه", [r[0] for r in ROADMAP], key="home_roadmap",
                default="Validation")
if step:
    en, ar, mid = next(r for r in ROADMAP if r[0] == step)
    m = get_module(mid)
    with st.container(border=True):
        st.markdown(f"**{ar} · {en}** — {m.description}")
        if m.objectives:
            st.markdown("\n".join(f"- {o}" for o in m.objectives))
        st.page_link(m.file, label=f"اذهب إلى: {m.title_ar}", icon=m.icon)

# ------------------------------------------------------------ what covered
st.markdown("## ماذا ستتعلّم؟")
st.markdown(
    "كل خوارزمية تُعرض بقالب من **27 عنصرًا**: المسألة ← الحدس ← الصياغة الرياضية ← الهندسة ← الخسارة ← خوارزمية "
    "التدريب ← المعاملات والمعاملات الفائقة ← الافتراضات والمعالجة ← التعقيد ← المزايا والعيوب وأنماط الفشل ← التشخيص "
    "والتفسير ← تنفيذ من الصفر ← تنفيذ بالمكتبة الرسمية ← مختبر تفاعلي وساحة معاملات ← تمارين واختبار ← مراجع."
)
card_grid([
    ("قبل تعلّم الآلة", "جاهزية البيانات، Baselines، التقسيم، التسرّب بأنواعه التسعة، والـPipelines."),
    ("الرياضيات والتعلّم الإحصائي", "الجبر الخطي والتفاضل والاحتمال والإحصاء، المخاطرة التجريبية، التحيّز والتباين، نظرية التعلّم."),
    ("الخسارة والتحسين", "دوال الخسارة، الانحدار التدرجي وSGD، Newton وL-BFGS وCoordinate descent والتوقف المبكر."),
    ("التعلّم الموجّه", "الانحدار والتنظيم، اللوجستي، Naive Bayes، LDA/QDA، kNN، SVM، الأشجار، الغابات، والتعزيز الحديث."),
    ("التقييم والضبط", "CV بأنواعه، المقاييس، العتبة، المعايرة، عدم التوازن، التنبؤ المطابق، Grid/Random/Optuna."),
    ("غير الموجّه والتفسير", "K-Means وDBSCAN وGMM وPCA وt-SNE والشذوذ، ثم Permutation importance وPDP/ICE وSHAP."),
    ("Causal ML & Double ML", "النتائج الكامنة، التعامد، Cross-fitting، PLR/PLIV/IRM/IIVM، CATE/GATE، DiD، الحساسية، ومختبر Monte Carlo."),
    ("الإنتاج", "Pipelines قابلة للنشر، MLOps، الانجراف والمراقبة، العدالة، وسجل تجارب قابل لإعادة الإنتاج."),
    ("المشاريع", "عشرة مشاريع كاملة ومشروعان نهائيان: تنبؤي وسببي."),
], columns=3)

# --------------------------------------------------------- learning paths
st.markdown("## مسارات التعلّم · Learning paths")
tabs = st.tabs([p["title"] for p in LEARNING_PATHS.values()])
for tab, (pid, path) in zip(tabs, LEARNING_PATHS.items()):
    with tab:
        st.markdown(path["desc"])
        done = sum(is_complete(mid) for mid in path["modules"])
        st.progress(done / len(path["modules"]), text=f"{done} / {len(path['modules'])} وحدة مكتملة")
        cols = st.columns(2)
        for i, mid in enumerate(path["modules"]):
            m = get_module(mid)
            with cols[i % 2]:
                st.page_link(m.file, label=f"{i + 1}. {m.title_ar} · {m.title_en}",
                             icon=":material/check_circle:" if is_complete(mid) else m.icon)

with st.expander("خريطة المنهج الكاملة (المجموعات والتسلسل)", icon=":material/account_tree:"):
    mermaid("""
flowchart LR
  A[Before ML<br/>readiness · splits · leakage · pipelines] --> B[Foundations<br/>math · risk · bias-variance · loss]
  B --> C[Optimization<br/>GD · SGD · Newton · L-BFGS]
  C --> D[Supervised<br/>linear · logistic · NB · LDA · kNN · SVM]
  D --> E[Trees & Ensembles<br/>CART · RF · boosting · XGB/LGBM/CatBoost]
  E --> F[Validation & Evaluation<br/>CV · metrics · threshold · calibration]
  F --> G[HPO<br/>grid · random · halving · Optuna]
  B --> H[Unsupervised<br/>clustering · PCA · anomaly]
  G --> I[Interpretability<br/>permutation · PDP/ICE · SHAP]
  F --> J[Causal ML & DML<br/>orthogonality · cross-fitting · PLR/IRM · DiD]
  I --> K[Production<br/>MLOps · drift · fairness]
  J --> L[Projects & Capstones]
  K --> L
""")

st.caption(f"الإصدارات والقيم الافتراضية وواجهات المكتبات متحقَّق منها بتاريخ {VERIFIED_ON} — راجع صفحة «عن المنصة» والمراجع.")
footer()
