import pandas as pd
import streamlit as st

from components.algorithm_profile import _TEMPLATE_ITEMS
from components.cards import comparison_table
from components.diagrams import graphviz
from config import DATA_SCIENCE_PLATFORM_URL
from core.curriculum import GROUPS, LEARNING_PATHS, MODULES, get_module
from core.page import footer, page_header

page_header("course_guide")

st.markdown("## مستويات الشرح (من الشريط الجانبي)")
comparison_table([
    {"المستوى": "مبتدئ", "ما يظهر": "الحدس، الرسوم، رياضيات مبسطة، كود أساسي، المعاملات الفائقة الأساسية"},
    {"المستوى": "متقدم", "ما يظهر": "الاشتقاقات، الافتراضات، التعقيد، تفاعلات المعاملات، الحالات الحدية، التشخيص"},
    {"المستوى": "بحثي", "ما يظهر": "نظرية التعلّم الإحصائي، تحفظات الاستدلال، إعادة الإنتاج، السببي مقابل التنبؤي، المراجع وإرشادات الإبلاغ"},
])
st.markdown("## قالب الخوارزمية (27 عنصرًا)")
st.dataframe(pd.DataFrame({"العنصر": _TEMPLATE_ITEMS}), hide_index=True, width="stretch", height=300)
st.caption("في كل صفحة خوارزمية تجد «خريطة القالب» التي تحدد موضع كل عنصر.")

st.markdown("## عناصر كل صفحة")
comparison_table([
    {"العنصر": "بطاقة الوحدة", "الوصف": "الأهداف، المتطلبات، الصعوبة، الزمن، المختبر، المفضلة"},
    {"العنصر": "صناديق الصيغ", "الوصف": "المعادلة + الرموز + الحدس + مثال عددي"},
    {"العنصر": "المختبرات", "الوصف": "منزلقات وأزرار تشغّل كودًا محددًا مسبقًا (آمن)"},
    {"العنصر": "Animations", "الوصف": "تشغيل/إيقاف/السابق/التالي/إعادة/السرعة + تقليل الحركة"},
    {"العنصر": "Callouts", "الوصف": "حدس، لماذا؟، أخطاء شائعة، ملاحظة الباحث، تنبيه سببي"},
    {"العنصر": "الاختبار والتمارين", "الوصف": "أسئلة متنوعة مع تفسير، وتمارين بثلاثة مستويات مع تلميح وحل"},
    {"العنصر": "سجل التجارب", "الوصف": "زر «سجّل» في المختبرات يحفظ الإعدادات والنتائج والإصدارات"},
])

st.markdown("## خريطة المتطلبات السابقة")
group = st.selectbox("المجموعة", list(GROUPS), format_func=GROUPS.get, index=list(GROUPS).index("causal_ml"), key="cg_group")
mods = [m for m in MODULES if m.group == group]
ids = {m.id for m in mods}
lines = ["digraph G {", 'rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#E7F5FF", color="#4DABF7", fontname="Helvetica"];']
for m in mods:
    lines.append(f'"{m.id}" [label="{m.title_en}"];')
    for p in m.prereqs:
        if p not in ids:
            lines.append(f'"{p}" [label="{get_module(p).title_en}", fillcolor="#F3F0FF", color="#9775FA"];')
        lines.append(f'"{p}" -> "{m.id}";')
lines.append("}")
graphviz("\n".join(lines))
st.caption("الأزرق: وحدات المجموعة؛ البنفسجي: متطلبات من مجموعات أخرى.")

st.markdown("## مسارات مقترحة")
for p in LEARNING_PATHS.values():
    with st.expander(p["title"], icon=":material/route:"):
        st.markdown(p["desc"])
        for i, mid in enumerate(p["modules"], 1):
            st.page_link(get_module(mid).file, label=f"{i}. {get_module(mid).title_ar}", icon=get_module(mid).icon)
st.markdown("## للأساتذة")
st.markdown("- اختر المستوى «مبتدئ» للعرض الأول ثم «متقدم» للنقاش.\n- المختبرات مصممة للعرض الحي: غيّر منزلقًا واحدًا واطلب من الطلبة التنبؤ بالنتيجة قبل ظهورها.\n"
            "- التمارين البحثية مناسبة لطلبة الدراسات العليا كواجبات.\n"
            f"- إن احتاج الطلبة مراجعة تنظيف البيانات: [DSplat]({DATA_SCIENCE_PLATFORM_URL}).")
footer()
