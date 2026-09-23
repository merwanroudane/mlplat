import html

import pandas as pd
import streamlit as st

from config import (APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN, APP_VERSION, DATA_SCIENCE_PLATFORM_URL, DATA_SCIENCE_REPO_URL,
                    VERIFIED_ON)
from core.curriculum import GROUPS, MODULES
from core.page import footer, page_header
from core.registry import OPTIONAL_DEPS, installed_version
from utils.validation import package_versions

page_header("about")
st.html(f"""<div class="ml-hero"><div class="ml-kicker">عن المنصة</div><h1>{html.escape(APP_NAME_AR)}</h1>
<div class="ml-en">{html.escape(APP_NAME_EN)} · v{APP_VERSION}</div>
<div class="ml-author">إعداد وتطوير: <b>{html.escape(APP_AUTHOR_AR)}</b> · <span class="ml-en">{html.escape(APP_AUTHOR_EN)}</span></div></div>""")
st.markdown(f"""
منصة عربية أكاديمية تفاعلية لتدريس تعلّم الآلة من جاهزية البيانات إلى الإنتاج، مع مسار مستقل ومتعمق في **Double Machine Learning**.
صُممت للأساتذة (للاستخدام أثناء المحاضرة)، والباحثين (مستوى «بحثي» حقيقي بمراجع متحقَّق منها)، وطلبة الدراسات العليا.

- **المنصة المرافقة:** [DSplat — أكاديمية علم البيانات التفاعلية]({DATA_SCIENCE_PLATFORM_URL}) ([GitHub]({DATA_SCIENCE_REPO_URL})) — نفس نظام التصميم.
- **المحتوى:** {len(MODULES)} صفحة في {len(GROUPS)} مجموعة، {sum(1 for m in MODULES if m.lab)} مختبرًا/وحدة بمختبر.
- **تاريخ التحقق من الإصدارات والواجهات:** {VERIFIED_ON}.
""")
st.markdown("### المنهجية")
st.markdown("- كل خوارزمية بقالب من 27 عنصرًا، وثلاثة مستويات شرح (مبتدئ/متقدم/بحثي).\n"
            "- القيم الافتراضية للمعاملات الفائقة مقروءة من المكتبات المثبتة ومختبرة آليًا.\n"
            "- مقدِّرات DML مكتوبة من الصيغ ومطابقة لحزمة DoubleML على الطيات نفسها (اختبارات آلية).\n"
            "- لا تنفيذ لأي كود يكتبه المستخدم (لا eval/exec/subprocess)؛ المختبرات تشغّل دوال محددة مسبقًا.\n"
            "- لا تُرسل أي بيانات إلى خدمات خارجية؛ كل البيانات اصطناعية ومولَّدة محليًا ببذور ثابتة.")
st.markdown("### البيئة الحالية")
pv = package_versions()
st.dataframe(pd.DataFrame(list(pv.items()), columns=["package", "installed"]), hide_index=True, width="stretch")
st.markdown("### المكتبات الاختيارية")
st.dataframe(pd.DataFrame([{"package": d.pip_name, "purpose": d.purpose_ar, "verified": d.verified_version,
                            "installed here": installed_version(k) or "not installed"} for k, d in OPTIONAL_DEPS.items()]),
             hide_index=True, width="stretch")
st.caption("إن غابت مكتبة اختيارية تعرض صفحتها المحتوى النظري والكود المرجعي ورسالة تثبيت، دون أن تنهار.")
footer()
