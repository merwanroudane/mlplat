import pandas as pd
import streamlit as st

from config import VERIFIED_ON
from content.references import OFFICIAL_DOCS, REFERENCES
from core.page import footer, page_header

page_header("references")
st.caption(f"كل DOI هنا حُلّ على doi.org وطابق عنوانه ومؤلفه على Crossref بتاريخ {VERIFIED_ON}، وكل معرّف arXiv طابق عنوانه على arxiv.org، "
           "وكل رابط URL أعاد 200. التفاصيل في docs/research_notes.md.")
q = st.text_input("ابحث في المراجع", key="ref_q")
rows = [{"citation": r.citation, "link": r.url, "id": r.link, "topics": ", ".join(r.topics)} for r in REFERENCES]
df = pd.DataFrame(rows)
if q:
    df = df[df.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)]
st.metric("مراجع", len(df))
st.dataframe(df, hide_index=True, width="stretch", height=520, column_config={"link": st.column_config.LinkColumn("link", display_text="open")})
st.markdown("## الوثائق الرسمية المتحقَّق منها")
st.dataframe(pd.DataFrame(OFFICIAL_DOCS, columns=["documentation", "version verified", "url"]), hide_index=True, width="stretch",
             column_config={"url": st.column_config.LinkColumn("url", display_text="docs")})
st.markdown("## كيف تستشهد بالمنصة")
st.code("Roudane, M. (2026). Machine Learning Interactive Academy (أكاديمية تعلّم الآلة التفاعلية) [Interactive course, v1.0.0].", language="text")
footer()
