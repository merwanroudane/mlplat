import pandas as pd
import streamlit as st

from content.glossary import GLOSSARY
from core.curriculum import get_module
from core.page import footer, page_header

page_header("glossary")
q = st.text_input("ابحث في المسرد", key="gl_q")
df = pd.DataFrame(GLOSSARY, columns=["English", "العربية", "التعريف", "module"])
if q:
    df = df[df.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)]
df = df.sort_values("English")
df["الوحدة"] = df["module"].map(lambda m: get_module(m).title_ar)
st.metric("مصطلحات", len(df))
st.dataframe(df.drop(columns="module"), hide_index=True, width="stretch", height=560)
st.download_button("تنزيل المسرد CSV", df.drop(columns="module").to_csv(index=False).encode("utf-8-sig"), "glossary.csv", "text/csv",
                   icon=":material/download:")
footer()
