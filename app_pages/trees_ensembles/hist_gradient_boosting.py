import time

import pandas as pd
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import xy
from utils.preprocessing import ordinal_categoricals

page_header("hist_gradient_boosting")
algorithm_profile("hist_gb")

st.markdown("## الفكرة: Binning")
formula(r"x_j \mapsto \operatorname{bin}(x_j)\in\{0,\dots,\texttt{max\_bins}-1\}\ \cup\ \{\text{missing}\}",
        title="Histogram-based splitting",
        intuition="بدل فرز القيم وتجريب كل عتبة (O(n log n))، نقطّع كل خاصية إلى ≤255 صندوقًا مرة واحدة، ثم نبني مدرجات "
                  "التدرجات لكل صندوق: البحث عن التقسيم O(bins) بدل O(n).",
        example="مليون صف × 255 صندوقًا: البحث عن أفضل عتبة لخاصية = 255 مقارنة بدل مليون.")

X, y = xy("mixed")
st.markdown("## القيم المفقودة والفئات أصليًا")
st.markdown(f"بيانات القروض: `income` مفقود {X['income'].isna().mean():.0%} و`credit_score` {X['credit_score'].isna().mean():.0%}، "
            "و3 أعمدة فئوية منها `city` بـ40 فئة. HGB يتعلم عند كل تقسيم **الاتجاه الأفضل للقيم المفقودة**، ويقسّم الفئات "
            "إلى مجموعتين مباشرة دون One-hot.")
Xc = ordinal_categoricals(X)


@st.cache_data(show_spinner="يقارن 3 إعدادات بـ5 طيات…")
def _compare():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cats = ["employment", "region", "city"]
    rows = []
    t0 = time.perf_counter()
    s = cross_val_score(HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0), Xc, y, cv=cv, scoring="roc_auc")
    rows.append({"setup": "HGB native categories + NaN", "ROC-AUC": s.mean(), "sec": time.perf_counter() - t0})
    t0 = time.perf_counter()
    ord_pipe = make_pipeline(make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                                                     remainder="passthrough"),
                             HistGradientBoostingClassifier(random_state=0))
    s = cross_val_score(ord_pipe, X, y, cv=cv, scoring="roc_auc")
    rows.append({"setup": "HGB with ordinal codes treated as numbers", "ROC-AUC": s.mean(), "sec": time.perf_counter() - t0})
    t0 = time.perf_counter()
    Xn = X.select_dtypes("number").fillna(X.select_dtypes("number").median())
    s = cross_val_score(GradientBoostingClassifier(random_state=0), Xn, y, cv=cv, scoring="roc_auc")
    rows.append({"setup": "GradientBoosting (numeric only, median-imputed)", "ROC-AUC": s.mean(), "sec": time.perf_counter() - t0})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="hgb_run", icon=":material/play_arrow:"):
    st.session_state["hgb_done"] = True
if st.session_state.get("hgb_done"):
    st.dataframe(_compare().round(4), hide_index=True, width="stretch")
    st.caption("لاحظ الزمن: HGB أسرع بكثير من GradientBoosting الكلاسيكي حتى على 2000 صف؛ الفرق يصبح هائلًا مع مئات الآلاف.")
st.code("""X = X.astype({c: "category" for c in ["employment", "region", "city"]})
hgb = HistGradientBoostingClassifier(categorical_features="from_dtype",   # default: pandas 'category' columns
                                     learning_rate=0.1, max_leaf_nodes=31, early_stopping="auto")
hgb.fit(X, y)          # NaN handled natively — no imputer needed""", language="python")
why("استخدم HGB كخيار أول للبيانات الجدولية المتوسطة/الكبيرة في scikit-learn.",
    "سريع، يدعم NaN والفئات، له توقف مبكر، ومتاح دون مكتبات إضافية (مهم على Streamlit Cloud).")
intuition("early_stopping='auto' يفعّل التوقف المبكر تلقائيًا فقط إذا كان n > 10,000؛ على البيانات الصغيرة يدرّب كل max_iter.")

st.markdown("## القيود: الرتابة والتفاعلات")
st.code("""HistGradientBoostingRegressor(monotonic_cst={"area": 1, "age": -1},     # price ↑ with area, ↓ with age
                              interaction_cst=[{"area", "rooms"}, {"age"}])  # which features may interact""", language="python")
hyperparameter_table("HistGradientBoostingClassifier")

if at_least("advanced"):
    st.markdown("## متقدم: ما تشترك فيه مع LightGBM")
    st.markdown("HGB مستوحى من LightGBM: مدرجات، نمو أفضل-أولًا بحد max_leaf_nodes، تدرجات وهيسيانات من الرتبة الثانية "
                "لقيم الأوراق، وطرح المدرجات (مدرج الابن = الأب − الأخ). **التعقيد:** O(n·p) لكل شجرة للمدرجات.")
if at_least("research"):
    researcher_note(["منذ 1.9 يحترم HGB أوزان العينات في حساب حدود الصناديق (Weighted quantiles).",
                     "لاستخدامه كمتعلم إزعاج في DML: اضبط early_stopping=False أو اعلم أنه يستخدم جزءًا داخليًا للتحقق."])
    st.markdown(cite("ke2017", "friedman2001"))
template_checklist({3: "Binning", 13: "الفئات أصليًا", 14: "NaN أصليًا", 9: "جدول المعاملات الفائقة",
                    22: "HistGradientBoostingClassifier", 23: "مختبر المقارنة"})
mistakes(["One-hot لكل الفئات قبل HGB دون حاجة.", "تعويض NaN رغم الدعم الأصلي (يضيع معلومة الفقد).",
          "افتراض أن early_stopping مفعّل دائمًا."])
page_footer("hist_gradient_boosting",
            takeaways=["Binning يجعل البحث عن التقسيم سريعًا جدًا.", "NaN والفئات مدعومة أصليًا.", "القيود الرتيبة والتفاعلية متاحة."])
