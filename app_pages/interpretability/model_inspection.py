import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import causal_caution, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.inspection import permutation_importance_manual
from utils.plotting import bars, plot

page_header("model_inspection")

comparison_table([
    {"الأداة": "Coefficients", "للنماذج": "خطية", "تقيس": "أثر هامشي شرطي بوحدة الخاصية", "تحذير": "تعتمد على المقياس والترابط"},
    {"الأداة": "Impurity importance (MDI)", "للنماذج": "الأشجار", "تقيس": "مجموع انخفاض الشوائب", "تحذير": "على التدريب؛ منحازة للكاردينالية"},
    {"الأداة": "Permutation importance", "للنماذج": "أي نموذج", "تقيس": "انخفاض الأداء عند كسر العلاقة", "تحذير": "الترابط يقسم الأهمية"},
    {"الأداة": "PDP / ICE", "للنماذج": "أي نموذج", "تقيس": "شكل العلاقة المتعلَّمة", "تحذير": "استقراء خارج التوزيع"},
    {"الأداة": "SHAP", "للنماذج": "أي نموذج (Tree سريع)", "تقيس": "مساهمة محلية جمعية", "تحذير": "الخلفية والترابط"},
    {"الأداة": "Learning / validation curves", "للنماذج": "أي نموذج", "تقيس": "التحيز والتباين", "تحذير": "—"},
])
formula(r"\text{PI}_j = s(f, X, y) - \frac1K\sum_{k=1}^K s\big(f, X^{(\pi_k, j)}, y\big)", title="Permutation importance (Breiman, 2001)",
        symbols={"s": "مقياس الأداء", r"X^{(\pi, j)}": "X مع خلط العمود j عشوائيًا"},
        intuition="اكسر العلاقة بين الخاصية والهدف بخلطها؛ بقدر ما ينهار الأداء تكون الخاصية مهمة **للنموذج**.")

st.markdown("## Feature Importance Lab")
X, y = xy("regression")
rng = np.random.default_rng(0)
add_dup = st.toggle("أضف نسخة مترابطة من x1 (x1_copy ≈ x1)", value=False, key="mi_dup")
add_id = st.toggle("أضف خاصية ضجيج عالية الكاردينالية", value=False, key="mi_id")
Xm = X.copy()
if add_dup:
    Xm["x1_copy"] = Xm["x1"] + rng.normal(scale=0.05, size=len(Xm))
if add_id:
    Xm["row_noise"] = rng.permutation(len(Xm)).astype(float)
Xa, Xb, ya, yb = train_test_split(Xm, y, test_size=0.3, random_state=0)


@st.cache_data(show_spinner="يدرّب ويحسب الأهميات…", max_entries=8)
def _importances(cols: tuple):
    rf = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xa, ya)
    pi_te = permutation_importance(rf, Xb, yb, n_repeats=10, random_state=0)
    pi_tr = permutation_importance(rf, Xa, ya, n_repeats=10, random_state=0)
    lin = make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 20))).fit(Xa, ya)
    manual, _ = permutation_importance_manual(rf, Xb.to_numpy(), yb.to_numpy(), scoring="r2", n_repeats=5, seed=0)
    return (rf.feature_importances_, pi_te.importances_mean, pi_te.importances_std, pi_tr.importances_mean,
            lin[-1].coef_, manual, rf.score(Xb, yb))


mdi, pim, pis, pitr, coef, manual, r2 = _importances(tuple(Xm.columns))
cols = np.array(Xm.columns)
c1, c2 = st.columns(2)
with c1:
    o = np.argsort(-mdi)
    plot(bars(cols[o], mdi[o], title="RF impurity importance (training data)", horizontal=True, color=PALETTE["coral"]), height=360)
with c2:
    o = np.argsort(-pim)
    plot(bars(cols[o], pim[o], errors=pis[o], title=f"Permutation importance on held-out (R² = {r2:.2f})", horizontal=True,
              color=PALETTE["teal"]), height=360)
c3, c4 = st.columns(2)
with c3:
    o = np.argsort(-np.abs(coef))
    plot(bars(cols[o], coef[o], title="Standardised Ridge coefficients", horizontal=True, color=PALETTE["sky"]), height=360)
with c4:
    comp = pd.DataFrame({"feature": cols, "PI held-out": pim, "PI training": pitr, "PI (our implementation)": manual}).round(3)
    st.dataframe(comp.sort_values("PI held-out", ascending=False), hide_index=True, width="stretch")
st.markdown("**الحقيقة:** y = 3·x1 − 2·x2 + 1.5·x3 + 2·sin(2·x4) + ε. لاحظ: x4 قوي لكن علاقته غير خطية فمعامله الخطي صغير؛ "
            "والخصائص x5..x8 عديمة الأثر.")
if add_dup:
    st.warning("مع x1_copy تنقسم أهمية x1 بين النسختين: خلط واحدة لا يضر كثيرًا لأن الأخرى تحمل المعلومة. الأهمية تقيس «ما يفقده "
               "النموذج دون هذا العمود» لا «أهمية المفهوم».")
why("احسب Permutation importance على بيانات التحقق/الاختبار.",
    "على التدريب تقيس ما حفظه النموذج (بما فيه الضجيج)، لا ما يعمم.")
causal_caution("الأهمية تصف **النموذج**، لا **العالم**: خاصية مهمة للتنبؤ قد لا تكون سببًا، وتغييرها قد لا يغير النتيجة.")

if at_least("advanced"):
    st.markdown("## متقدم: الخصائص المترابطة")
    st.markdown("- اجمع الخصائص المترابطة في مجموعات وخلطها معًا (Grouped permutation).\n"
                "- Conditional permutation يخلط داخل طبقات الخصائص الأخرى لتجنب نقاط غير واقعية.\n"
                "- Drop-column importance (إعادة التدريب دون الخاصية) أصدق لكنه مكلف.")
if at_least("research"):
    researcher_note(["Strobl et al. (2007): MDI منحاز للخصائص المستمرة وعالية الكاردينالية.",
                     "Hooker et al. (2021): الخلط يولّد نقاطًا خارج التوزيع مع الترابط ⇒ «Please stop permuting features»؛ البدائل الشرطية."])
    st.markdown(cite("breiman2001"))
mistakes(["MDI على التدريب كحقيقة.", "تفسير الأهمية سببيًا.", "تجاهل الترابط عند المقارنة.", "مقارنة معاملات غير مقاسة."])
page_footer("model_inspection",
            takeaways=["Permutation importance على بيانات منفصلة هي الأداة العامة الأصدق.", "الترابط يقسم الأهمية بين الخصائص.",
                       "الأهمية تصف النموذج لا العالم السببي."])
