import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils import metrics as M
from utils.plotting import overlay_hist, plot

page_header("drift")

comparison_table([
    {"النوع": "Data (covariate) drift", "ما يتغير": "P(X)", "مثال": "عملاء أصغر سنًا بعد حملة", "الكشف": "PSI/KS على الخصائص — فوري"},
    {"النوع": "Label (prior) drift", "ما يتغير": "P(y)", "مثال": "ارتفاع نسبة الاحتيال موسميًا", "الكشف": "توزيع التنبؤات/التسميات"},
    {"النوع": "Concept drift", "ما يتغير": "P(y | X)", "مثال": "المحتالون يغيّرون أسلوبهم", "الكشف": "الأداء بعد وصول التسميات"},
    {"النوع": "Performance drift", "ما يتغير": "المقياس نفسه", "مثال": "AUC ينخفض", "الكشف": "يتطلب تسميات (غالبًا متأخرة)"},
])
formula(r"\text{PSI} = \sum_{b}\big(a_b - e_b\big)\ln\frac{a_b}{e_b}", title="Population stability index",
        symbols={"e_b": "نسبة المرجع (التدريب) في الصندوق b", "a_b": "النسبة الحالية"},
        intuition="قاعدة إبهام شائعة: < 0.1 مستقر، 0.1–0.25 تغير معتدل، > 0.25 تغير كبير — عتبات تقليدية لا قوانين.")

st.markdown("## Drift Simulator")
c1, c2, c3, c4 = st.columns(4)
shift = c1.slider("إزاحة متوسط x1 (Data drift)", 0.0, 2.0, 0.0, 0.1, key="dr_shift")
scale = c2.slider("مضاعف تباين x1", 0.5, 3.0, 1.0, 0.1, key="dr_scale")
concept = c3.slider("تغير العلاقة (Concept drift)", 0.0, 1.0, 0.0, 0.1, key="dr_concept")
prior = c4.slider("تغير الانتشار (Label drift)", -2.0, 2.0, 0.0, 0.25, key="dr_prior")
rng = np.random.default_rng(0)


def gen(n, shift=0.0, scale=1.0, concept=0.0, prior=0.0, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    X[:, 0] = X[:, 0] * scale + shift
    w = (1 - concept) * np.array([1.5, -1.0, 0.5]) + concept * np.array([-0.5, -1.0, 1.5])
    logit = X @ w - 1.0 + prior
    return X, (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)


Xtr, ytr = gen(3000, seed=1)
model = LogisticRegression().fit(Xtr, ytr)
Xnow, ynow = gen(3000, shift, scale, concept, prior, seed=2)
p_tr, p_now = model.predict_proba(Xtr)[:, 1], model.predict_proba(Xnow)[:, 1]
psi_x1 = M.psi(Xtr[:, 0], Xnow[:, 0])
ks = stats.ks_2samp(Xtr[:, 0], Xnow[:, 0])
psi_pred = M.psi(p_tr, p_now)
with st.container(horizontal=True):
    st.metric("PSI(x1)", f"{psi_x1:.3f}", border=True)
    st.metric("KS(x1)", f"{ks.statistic:.3f}", f"p = {ks.pvalue:.1e}", delta_color="off", border=True)
    st.metric("PSI(predictions)", f"{psi_pred:.3f}", border=True)
    st.metric("AUC reference → now", f"{roc_auc_score(ytr, p_tr):.3f} → {roc_auc_score(ynow, p_now):.3f}", border=True)
    st.metric("positive rate", f"{ytr.mean():.2f} → {ynow.mean():.2f}", border=True)
c1, c2 = st.columns(2)
with c1:
    plot(overlay_hist({"reference (training)": Xtr[:, 0], "current": Xnow[:, 0]}, title="x1 distribution", nbins=40), height=300)
with c2:
    plot(overlay_hist({"reference": p_tr, "current": p_now}, title="Predicted probability distribution", nbins=40), height=300)
st.caption(f"KS من تنفيذنا (utils/metrics.ks_statistic) = {M.ks_statistic(Xtr[:, 0], Xnow[:, 0]):.4f} يطابق SciPy.")
intuition("جرّب: Concept drift وحده ⇒ PSI للخصائص ≈ 0 بينما ينهار AUC — لا تكتشفه مراقبة المدخلات؛ تحتاج تسميات. وData drift وحده "
          "⇒ PSI كبير وقد يبقى AUC جيدًا — ليس كل انجراف ضارًا.")

st.markdown("## مراقبة عبر الزمن وسياسة إعادة التدريب")


@st.cache_data(show_spinner=False)
def _timeline():
    rows = []
    for month in range(1, 13):
        c = 0.0 if month < 6 else min(1.0, (month - 5) * 0.2)
        Xm, ym = gen(1500, shift=0.08 * month, concept=c, seed=100 + month)
        pm = model.predict_proba(Xm)[:, 1]
        rows.append({"month": month, "PSI(x1)": M.psi(Xtr[:, 0], Xm[:, 0]), "AUC": roc_auc_score(ym, pm)})
    return pd.DataFrame(rows)


tl = _timeline()
fig = go.Figure()
fig.add_trace(go.Scatter(x=tl["month"], y=tl["PSI(x1)"], name="PSI(x1)", line=dict(color=PALETTE["coral"], width=3)))
fig.add_trace(go.Scatter(x=tl["month"], y=tl["AUC"], name="AUC", yaxis="y2", line=dict(color=PALETTE["sky"], width=3)))
fig.add_hline(y=0.25, line=dict(color=PALETTE["coral"], dash="dot"), annotation_text="PSI alert 0.25")
fig.update_layout(title="Monthly monitoring: slow covariate drift, then concept drift from month 6", height=340,
                  yaxis=dict(title="PSI"), yaxis2=dict(title="AUC", overlaying="y", side="right"), xaxis_title="month")
plot(fig)
why("ضع سياسة مكتوبة: متى تنبّه، ومتى تعيد التدريب، ومتى تتراجع.",
    "مثال: PSI > 0.25 على خاصية مهمة ⇒ تحقيق؛ انخفاض AUC > 0.03 على نافذة تسميات ⇒ إعادة تدريب؛ فشل بوابة التحقق ⇒ تراجع.")

if at_least("research"):
    researcher_note(["Rabanser et al. (2019): اختبارات ثنائية العينة على تمثيلات مخفضة الأبعاد فعالة لكشف الانزياح.",
                     "Gama et al. (2014): مسح لتكيّف النماذج مع انجراف المفهوم.",
                     "مع انزياح X فقط (Covariate shift) يمكن إعادة الوزن بنسبة الكثافات إن بقيت P(y|X) ثابتة."])
    st.markdown(cite("rabanser2019", "gama2014"))
mistakes(["مراقبة المدخلات فقط.", "إعادة تدريب آلية على بيانات منجرفة دون تحقق.", "عتبات PSI كقوانين مطلقة."])
page_footer("drift",
            takeaways=["أربعة أنواع: بيانات، تسميات، مفهوم، أداء.", "PSI/KS تكشف تغيّر المدخلات لا تغيّر العلاقة.",
                       "سياسة إعادة تدريب وتراجع مكتوبة."])
