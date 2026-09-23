import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("lda_qda")
algorithm_profile("lda")

formula(r"x\mid y=k \sim \mathcal N(\mu_k, \Sigma_k),\qquad P(y=k\mid x)\propto \pi_k\,\mathcal N(x;\mu_k,\Sigma_k)",
        title="Generative model for each class",
        symbols={r"\pi_k": "الأولوية (prior)", r"\mu_k": "متوسط الفئة", r"\Sigma_k": "تغاير الفئة"})
formula(r"\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k",
        title="LDA discriminant (shared Σ ⇒ linear in x)",
        intuition="حين تتشارك الفئات مصفوفة التغاير، يلغي الحد التربيعي xᵀΣ⁻¹x نفسه بين الفئات ⇒ حد خطي. مع Σ_k مختلفة "
                  "(QDA) يبقى ⇒ حد تربيعي.")

st.markdown("## مختبر الحدود: LDA مقابل QDA")
c1, c2, c3 = st.columns(3)
rot = c1.slider("دوران تغاير الفئة 2 (درجات)", 0, 90, 60, 5, key="lq_rot")
stretch = c2.slider("استطالة تغاير الفئة 2", 1.0, 6.0, 4.0, 0.5, key="lq_str")
reg = c3.slider("QDA reg_param", 0.0, 1.0, 0.0, 0.05, key="lq_reg")
rng = np.random.default_rng(0)
a = np.deg2rad(rot)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
S2 = R @ np.diag([stretch, 0.4]) @ R.T
X = np.vstack([rng.multivariate_normal([0, 0], np.eye(2), 200), rng.multivariate_normal([2.5, 1.0], S2, 200)])
y = np.r_[np.zeros(200, int), np.ones(200, int)]
lda = LinearDiscriminantAnalysis().fit(X, y)
qda = QuadraticDiscriminantAnalysis(reg_param=reg).fit(X, y)
c1, c2 = st.columns(2)
with c1:
    boundary_chart(lda, X, y, title=f"LDA · CV acc {cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean():.3f}", height=380)
with c2:
    boundary_chart(qda, X, y, title=f"QDA · CV acc {cross_val_score(QuadraticDiscriminantAnalysis(reg_param=reg), X, y, cv=5).mean():.3f}",
                   height=380)
intuition("حين تختلف أشكال الفئات (دوران/استطالة) يتفوق QDA. مع بيانات قليلة وأبعاد كثيرة يتفوق LDA لأنه يقدّر معاملات أقل: "
          "p(p+1)/2 للتغاير المشترك مقابل K·p(p+1)/2.")

st.markdown("## LDA كتقليل أبعاد موجَّه")
from sklearn.datasets import load_wine  # noqa: E402
wine = load_wine()
Z = LinearDiscriminantAnalysis(n_components=2).fit_transform(wine.data, wine.target)
fig = go.Figure()
for k, (color, sym) in enumerate(zip([PALETTE["sky"], PALETTE["coral"], PALETTE["purple"]], ["circle", "diamond", "square"])):
    m = wine.target == k
    fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=f"cultivar {k}",
                             marker=dict(color=color, symbol=sym, size=8)))
fig.update_layout(title="Wine (13 features) projected on 2 LDA directions (K − 1 = 2 max)", xaxis_title="LD1",
                  yaxis_title="LD2", height=380)
plot(fig)
st.caption("على عكس PCA (غير موجَّه يعظّم التباين)، LDA يعظّم فصل الفئات: نسبة التباين بين الفئات إلى داخلها.")

comparison_table([
    {"": "Logistic", "النوع": "تمييزي", "الحد": "خطي", "الافتراضات": "خطية log-odds", "بيانات قليلة": "متوسط",
     "الشواذ": "أكثر متانة"},
    {"": "LDA", "النوع": "توليدي", "الحد": "خطي", "الافتراضات": "طبيعية + تغاير مشترك", "بيانات قليلة": "جيد (أكفأ إن صح الافتراض)",
     "الشواذ": "حساس"},
    {"": "QDA", "النوع": "توليدي", "الحد": "تربيعي", "الافتراضات": "طبيعية لكل فئة", "بيانات قليلة": "ضعيف (معاملات كثيرة)",
     "الشواذ": "حساس"},
    {"": "Naive Bayes", "النوع": "توليدي", "الحد": "تربيعي محوري", "الافتراضات": "استقلال شرطي", "بيانات قليلة": "ممتاز",
     "الشواذ": "متوسط"},
])
hyperparameter_table("LinearDiscriminantAnalysis")
hyperparameter_table("QuadraticDiscriminantAnalysis")
st.caption("منذ scikit-learn 1.8 أُضيفت solver وshrinkage وcovariance_estimator إلى QDA (تحقق من التوقيع في 1.9.1).")

if at_least("advanced"):
    st.markdown("## متقدم: Shrinkage مع p قريب من n")
    n_small = st.slider("n لكل فئة", 10, 200, 25, 5, key="lq_n")
    p = 40
    rows = []
    for shrink in (None, "auto"):
        accs = []
        for rep in range(10):
            r = np.random.default_rng(rep)
            Xa = np.vstack([r.normal(0, 1, (n_small, p)), r.normal(0.3, 1, (n_small, p))])
            ya = np.r_[np.zeros(n_small), np.ones(n_small)]
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            accs.append(cross_val_score(m, Xa, ya, cv=5).mean())
        rows.append({"shrinkage": str(shrink), "CV accuracy (10 datasets)": np.mean(accs)})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.markdown("Ledoit–Wolf (`shrinkage='auto'`) يدمج التغاير التجريبي مع مصفوفة قطرية ⇒ استقرار كبير حين p/n كبير.")
if at_least("research"):
    researcher_note(["Efron (1975): إن صح افتراض LDA فهو أكفأ من Logistic بنحو 30–50% في حجم العينة المطلوب؛ إن خطئ فـLogistic أمتن.",
                     "Fisher (1936) قدّم LDA لبيانات Iris — أصل التحليل التمييزي."])
    st.markdown(cite("fisher1936", "esl"))
template_checklist({3: "النموذج التوليدي", 5: "مختبر LDA/QDA", 7: "تقدير المتوسطات والتغاير (حل مغلق)",
                    9: "جداول المعاملات الفائقة", 22: "LinearDiscriminantAnalysis / QDA", 23: "مختبر الحدود"})
mistakes(["QDA مع بيانات قليلة وأبعاد كثيرة.", "shrinkage مع solver='svd' (غير مدعوم).", "الخلط بين LDA التصنيف وLDA نماذج الموضوعات."])
page_footer("lda_qda",
            takeaways=["LDA/QDA تنمذج توزيع كل فئة ثم تطبق Bayes.", "تغاير مشترك ⇒ خطي، منفصل ⇒ تربيعي.",
                       "LDA أيضًا تقليل أبعاد موجَّه (K − 1 مكونًا)."])
