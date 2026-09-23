import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from components.callouts import intuition, mistakes, researcher_note
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import hist, plot

page_header("statistics_ml")

st.markdown("## المجتمع والعينة والمقدِّر")
formula(r"\operatorname{Bias}(\hat\theta) = \mathbb E[\hat\theta]-\theta,\qquad "
        r"\operatorname{MSE}(\hat\theta) = \operatorname{Bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta)",
        title="Bias, variance and MSE of an estimator",
        symbols={r"\theta": "المعلمة الحقيقية في المجتمع", r"\hat\theta": "المقدِّر (دالة في العينة، متغير عشوائي)"},
        intuition="مقدِّر متحيز قليلًا بتباين أقل قد يكون أفضل (MSE أقل) — هذه فكرة Ridge وكل التنظيم.",
        example="S² بمقام n متحيز (يقلل التباين)، وبمقام n−1 غير متحيز.")

st.markdown("## مختبر: تحيز المقدِّر وتباينه بالمحاكاة")
c1, c2, c3 = st.columns(3)
n = c1.slider("حجم العينة n", 5, 200, 20, 5, key="st_n")
reps = c2.slider("عدد العينات", 100, 2000, 1000, 100, key="st_reps")
est = c3.segmented_control("المقدِّر", ["mean", "median", "var (n)", "var (n−1)"], default="var (n)", key="st_est",
                           required=True)
rng = np.random.default_rng(0)
samples = rng.normal(10, 3, size=(reps, n))
vals = {"mean": samples.mean(1), "median": np.median(samples, 1), "var (n)": samples.var(1, ddof=0),
        "var (n−1)": samples.var(1, ddof=1)}[est]
truth = 9.0 if est.startswith("var") else 10.0
fig = hist(vals, title=f"Sampling distribution of {est} (n = {n})", color=PALETTE["sky"])
fig.add_vline(x=truth, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text="truth")
fig.add_vline(x=vals.mean(), line=dict(color=PALETTE["coral"]), annotation_text="E[estimator]", annotation_position="bottom right")
c1, c2 = st.columns([1.6, 1])
with c1:
    plot(fig, height=340)
with c2:
    st.metric("Bias", f"{vals.mean() - truth:+.3f}")
    st.metric("SD (= standard error)", f"{vals.std(ddof=1):.3f}")
    st.metric("MSE", f"{np.mean((vals - truth) ** 2):.3f}")
intuition("الخطأ المعياري هو الانحراف المعياري لتوزيع المعاينة: كم سيتغير التقدير لو أعدنا جمع البيانات.")

st.markdown("## Bootstrap: عدم اليقين دون صيغ")
formula(r"\hat\theta^{*b} = s\big(\text{sample with replacement}\big),\ b = 1..B;\qquad "
        r"\widehat{SE} = \operatorname{sd}(\hat\theta^{*1},\dots,\hat\theta^{*B})",
        title="Nonparametric bootstrap (Efron, 1979)",
        intuition="العينة تقوم مقام المجتمع؛ نعيد السحب منها لنرى تقلب الإحصائية.",
        example="فترة percentile 95%: المئينان 2.5 و97.5 لقيم θ̂*.")
st.markdown("### مختبر Bootstrap: فترة ثقة لـROC-AUC على مجموعة اختبار")
c1, c2 = st.columns(2)
n_test = c1.slider("حجم الاختبار", 50, 2000, 300, 50, key="bs_n")
B = c2.slider("B (عدد إعادات السحب)", 100, 2000, 500, 100, key="bs_B")


@st.cache_data(show_spinner=False, max_entries=32)
def _boot_auc(n_test: int, B: int):
    r = np.random.default_rng(1)
    y = r.integers(0, 2, n_test)
    score = y * 0.9 + r.normal(size=n_test)
    auc = roc_auc_score(y, score)
    boots = []
    for _ in range(B):
        i = r.integers(0, n_test, n_test)
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], score[i]))
    return auc, np.array(boots)


auc, boots = _boot_auc(n_test, B)
lo, hi = np.percentile(boots, [2.5, 97.5])
fig = hist(boots, title=f"Bootstrap distribution of test AUC — 95% CI [{lo:.3f}, {hi:.3f}]", color=PALETTE["purple"])
fig.add_vline(x=auc, line=dict(color=PALETTE["coral"], dash="dash"), annotation_text=f"AUC = {auc:.3f}")
plot(fig, height=320)
st.caption("مع اختبار صغير، فترة AUC عريضة: فرق 0.01 بين نموذجين غالبًا غير ذي معنى.")

st.markdown("## الارتباط والتعدد الخطي")
st.caption("خاصيتان مرتبطتان بقوة ρ؛ نكرر جمع البيانات ونقدّر معاملات الانحدار. لاحظ تضخم التباين.")
rho = st.slider("ρ بين x1 وx2", 0.0, 0.99, 0.9, 0.01, key="st_rho")


@st.cache_data(show_spinner=False)
def _vif_sim(rho: float, reps: int = 300, n: int = 100):
    r = np.random.default_rng(2)
    cov = np.array([[1, rho], [rho, 1]])
    coefs = []
    for _ in range(reps):
        X = r.multivariate_normal([0, 0], cov, size=n)
        y = 1.0 * X[:, 0] + 1.0 * X[:, 1] + r.normal(size=n)
        coefs.append(LinearRegression().fit(X, y).coef_)
    return np.array(coefs)


coefs = _vif_sim(rho)
fig = go.Figure(go.Scatter(x=coefs[:, 0], y=coefs[:, 1], mode="markers", marker=dict(color=PALETTE["sky"], opacity=0.5)))
fig.add_trace(go.Scatter(x=[1], y=[1], mode="markers", marker=dict(symbol="star", size=16, color=PALETTE["coral"]), name="truth"))
fig.update_layout(title=f"β̂₁ vs β̂₂ over 300 samples · VIF = 1/(1−ρ²) = {1 / (1 - rho ** 2):.1f}", xaxis_title="β̂₁",
                  yaxis_title="β̂₂", height=380, showlegend=False)
plot(fig)
st.markdown("المعاملان منفردَين غير مستقرين، لكن **مجموعهما** (أثرهما المشترك) مستقر: التعدد الخطي يضر التفسير لا التنبؤ "
            "بالضرورة.")

if at_least("advanced"):
    st.markdown("## متقدم: الخطأ المعياري في DML يأتي من الدرجة")
    st.latex(r"\hat\sigma^2 = \frac{\frac1n\sum_i \psi(W_i;\hat\theta,\hat\eta)^2}{\big(\frac1n\sum_i \psi^a(W_i;\hat\eta)\big)^2},"
             r"\qquad \widehat{SE} = \hat\sigma/\sqrt n")
    st.markdown("هذا تقدير Sandwich للتباين التقاربي لمقدِّر Z؛ صالح لأن الدرجة متعامدة والإزعاج مقدَّر بـCross-fitting.")
if at_least("research"):
    researcher_note(["Bootstrap يفشل لبعض الإحصائيات غير الملساء (الحد الأقصى) وللبيانات المعتمدة؛ استخدم Block bootstrap للزمن.",
                     "في DoubleML، bootstrap() يحسب فترات مشتركة (Multiplier bootstrap) للمعالجات المتعددة."])
    st.markdown(cite("efron1979"))
mistakes(["اعتبار تقدير نقطي بلا خطأ معياري «نتيجة».", "تفسير معاملات منفردة لخصائص شديدة الترابط.",
          "Bootstrap لبيانات مجمّعة بسحب الصفوف بدل المجموعات."])
page_footer("statistics_ml",
            takeaways=["المقدِّر متغير عشوائي له تحيز وتباين.", "Bootstrap يقدّر عدم اليقين لأي إحصائية تقريبًا.",
                       "التعدد الخطي يضخم تباين المعاملات لا التنبؤات."])
