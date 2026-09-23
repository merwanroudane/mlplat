import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (d2_absolute_error_score, mean_absolute_error, mean_poisson_deviance,
                             r2_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils import metrics as M
from utils.datasets import xy
from utils.plotting import plot

page_header("regression_metrics")

formula(r"\text{MAE}=\tfrac1n\sum|y_i-\hat y_i|,\ \ \text{RMSE}=\sqrt{\tfrac1n\sum(y_i-\hat y_i)^2},\ \ "
        r"R^2 = 1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}",
        title="Core regression metrics",
        intuition="MAE بوحدة y ومتين نسبيًا؛ RMSE بوحدة y ويعاقب الأخطاء الكبيرة؛ R² نسبة التباين المفسَّر مقارنة بالتنبؤ بالمتوسط.",
        example="أخطاء (1, 1, 1, 9): MAE = 3، RMSE = √(84/4) = 4.58 — خطأ واحد كبير يرفع RMSE كثيرًا.")
comparison_table([
    {"المقياس": "MAE", "الوحدة": "y", "حساسية للشواذ": "منخفضة", "ملاحظة": "هدفه الوسيط"},
    {"المقياس": "MSE / RMSE", "الوحدة": "y² / y", "حساسية للشواذ": "عالية", "ملاحظة": "هدفه المتوسط"},
    {"المقياس": "R²", "الوحدة": "بلا وحدة", "حساسية للشواذ": "عالية", "ملاحظة": "قد يكون سالبًا على الاختبار"},
    {"المقياس": "Adjusted R²", "الوحدة": "بلا وحدة", "حساسية للشواذ": "عالية", "ملاحظة": "يعاقب p داخل العينة؛ لا يغني عن التحقق"},
    {"المقياس": "MAPE", "الوحدة": "%", "حساسية للشواذ": "—", "ملاحظة": "ينفجر قرب y = 0 ويعاقب التقدير الزائد أكثر"},
    {"المقياس": "RMSLE", "الوحدة": "log", "حساسية للشواذ": "منخفضة", "ملاحظة": "أخطاء نسبية؛ y ≥ 0"},
    {"المقياس": "Poisson/Gamma/Tweedie deviance", "الوحدة": "—", "حساسية للشواذ": "حسب التوزيع", "ملاحظة": "بيانات العدّ/الموجبة"},
    {"المقياس": "D² (d2_*_score)", "الوحدة": "بلا وحدة", "حساسية للشواذ": "حسب الخسارة", "ملاحظة": "تعميم R² لأي Deviance"},
])

st.markdown("## Residual Explorer")
X, y = xy("regression")
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0)
model_name = st.segmented_control("النموذج", ["LinearRegression", "HistGradientBoosting"], default="LinearRegression",
                                  key="rm_model", required=True)
outl = st.slider("أضف أخطاء كبيرة في y لنسبة من الاختبار", 0.0, 0.2, 0.0, 0.02, key="rm_out")


@st.cache_data(show_spinner=False)
def _pred(model_name: str):
    m = LinearRegression() if model_name == "LinearRegression" else HistGradientBoostingRegressor(random_state=0)
    return m.fit(Xa, ya).predict(Xb)


pred = _pred(model_name)
yt = yb.to_numpy().copy()
k = int(outl * len(yt))
if k:
    yt[np.random.default_rng(0).choice(len(yt), k, replace=False)] += 25
res = yt - pred
with st.container(horizontal=True):
    st.metric("MAE", f"{mean_absolute_error(yt, pred):.3f}", border=True)
    st.metric("RMSE", f"{root_mean_squared_error(yt, pred):.3f}", border=True)
    st.metric("R²", f"{r2_score(yt, pred):.3f}", border=True)
    st.metric("Adj. R²", f"{M.adjusted_r2(yt, pred, X.shape[1]):.3f}", border=True)
    st.metric("D² (absolute)", f"{d2_absolute_error_score(yt, pred):.3f}", border=True)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Scatter(x=pred, y=yt, mode="markers", marker=dict(color=PALETTE["sky"], opacity=0.6)))
    lo, hi = float(min(pred.min(), yt.min())), float(max(pred.max(), yt.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PALETTE["muted"], dash="dot")))
    fig.update_layout(title="Predicted vs actual", xaxis_title="ŷ", yaxis_title="y", showlegend=False, height=340)
    plot(fig)
with c2:
    fig = go.Figure(go.Histogram(x=res, nbinsx=40, marker_color=PALETTE["coral"]))
    fig.update_layout(title="Residual distribution", xaxis_title="y − ŷ", height=340)
    plot(fig)
if k:
    warning(f"{k} أخطاء كبيرة رفعت RMSE أكثر بكثير من MAE: RMSE/MAE = {root_mean_squared_error(yt, pred) / mean_absolute_error(yt, pred):.2f}.")
st.caption("تحقق من التنفيذ: " + f"utils.metrics.rmse = {M.rmse(yt, pred):.4f} · sklearn = {root_mean_squared_error(yt, pred):.4f}")

st.markdown("## حدود MAPE")
yy = np.array([0.1, 1, 10, 100])
pp = yy + 1
st.dataframe(pd.DataFrame({"y": yy, "ŷ = y + 1": pp, "absolute error": np.abs(yy - pp),
                           "APE": np.abs(yy - pp) / yy}).style.format({"APE": "{:.0%}"}), hide_index=True)
intuition("الخطأ المطلق نفسه (1) يساوي 1000% عند y = 0.1 و1% عند y = 100. MAPE لا يصلح حين تقترب y من الصفر. "
          "في scikit-learn، `mean_absolute_percentage_error` يعيد كسرًا (0.1 = 10%) لا نسبة مئوية.")
cnt = np.random.default_rng(1).poisson(3, 500)
st.caption(f"مثال عدّ: Poisson deviance للتنبؤ بالمتوسط = {mean_poisson_deviance(cnt, np.full(500, cnt.mean())):.3f} "
           f"(تنفيذنا: {M.mean_poisson_deviance(cnt, np.full(500, cnt.mean())):.3f}).")
why("اختر المقياس بوحدة تفهمها الجهة المستفيدة وبدالة تكلفة تشبه الواقع.",
    "RMSE مناسب إذا كانت تكلفة الخطأ تربيعية؛ MAE إذا كانت خطية؛ Pinball إذا كانت غير متماثلة.")

if at_least("advanced"):
    st.markdown("## متقدم: R² على الاختبار")
    st.markdown("R² خارج العينة = 1 − MSE(model)/MSE(متوسط التدريب أو الاختبار)؛ قد يكون **سالبًا** إذا كان النموذج أسوأ من "
                "التنبؤ بالمتوسط. `r2_score` في scikit-learn يستخدم متوسط y_true (الاختبار).")
if at_least("research"):
    researcher_note(["للاستدلال على الفرق بين نموذجين: اختبار Diebold–Mariano للسلاسل الزمنية أو Bootstrap مزدوج للبيانات المستقلة.",
                     "أبلغ عن مقياسين على الأقل (مثل MAE وRMSE) لأنهما يلتقطان جوانب مختلفة من توزيع الأخطاء."])
mistakes(["MAPE مع y قرب الصفر.", "R² وحده دون مقياس بوحدة y.", "تفسير R² مرتفع كدليل على صحة النموذج السببي."])
page_footer("regression_metrics",
            takeaways=["MAE متين، RMSE يعاقب الكبير، R² نسبي للمتوسط.", "MAPE ينهار قرب الصفر.",
                       "اختر المقياس من دالة التكلفة الحقيقية."])
