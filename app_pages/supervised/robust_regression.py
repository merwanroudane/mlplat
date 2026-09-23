import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, RANSACRegressor, TheilSenRegressor

from components.algorithm_profile import algorithm_profile
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import SEQUENCE
from utils.plotting import plot

page_header("robust_regression")
algorithm_profile("huber")

st.markdown("## مختبر المتانة أمام الشواذ")
c1, c2, c3 = st.columns(3)
frac = c1.slider("نسبة الشواذ", 0.0, 0.4, 0.15, 0.05, key="rr_frac")
kind = c2.segmented_control("نوع الشذوذ", ["في y (vertical)", "رافعة في x"], default="في y (vertical)", key="rr_kind",
                            required=True)
seed = c3.number_input("seed", 0, 100, 0, key="rr_seed")


@st.cache_data(show_spinner=False)
def _fit_all(frac: float, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.uniform(0, 10, n)
    y = 1 + 2 * x + rng.normal(size=n)
    k = int(frac * n)
    idx = rng.choice(n, k, replace=False)
    if kind.startswith("في y"):
        y[idx] = y[idx] + rng.uniform(20, 40, k)
    else:
        x[idx] = rng.uniform(18, 25, k)
        y[idx] = rng.uniform(-5, 5, k)
    X = x[:, None]
    g = np.linspace(x.min(), x.max(), 60)[:, None]
    models = {
        "OLS": LinearRegression(),
        "Huber": HuberRegressor(epsilon=1.35, max_iter=1000),
        "RANSAC": RANSACRegressor(random_state=0),
        "Theil-Sen": TheilSenRegressor(random_state=0),
        "Median (τ=0.5)": QuantileRegressor(quantile=0.5, alpha=0, solver="highs"),
    }
    out, slopes = {}, {}
    for name, m in models.items():
        m.fit(X, y)
        out[name] = m.predict(g)
        est = m.estimator_ if name == "RANSAC" else m
        slopes[name] = float(np.ravel(est.coef_)[0])
    return x, y, idx, g.ravel(), out, slopes


x, y, idx, g, preds, slopes = _fit_all(frac, kind, int(seed))
fig = go.Figure()
mask = np.zeros(len(x), bool)
mask[idx] = True
fig.add_trace(go.Scatter(x=x[~mask], y=y[~mask], mode="markers", name="clean", marker=dict(color="#ADB5BD", size=6)))
fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name="outliers", marker=dict(color="#E8590C", symbol="x", size=9)))
for i, (name, p) in enumerate(preds.items()):
    fig.add_trace(go.Scatter(x=g, y=p, mode="lines", name=name, line=dict(color=SEQUENCE[i], width=2.6)))
fig.add_trace(go.Scatter(x=g, y=1 + 2 * g, mode="lines", name="truth", line=dict(color="#212529", dash="dot")))
fig.update_layout(height=440, title="Which estimators resist which outliers?")
plot(fig)
st.dataframe(pd.DataFrame({"estimator": list(slopes), "slope (truth = 2)": list(slopes.values()),
                           "|error|": [abs(v - 2) for v in slopes.values()]}).round(3), hide_index=True, width="stretch")
intuition("Huber وMedian يقاومان الشواذ الرأسية (في y) لأنهما يحدان من وزن البواقي الكبيرة، لكنهما أضعف أمام نقاط الرافعة "
          "(في x). RANSAC وTheil-Sen أكثر مقاومة لنقاط الرافعة بنقطة انهيار أعلى — بتكلفة حسابية أكبر.")

comparison_table([
    {"المقدِّر": "OLS", "نقطة الانهيار": "0% (نقطة واحدة تكفي)", "الفكرة": "مربعات البواقي"},
    {"المقدِّر": "Huber", "نقطة الانهيار": "منخفضة أمام الرافعة", "الفكرة": "تربيعي صغير/خطي كبير؛ epsilon يحدد الحد"},
    {"المقدِّر": "Quantile (τ=0.5)", "نقطة الانهيار": "مقاوم في y", "الفكرة": "القيم المطلقة ⇒ الوسيط الشرطي"},
    {"المقدِّر": "Theil-Sen", "نقطة الانهيار": "≈ 29% (بسيط)", "الفكرة": "وسيط ميول أزواج/مجموعات النقاط"},
    {"المقدِّر": "RANSAC", "نقطة الانهيار": "قد تصل ~50%", "الفكرة": "ملاءمة على عينات عشوائية واختيار أكبر مجموعة متوافقة"},
])
why("لا تحذف الشواذ آليًا قبل النمذجة.",
    "افحص أولًا: خطأ قياس أم حالة حقيقية نادرة؟ المقدِّر المتين يقلل أثرها دون إخفائها، ويبقى تفسيرها قرارًا بشريًا.")

if at_least("advanced"):
    st.markdown("## متقدم: Huber = أوزان متكيفة (IRLS)")
    st.latex(r"\min_{\beta,\sigma}\ \sum_i\Big(\sigma + H_\epsilon\big(\tfrac{y_i-x_i^\top\beta}{\sigma}\big)\sigma\Big) + \alpha\|\beta\|^2")
    st.markdown("HuberRegressor يقدّر المقياس σ مع المعاملات، فيصبح epsilon عديم الأبعاد (افتراضي 1.35 ≈ 95% كفاءة "
                "تحت الطبيعية). `outliers_` يعطي قناع النقاط المعاملة كشاذة.")
if at_least("research"):
    researcher_note(["Huber (1964) أسس نظرية M-estimation: موازنة الكفاءة تحت الطبيعية مع المتانة.",
                     "في DML يمكن استخدام متعلمين متينين للإزعاج، لكن الدرجة المتعامدة نفسها (الخطية في البواقي) ليست متينة؛ "
                     "امتدادات Quantile DML تعالج ذلك."])
    st.markdown(cite("huber1964"))
mistakes(["حذف الشواذ دون تحقيق.", "افتراض أن Huber يحمي من نقاط الرافعة.", "تقييم نموذج متين بـMSE وحده."])
page_footer("robust_regression",
            takeaways=["OLS هش أمام نقطة واحدة.", "Huber/Median للشواذ في y؛ RANSAC/Theil-Sen للرافعة أيضًا.",
                       "المتانة لا تغني عن فحص الشواذ."])
