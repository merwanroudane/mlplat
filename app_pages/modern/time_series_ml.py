import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset
from utils.plotting import bars, plot

page_header("time_series_ml")

ts = load_dataset("timeseries")
fig = go.Figure(go.Scatter(x=ts["date"], y=ts["sales"], mode="lines", line=dict(color=PALETTE["sky"])))
fig.update_layout(title="Daily sales: trend + weekly & yearly seasonality + promotions + AR(1) noise", height=300)
plot(fig)

st.markdown("## من سلسلة إلى جدول: خصائص زمنية")
comparison_table([
    {"الخاصية": "Lag", "مثال": "sales(t−1)، sales(t−7)", "آمنة؟": "نعم إن كانت ≥ أفق التنبؤ"},
    {"الخاصية": "Rolling", "مثال": "متوسط آخر 7 أيام حتى t−1", "آمنة؟": "نعم مع shift(1) قبل rolling"},
    {"الخاصية": "Calendar / seasonal", "مثال": "يوم الأسبوع، الشهر، sin/cos للسنة", "آمنة؟": "نعم (معروفة مسبقًا)"},
    {"الخاصية": "Known future covariates", "مثال": "عرض ترويجي مخطط", "آمنة؟": "نعم إن كان معروفًا فعلًا وقت التنبؤ"},
    {"الخاصية": "Rolling بلا shift", "مثال": "متوسط يشمل t", "آمنة؟": "❌ تسرّب زمني"},
])


def make_features(df: pd.DataFrame, leaky: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    s = df["sales"]
    for lag in (1, 2, 7, 14):
        out[f"lag_{lag}"] = s.shift(lag)
    base = s if leaky else s.shift(1)
    out["roll_mean_7"] = base.rolling(7).mean()
    out["roll_std_7"] = base.rolling(7).std()
    out["dow"] = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    out["sin_year"], out["cos_year"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["promo"] = df["promo"]
    out["t"] = np.arange(len(df))
    return out


leaky = st.toggle("أدخل خطأً شائعًا: rolling يتضمن اليوم الحالي (تسرّب)", value=False, key="ts_leaky")
F = make_features(ts, leaky=leaky)
data = pd.concat([F, ts["sales"].rename("y")], axis=1).dropna()
st.code("""feats["lag_1"] = sales.shift(1)
feats["roll_mean_7"] = sales.shift(1).rolling(7).mean()     # shift FIRST, then roll — otherwise today's value leaks
feats["dow"] = date.dt.dayofweek""", language="python")

st.markdown("## Walk-forward Lab")
n_splits = st.slider("عدد نوافذ Walk-forward", 3, 8, 5, key="ts_splits")
gap = st.slider("gap (أيام بين التدريب والتحقق)", 0, 14, 0, key="ts_gap")
splits = list(TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=60).split(data))


def _frame(i: int) -> None:
    fig = go.Figure()
    for k, (tr, te) in enumerate(splits[: i + 1]):
        fig.add_trace(go.Scatter(x=data.index[tr], y=[k] * len(tr), mode="lines", line=dict(color=PALETTE["sky"], width=10),
                                 showlegend=k == 0, name="train"))
        fig.add_trace(go.Scatter(x=data.index[te], y=[k] * len(te), mode="lines", line=dict(color=PALETTE["coral"], width=10),
                                 showlegend=k == 0, name="validate (future)"))
    fig.update_layout(title=f"Expanding-window walk-forward: split {i + 1}", yaxis=dict(autorange="reversed", title="split"),
                      xaxis_title="time index", height=120 + 40 * len(splits))
    plot(fig)


stepper(f"ts_wf_{n_splits}_{gap}", len(splits), _frame, labels=[f"split {i + 1}" for i in range(len(splits))])


@st.cache_data(show_spinner="يقيّم بـWalk-forward…", max_entries=16)
def _evaluate(leaky: bool, n_splits: int, gap: int):
    F = make_features(ts, leaky=leaky)
    d = pd.concat([F, ts["sales"].rename("y")], axis=1).dropna()
    X, yv = d.drop(columns="y"), d["y"]
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=60)
    out = {}
    for name, m in (("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)),
                    ("Ridge", RidgeCV(np.logspace(-2, 3, 20)))):
        out[name] = -cross_val_score(m, X, yv, cv=tss, scoring="neg_mean_absolute_error").mean()
    naive, snaive = [], []
    s = yv.to_numpy()
    lag1, lag7 = X["lag_1"].to_numpy(), X["lag_7"].to_numpy()
    for tr, te in tss.split(X):
        naive.append(np.mean(np.abs(s[te] - lag1[te])))
        snaive.append(np.mean(np.abs(s[te] - lag7[te])))
    out["Naive (yesterday)"] = float(np.mean(naive))
    out["Seasonal naive (last week)"] = float(np.mean(snaive))
    out["HGB with random KFold (WRONG)"] = -cross_val_score(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
                                                            X, yv, cv=KFold(5, shuffle=True, random_state=0),
                                                            scoring="neg_mean_absolute_error").mean()
    return out


res = _evaluate(leaky, n_splits, gap)
order = sorted(res, key=res.get)
plot(bars(order, [res[k] for k in order], title="Walk-forward MAE (lower is better)", horizontal=True,
          color=[PALETTE["coral"] if "WRONG" in k else PALETTE["sky"] for k in order], text_fmt=".2f"), height=320)
if leaky:
    warning("مع الخطأ المُدخل انخفض الخطأ بشكل مريب: متوسط rolling يتضمن قيمة اليوم المراد التنبؤ به.")
st.caption("KFold العشوائي يعطي خطأً أقل زيفًا لأنه يتدرّب على «المستقبل» ويختبر على «الماضي» المجاور. Walk-forward يحاكي النشر.")
why("قارن دائمًا مع Naive وSeasonal naive.", "نموذج شجري معقد لا يتفوق على «قيمة الأسبوع الماضي» لا يستحق النشر.")
intuition("النماذج الشجرية لا تستقرئ الاتجاه (تتنبأ داخل مدى التدريب)؛ لذلك نضيف t أو ننمذج الفروق، أو نستخدم نموذجًا خطيًا للاتجاه.")

if at_least("advanced"):
    st.markdown("## متقدم: التنبؤ متعدد الخطوات")
    comparison_table([
        {"الاستراتيجية": "Recursive", "الفكرة": "تنبأ بخطوة ثم استخدمها كـlag للتالية", "عيب": "تراكم الخطأ"},
        {"الاستراتيجية": "Direct", "الفكرة": "نموذج منفصل لكل أفق h", "عيب": "نماذج كثيرة"},
        {"الاستراتيجية": "Multi-output", "الفكرة": "نموذج واحد يخرج كل الآفاق", "عيب": "افتراض بنية مشتركة"},
    ])
    st.markdown("مع أفق h يجب أن تكون كل الـlags ≥ h، واستخدم gap ≥ h − 1 في TimeSeriesSplit.")
if at_least("research"):
    researcher_note(["هذه المنصة لا تعيد مقرر الاقتصاد القياسي للسلاسل (ARIMA، التكامل المشترك)؛ ركّزنا على ML بخصائص Lag.",
                     "لمقارنة دقة التنبؤ إحصائيًا: اختبار Diebold–Mariano؛ وأبلغ عن MASE لقابلية المقارنة بين السلاسل."])
mistakes(["Rolling دون shift.", "KFold عشوائي للسلاسل الزمنية.", "lags أقصر من أفق التنبؤ.", "تجاهل Seasonal naive."])
page_footer("time_series_ml",
            takeaways=["حوّل السلسلة إلى جدول بخصائص Lag/Rolling/Calendar دون تسرب.", "قيّم بـWalk-forward (+ gap).",
                       "Seasonal naive خط أساس إلزامي."])
