import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from components.callouts import definition, intuition, mistakes, researcher_note, why
from components.formulas import formula
from config import MAX_MC_REPS
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import hist, plot

page_header("conformal_prediction")

st.info("**لماذا أُضيفت هذه الوحدة؟** (بند 89 من المواصفات: إضافات من البحث) — التنبؤ المطابق أصبح جزءًا من المقررات الحديثة "
        "لتقدير عدم اليقين لأنه يعطي فترات بتغطية مضمونة لأي نموذج دون افتراضات توزيعية، ويتكامل مع أي Pipeline. "
        "المتطلبات: Train/Validation، مقاييس الانحدار. الموضع: بعد التقييم وقبل الإنتاج.", icon=":material/add_circle:")
definition("Split conformal prediction", "درّب أي نموذج على جزء، واحسب «درجات عدم المطابقة» |yᵢ − ŷᵢ| على جزء معايرة منفصل، "
           "ثم ابنِ لكل حالة جديدة الفترة ŷ ± q̂ حيث q̂ مئين مصحح لهذه الدرجات.")
formula(r"\hat q = \text{the } \Big\lceil (n_{cal}+1)(1-\alpha)\Big\rceil\text{-th smallest of } \{|y_i-\hat f(x_i)|\}_{i\in cal},\qquad "
        r"C(x) = \big[\hat f(x)-\hat q,\ \hat f(x)+\hat q\big]",
        title="Split conformal interval (Vovk et al., 2005; Lei et al., 2018)",
        symbols={r"\alpha": "مستوى الخطأ المسموح (0.1 ⇒ تغطية 90%)", "n_{cal}": "حجم مجموعة المعايرة"},
        intuition="إن كانت البيانات قابلة للتبادل (Exchangeable)، فالدرجة الجديدة «واحدة من» درجات المعايرة في رتبة عشوائية؛ "
                  "لذا P(|y − ŷ| ≤ q̂) ≥ 1 − α بالضبط في العينة المحدودة.",
        example="n_cal = 99، α = 0.1 ⇒ q̂ = الدرجة رقم ⌈100·0.9⌉ = 90 بعد الفرز.")

st.markdown("## Conformal Coverage Lab")
c1, c2, c3, c4 = st.columns(4)
alpha = c1.select_slider("α", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key="cp_alpha")
model_name = c2.selectbox("النموذج", ["LinearRegression", "RandomForest", "GradientBoosting"], key="cp_model")
hetero = c3.toggle("ضجيج غير متجانس", value=True, key="cp_het")
n_cal = c4.select_slider("n_cal", [20, 50, 100, 300, 1000], value=300, key="cp_ncal")


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 5, n)
    sd = 0.2 + 0.4 * x if hetero else np.full(n, 0.8)
    y = np.sin(x) * 2 + x + rng.normal(scale=sd)
    return x[:, None], y


def _model():
    return {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(100, min_samples_leaf=5, random_state=0,
                                                                                          n_jobs=1),
            "GradientBoosting": GradientBoostingRegressor(random_state=0)}[model_name]


Xtr, ytr = _data(600, 1)
Xcal, ycal = _data(n_cal, 2)
Xte, yte = _data(2000, 3)
m = _model().fit(Xtr, ytr)
scores = np.sort(np.abs(ycal - m.predict(Xcal)))
k = int(np.ceil((n_cal + 1) * (1 - alpha)))
q = scores[min(k, n_cal) - 1] if k <= n_cal else np.inf
pred = m.predict(Xte)
cover = np.mean(np.abs(yte - pred) <= q)
xs = np.linspace(0, 5, 200)[:, None]
px = m.predict(xs)
fig = go.Figure(go.Scatter(x=Xte[:400, 0], y=yte[:400], mode="markers", name="test points",
                           marker=dict(color=PALETTE["muted"], opacity=0.45, size=5)))
fig.add_trace(go.Scatter(x=np.r_[xs[:, 0], xs[::-1, 0]], y=np.r_[px + q, (px - q)[::-1]], fill="toself", name=f"{1 - alpha:.0%} interval",
                         fillcolor="rgba(112,72,232,0.15)", line=dict(color="rgba(0,0,0,0)")))
fig.add_trace(go.Scatter(x=xs[:, 0], y=px, name="prediction", line=dict(color=PALETTE["purple"], width=3)))
fig.update_layout(title=f"Split conformal: q̂ = {q:.2f} · empirical test coverage = {cover:.1%} (target ≥ {1 - alpha:.0%})", height=420)
plot(fig)
if hetero:
    st.caption("مع ضجيج غير متجانس، الفترة ذات العرض الثابت تغطي أكثر من اللازم يسارًا وأقل من اللازم يمينًا؛ التغطية مضمونة "
               "**هامشيًا** (في المتوسط على X) لا شرطيًا لكل x. الحل: Conformalized Quantile Regression أو درجات مطبَّعة.")


@st.cache_data(show_spinner="يكرر التجربة…", max_entries=16)
def _repeat(alpha: float, n_cal: int, reps: int, hetero: bool):
    covs = []
    base = LinearRegression().fit(Xtr, ytr)
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        x = rng.uniform(0, 5, n_cal + 500)[:, None]
        sd = 0.2 + 0.4 * x[:, 0] if hetero else np.full(len(x), 0.8)
        yy = np.sin(x[:, 0]) * 2 + x[:, 0] + rng.normal(scale=sd)
        s = np.sort(np.abs(yy[:n_cal] - base.predict(x[:n_cal])))
        kk = int(np.ceil((n_cal + 1) * (1 - alpha)))
        qq = s[min(kk, n_cal) - 1]
        covs.append(np.mean(np.abs(yy[n_cal:] - base.predict(x[n_cal:])) <= qq))
    return np.array(covs)


covs = _repeat(alpha, n_cal, MAX_MC_REPS, hetero)
fig = hist(covs, title=f"Coverage over {MAX_MC_REPS} repeated calibration sets (linear model)", color=PALETTE["teal"])
fig.add_vline(x=1 - alpha, line=dict(color=PALETTE["coral"], dash="dash"), annotation_text="1 − α")
plot(fig, height=300)
st.caption(f"متوسط التغطية = {covs.mean():.3f} ≥ {1 - alpha}. مع n_cal صغير تتقلب التغطية أكثر (توزيعها Beta)، لكن الضمان في المتوسط يبقى.")
why("استخدم مجموعة معايرة منفصلة عن التدريب.", "الضمان يعتمد على أن درجات المعايرة وتلك الجديدة قابلة للتبادل؛ بواقي التدريب "
    "أصغر من الحقيقية فتعطي فترات ضيقة جدًا.")
intuition("الفترة المطابقة تغلّف أي نموذج — حتى نموذجًا سيئًا — بتغطية صحيحة؛ النموذج الجيد يعطي فترات **أضيق** فقط.")

if at_least("advanced"):
    st.markdown("## متقدم: التصنيف")
    st.markdown("لكل فئة درجة 1 − p̂_y(x)؛ مجموعة التنبؤ = كل الفئات التي درجتها ≤ q̂. تحتوي الفئة الحقيقية باحتمال ≥ 1 − α، "
                "ويكون حجم المجموعة مقياسًا طبيعيًا لعدم اليقين.")
if at_least("research"):
    researcher_note(["الضمان هامشي؛ التغطية الشرطية التامة مستحيلة دون افتراضات (Lei et al., 2018).",
                     "مع الانجراف تسقط قابلية التبادل؛ امتدادات Weighted conformal تعالج Covariate shift.",
                     "Cross-conformal وJackknife+ يستخدمان البيانات بكفاءة أكبر من Split."])
    st.markdown(cite("vovk2005", "lei2018", "angelopoulos2021"))
mistakes(["استخدام بواقي التدريب للمعايرة.", "ادعاء تغطية شرطية لكل x.", "تطبيقه مع بيانات زمنية منجرفة دون تعديل."])
page_footer("conformal_prediction",
            takeaways=["Split conformal يعطي فترات بتغطية ≥ 1 − α لأي نموذج.", "الضمان هامشي ويتطلب قابلية التبادل.",
                       "النموذج الأفضل = فترات أضيق لا تغطية أعلى."])
