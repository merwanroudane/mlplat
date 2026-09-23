import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression

from components.algorithm_profile import algorithm_profile, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.models import LinearRegressionGD
from utils.plotting import plot

page_header("linear_regression")
algorithm_profile("linear_regression")

st.markdown("## 1–4. المسألة والحدس والصياغة والرموز")
formula(r"y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i \quad\Longleftrightarrow\quad y = X\beta + \varepsilon",
        title="Multiple linear regression",
        symbols={r"\beta_0": "الحد الثابت (قيمة ŷ حين كل x = 0)", r"\beta_j": "الميل الجزئي للخاصية j",
                 r"\varepsilon_i": "الخطأ العشوائي", "X": "مصفوفة n×(p+1) بعمود واحدات للحد الثابت"},
        intuition="البسيط (p = 1): أفضل خط. المتعدد: أفضل مستوى/مستوى فائق.",
        example="ŷ = 20 + 3·(years_experience) − 1.5·(commute_hours): كل سنة خبرة إضافية ترتبط بـ+3 مع ثبات التنقل.")

st.markdown("## 5–6. الهدف والهندسة: الإسقاط")
formula(r"\hat\beta = \arg\min_\beta \|y - X\beta\|_2^2 = (X^\top X)^{-1}X^\top y,\qquad \hat y = X\hat\beta = Hy,\ H = X(X^\top X)^{-1}X^\top",
        title="Least squares and the hat matrix",
        symbols={"H": "مصفوفة الإسقاط (Hat matrix)", r"e = y-\hat y": "البواقي، عمودية على كل أعمدة X"},
        intuition="ŷ هو الإسقاط العمودي لـy على الفضاء الذي تولّده أعمدة X؛ البواقي هي الجزء العمودي عليه: Xᵀe = 0.",
        example="لذلك مجموع البواقي = 0 (عمودية على عمود الواحدات) ولا ترتبط بأي خاصية في عينة التدريب.")

st.markdown("## مختبر OLS: الخط والبواقي والنقاط المؤثرة")
c1, c2, c3, c4 = st.columns(4)
n = c1.slider("n", 10, 200, 40, 5, key="lr_n")
slope = c2.slider("الميل الحقيقي", -3.0, 3.0, 1.5, 0.25, key="lr_slope")
noise = c3.slider("σ", 0.1, 5.0, 1.0, 0.1, key="lr_noise")
lev = c4.toggle("أضف نقطة رافعة شاذة", value=False, key="lr_lev")
rng = np.random.default_rng(4)
x = rng.uniform(0, 10, n)
y = 2 + slope * x + rng.normal(scale=noise, size=n)
if lev:
    x = np.r_[x, 25.0]
    y = np.r_[y, 2 + slope * 25 - 30]
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
infl = ols.get_influence()
cooks = infl.cooks_distance[0]
yhat = ols.fittedvalues
fig = go.Figure()
for xi, yi, fi in zip(x, y, yhat):
    fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, fi], mode="lines", line=dict(color="rgba(247,103,7,0.45)", width=1),
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data",
                         marker=dict(size=6 + 30 * cooks / max(cooks.max(), 1e-9), color=PALETTE["sky"], opacity=0.75,
                                     line=dict(width=1, color="white"))))
gx = np.linspace(x.min(), x.max(), 50)
fig.add_trace(go.Scatter(x=gx, y=ols.params[0] + ols.params[1] * gx, name="OLS fit", line=dict(color=PALETTE["purple"], width=3)))
fig.add_trace(go.Scatter(x=gx, y=2 + slope * gx, name="truth", line=dict(color=PALETTE["teal"], dash="dash")))
fig.update_layout(title="Residuals (orange) and influence (marker size = Cook's distance)", height=420)
c1, c2 = st.columns([1.7, 1])
with c1:
    plot(fig)
with c2:
    st.metric("β̂₀ (intercept)", f"{ols.params[0]:.3f}", "true 2", delta_color="off")
    st.metric("β̂₁ (slope)", f"{ols.params[1]:.3f}", f"true {slope}", delta_color="off")
    st.metric("R²", f"{ols.rsquared:.3f}")
    st.metric("max Cook's D", f"{cooks.max():.2f}")
    if lev:
        st.warning("نقطة واحدة بعيدة في x (رافعة عالية) وشاذة في y تسحب الخط كله. OLS يعطيها وزنًا تربيعيًا.")

st.markdown("### تشخيص البواقي")
c1, c2 = st.columns(2)
with c1:
    f = go.Figure(go.Scatter(x=yhat, y=ols.resid, mode="markers", marker=dict(color=PALETTE["coral"], opacity=0.7)))
    f.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dot"))
    f.update_layout(title="Residuals vs fitted (look for patterns)", xaxis_title="fitted", yaxis_title="residual", height=300)
    plot(f)
with c2:
    from scipy import stats
    osm, osr = stats.probplot(ols.resid, dist="norm")[0]
    f = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=PALETTE["purple"])))
    f.add_trace(go.Scatter(x=osm, y=osm * np.std(ols.resid, ddof=1) + np.mean(ols.resid), mode="lines",
                           line=dict(color=PALETTE["teal"], dash="dash")))
    f.update_layout(title="Normal Q-Q of residuals", xaxis_title="theoretical quantiles", yaxis_title="sample quantiles",
                    height=300, showlegend=False)
    plot(f)

st.markdown("## 7–8. التدريب والمعاملات المتعلَّمة")
comparison_table([
    {"الطريقة": "Normal equations / QR / SVD", "من يستخدمها؟": "LinearRegression (scipy lstsq)", "ملاحظة": "حل دقيق"},
    {"الطريقة": "Gradient descent", "من يستخدمها؟": "SGDRegressor، التنفيذ من الصفر", "ملاحظة": "للبيانات الكبيرة"},
    {"الطريقة": "statsmodels OLS", "من يستخدمها؟": "الاستدلال (SE, t, CI)", "ملاحظة": "للتفسير الإحصائي"},
])
Xr, yr = xy("regression")


def _code(p):
    return ("from sklearn.linear_model import LinearRegression\nimport statsmodels.api as sm\n\n"
            "lin = LinearRegression().fit(X, y)                  # library (least squares)\n"
            f"gd = LinearRegressionGD(lr={p['lr']}, n_iter={p['n_iter']}).fit(X, y)   # from scratch (utils/models.py)\n"
            "ols = sm.OLS(y, sm.add_constant(X)).fit()            # inference: SE, t, CI\n"
            "print(ols.summary())")


def _run(lr, n_iter):
    lin = LinearRegression().fit(Xr, yr)
    gd = LinearRegressionGD(lr=lr, n_iter=n_iter).fit(Xr.to_numpy(), yr.to_numpy())
    o = sm.OLS(yr, sm.add_constant(Xr)).fit()
    tab = pd.DataFrame({"sklearn coef_": np.r_[lin.intercept_, lin.coef_], "scratch GD": np.r_[gd.intercept_, gd.coef_],
                        "statsmodels": o.params.to_numpy(), "SE": o.bse.to_numpy(),
                        "95% CI low": o.conf_int()[0].to_numpy(), "95% CI high": o.conf_int()[1].to_numpy(),
                        "p-value": o.pvalues.to_numpy()},
                       index=["intercept"] + list(Xr.columns))
    return [tab.round(4), "**الحقيقة:** x1 = 3، x2 = −2، x3 = 1.5، x4 علاقة جيبية (يلتقط OLS جزءها الخطي فقط)، x5..x8 = 0."]


code_lab("ols_code", "OLS three ways: sklearn, from scratch, statsmodels", _code, _run,
         lambda: {"lr": st.select_slider("η (scratch)", [0.01, 0.05, 0.1, 0.2], value=0.1, key="lrc_lr"),
                  "n_iter": st.slider("iterations (scratch)", 50, 2000, 500, 50, key="lrc_it")},
         explanation="scikit-learn للتنبؤ (لا يعطي أخطاء معيارية)، statsmodels للاستدلال. التنفيذ من الصفر يطابق الحل "
                     "مع تكرارات كافية لأن المسألة محدبة.")

st.markdown("## 10. الافتراضات: التنبؤ مقابل الاستدلال")
comparison_table([
    {"الافتراض": "Linearity (in parameters)", "للتنبؤ": "مهم (وإلا تحيز)", "للاستدلال على β": "ضروري", "التشخيص": "Residuals vs fitted"},
    {"الافتراض": "Independence", "للتنبؤ": "لتقدير الأداء بصدق", "للاستدلال على β": "ضروري (وإلا SE خاطئة)",
     "التشخيص": "بنية البيانات؛ Cluster-robust SE"},
    {"الافتراض": "Homoscedasticity", "للتنبؤ": "غير ضروري", "للاستدلال على β": "للـSE الكلاسيكية (استخدم HC)", "التشخيص": "Breusch–Pagan"},
    {"الافتراض": "Normal errors", "للتنبؤ": "غير ضروري", "للاستدلال على β": "للعينات الصغيرة فقط", "التشخيص": "Q-Q plot"},
    {"الافتراض": "No perfect collinearity", "للتنبؤ": "لا يضر كثيرًا", "للاستدلال على β": "ضروري للتعريف", "التشخيص": "VIF، الرتبة"},
    {"الافتراض": "Exogeneity E[ε|X] = 0", "للتنبؤ": "غير ضروري للتنبؤ الترابطي", "للاستدلال على β": "ضروري للتفسير السببي",
     "التشخيص": "لا يُختبر بالبيانات — تصميم"},
])
why("استخدم Robust (HC) standard errors افتراضيًا في الاستدلال.",
    "تجانس التباين نادر في البيانات الحقيقية؛ HC3 يبقى صالحًا دونه (sm.OLS(...).fit(cov_type='HC3')).")
intuition("للتنبؤ نحتاج فقط أن تعمّم العلاقة المتعلَّمة على بيانات جديدة من التوزيع نفسه. للاستدلال السببي نحتاج أكثر "
          "بكثير: غياب المتغيرات المحذوفة المربكة.")

if at_least("advanced"):
    st.markdown("## متقدم: Frisch–Waugh–Lovell (جسر إلى DML)")
    formula(r"\hat\beta_1 = \frac{\sum_i \tilde x_{i1}\tilde y_i}{\sum_i \tilde x_{i1}^2},\quad \tilde x_1 = x_1 - \hat E[x_1\mid x_{-1}],\ "
            r"\tilde y = y - \hat E[y\mid x_{-1}]",
            title="FWL theorem",
            intuition="معامل x₁ في الانحدار المتعدد = انحدار بواقي y على بواقي x₁ بعد إزالة أثر بقية المتغيرات خطيًا. "
                      "DML يعمّم هذا باستبدال الإسقاط الخطي بـML مرن + Cross-fitting.")
    xm = Xr.drop(columns="x1").to_numpy()
    rx = Xr["x1"].to_numpy() - LinearRegression().fit(xm, Xr["x1"]).predict(xm)
    ry = yr.to_numpy() - LinearRegression().fit(xm, yr).predict(xm)
    st.code(f"FWL residual-on-residual slope = {np.sum(rx * ry) / np.sum(rx * rx):.6f}\n"
            f"multiple regression coef_[x1]  = {LinearRegression().fit(Xr, yr).coef_[0]:.6f}", language="text")
    st.markdown("**التعقيد (15):** بناء XᵀX = O(np²)، الحل O(p³)؛ التنبؤ O(p) لكل صف.")
if at_least("research"):
    researcher_note(["R² يقيس التنبؤ داخل العينة لا صحة النموذج؛ R² منخفض لا يعني معاملًا متحيزًا والعكس.",
                     "أبلغ عن SE قوية أمام عدم التجانس (HC3) أو مجمّعة (Cluster) حسب التصميم.",
                     "انتقاء النموذج ثم الاستدلال على العينة نفسها يبطل p-values الاسمية (Post-selection inference)."])
    st.markdown(cite("esl", "islp"))

template_checklist({3: "الصيغة أعلاه", 4: "الرموز في صندوق الصيغة", 5: "Hat matrix والإسقاط", 6: "‖y − Xβ‖²",
                    7: "جدول طرق الحل", 8: "coef_ وintercept_ في مختبر الكود", 9: "لا معاملات فائقة (راجع Ridge/Lasso)",
                    10: "جدول الافتراضات", 15: "قسم متقدم", 19: "تشخيص البواقي وCook's distance",
                    21: "LinearRegressionGD في مختبر الكود", 22: "LinearRegression وstatsmodels", 23: "مختبر OLS",
                    24: "المنزلقات في مختبر OLS", 25: "التمارين أسفل الصفحة", 26: "الاختبار أسفل الصفحة",
                    27: "المراجع (بحثي) وبطاقة الخوارزمية"})
mistakes(["تفسير β_j سببيًا دون تصميم.", "الاعتماد على R² وحده.", "تجاهل نقاط الرافعة المؤثرة.",
          "استخدام SE الكلاسيكية مع عدم تجانس واضح."])
page_footer("linear_regression",
            takeaways=["OLS = إسقاط y على فضاء أعمدة X.", "افتراضات التنبؤ أخف بكثير من افتراضات الاستدلال السببي.",
                       "FWL: الانحدار المتعدد = انحدار بواقي على بواقي — أساس DML."])
