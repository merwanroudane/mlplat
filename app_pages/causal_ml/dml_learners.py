import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from config import RANDOM_SEED
from content.references import cite
from core.page import page_footer, page_header
from core.registry import available
from core.state import at_least
from utils import causal as C
from utils.datasets import make_plr
from utils.plotting import interval_plot, plot

page_header("dml_learners")

comparison_table([
    {"المتعلم": "Lasso (مع أساسات)", "متى": "إزعاج متفرق/قريب من الخطي، أبعاد عالية", "الضبط": "LassoCV داخلي", "ملاحظة": "نظرية Belloni et al."},
    {"المتعلم": "Random Forest", "متى": "لاخطية وتفاعلات؛ افتراضي قوي", "الضبط": "min_samples_leaf، max_features", "ملاحظة": "لا قياس مطلوب"},
    {"المتعلم": "Gradient boosting / HGB", "متى": "بيانات جدولية كبيرة", "الضبط": "learning_rate، العمق، التوقف", "ملاحظة": "انتبه للتوقف المبكر الداخلي"},
    {"المتعلم": "XGBoost / LightGBM", "متى": "بيانات كبيرة وسرعة", "الضبط": "كما في وحداتها", "ملاحظة": "اختيارية"},
    {"المتعلم": "Stacking / Super Learner", "متى": "لا تعرف الأفضل مسبقًا", "الضبط": "داخل كل طية", "ملاحظة": "مكلف"},
])
warning("دقة الإزعاج التنبؤية **ليست** ضمانًا لصحة θ̂: تحيز الإرباك غير المقاس لا يظهر في RMSE، ونموذج إزعاج ممتاز لا يعالج "
        "افتراضًا تعريفيًا خاطئًا. RMSE تشخيص ضروري لا كافٍ.")

st.markdown("## Learner Comparison Lab")
nonlin = st.slider("اللاخطية في g₀ وm₀", 0.0, 1.0, 1.0, 0.25, key="dl_nl")


def _learners():
    ls = {
        "OLS (linear)": LinearRegression(),
        "Lasso + poly(2)": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LassoCV(cv=3, max_iter=5000,
                                                                                                               random_state=0)),
        "Random Forest": RandomForestRegressor(150, max_features=0.5, min_samples_leaf=5, random_state=0, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0),
    }
    if available("lightgbm"):
        import lightgbm as lgb
        ls["LightGBM"] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                           random_state=0, n_jobs=1, verbose=-1)
    return ls


@st.cache_data(show_spinner="يقدّر θ بكل متعلم…", max_entries=8)
def _compare(nonlin: float):
    df = make_plr(n=1000, nonlinearity=nonlin, seed=RANDOM_SEED)
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    rows = []
    for name, lrn in _learners().items():
        t0 = time.perf_counter()
        r = C.dml_plr(X, y, d, lrn, lrn, n_folds=5)
        rows.append({"learner": name, "θ̂": r.theta, "SE": r.se, "lo": r.ci[0], "hi": r.ci[1], "RMSE ℓ̂": r.extra["rmse_l"],
                     "RMSE m̂": r.extra["rmse_m"], "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)


res = _compare(nonlin)
plot(interval_plot(res["learner"], res["θ̂"], res["lo"], res["hi"], truth=0.5, title="DML-PLR with different nuisance learners"))
st.dataframe(res.drop(columns=["lo", "hi"]).round(4), hide_index=True, width="stretch")
intuition("مع إزعاج خطي (اللاخطية = 0) يكفي OLS؛ مع لاخطية قوية يتحيز OLS بينما المتعلمون المرنون يقتربون من 0.5. المتعلم الذي "
          "يعطي أقل RMSE للإزعاج غالبًا — لا دائمًا — يعطي θ̂ أدق.")
why("اختر المتعلم بخسارة الإزعاج خارج الطية (مثلًا `nuisance_loss` في DoubleML)، لا بقيمة θ̂.",
    "اختيار المتعلم حسب θ̂ أو p-value = بحث عن النتيجة المرغوبة.")

st.markdown("## الضبط داخل كل طية (دون تسرب)")
mermaid("""
flowchart LR
  F[outer fold k held out] --> T[training part = other folds]
  T --> I[inner CV on training part only: tune hyperparameters]
  I --> R[refit tuned learner on training part]
  R --> P[predict nuisance on fold k]
""")


@st.cache_data(show_spinner="ضبط داخلي لكل طية (3×3 شبكة × 3 طيات داخلية × 5 طيات خارجية)…")
def _tuned():
    df = make_plr(n=800, seed=5)
    X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    tuned = GridSearchCV(RandomForestRegressor(100, random_state=0, n_jobs=1),
                         {"min_samples_leaf": [2, 10, 30], "max_features": [0.3, 0.6, 1.0]}, cv=3)
    t0 = time.perf_counter()
    r_t = C.dml_plr(X, y, d, tuned, tuned, n_folds=5)
    t_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_d = C.dml_plr(X, y, d, RandomForestRegressor(100, random_state=0, n_jobs=1), RandomForestRegressor(100, random_state=0, n_jobs=1),
                    n_folds=5)
    return r_t, t_t, r_d, time.perf_counter() - t0


if st.button("شغّل المقارنة: افتراضي مقابل مضبوط داخل الطيات", key="dl_tune", icon=":material/tune:"):
    st.session_state["dl_tune_on"] = True
if st.session_state.get("dl_tune_on"):
    r_t, t_t, r_d, t_d = _tuned()
    st.dataframe(pd.DataFrame([
        {"setup": "RF defaults", "θ̂": r_d.theta, "SE": r_d.se, "RMSE ℓ̂": r_d.extra["rmse_l"], "RMSE m̂": r_d.extra["rmse_m"], "seconds": t_d},
        {"setup": "GridSearchCV inside each fold", "θ̂": r_t.theta, "SE": r_t.se, "RMSE ℓ̂": r_t.extra["rmse_l"],
         "RMSE m̂": r_t.extra["rmse_m"], "seconds": t_t}]).round(4), hide_index=True, width="stretch")
    st.caption("تمرير GridSearchCV كمتعلم يجعل الضبط يحدث داخل كل طية تلقائيًا (clone + fit على جزء التدريب فقط). التكلفة: "
               "شبكة × طيات داخلية × طيات خارجية تدريبًا.")
st.code("""# DoubleML (verified in 0.11.4): tune() with grids, or tune_ml_models() with Optuna settings
plr = dml.DoubleMLPLR(data, ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor())
plr.tune({"ml_l": {"min_samples_leaf": [2, 10]}, "ml_m": {"min_samples_leaf": [2, 10]}},
         tune_on_folds=True)                  # tune separately inside each cross-fitting fold
plr.fit(); plr.evaluate_learners()            # out-of-fold nuisance losses""", language="python")

if at_least("advanced"):
    st.markdown("## متقدم: التكلفة الحسابية")
    st.markdown("DML-PLR = 2 متعلمين × K طيات × n_rep تكرارات (× شبكة × طيات داخلية إن ضُبط). IRM = 3 نماذج (g₀، g₁، m). "
                "خطط للميزانية: ابدأ بافتراضيات معقولة، واضبط فقط إن أظهرت خسائر الإزعاج مجالًا للتحسن.")
if at_least("research"):
    researcher_note(["شرط DML: ‖m̂ − m₀‖·‖ℓ̂ − ℓ₀‖ = o(n^{-1/2}) — لذلك اختر متعلمين يلائمون بنية الإزعاج المتوقعة.",
                     "أبلغ عن المتعلمين وإعداداتهم وخسائر الإزعاج خارج الطية وأي ضبط؛ جرّب عدة متعلمين كتحليل حساسية."])
    st.markdown(cite("chernozhukov2018", "belloni2014", "bach2022"))
mistakes(["اختيار المتعلم حسب θ̂.", "ضبط المتعلمين على كل البيانات ثم Cross-fitting (تسرب).", "افتراض أن RMSE منخفض = θ̂ صحيح."])
page_footer("dml_learners",
            takeaways=["المتعلمون المرنون يقللون تحيز الإزعاج غير الخطي.", "اضبط داخل كل طية لتجنب التسرب.",
                       "خسارة الإزعاج تشخيص ضروري لا كافٍ."])
