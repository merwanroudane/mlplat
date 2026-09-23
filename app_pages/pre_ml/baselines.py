import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from components.callouts import intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import load_dataset, xy
from utils.plotting import lines, plot

page_header("baselines")

st.markdown("## لماذا نبدأ بنموذج «غبي»؟")
intuition("Baseline يجيب عن سؤال: **كم نكسب فعلًا من النمذجة؟** 95% دقة تبدو رائعة، حتى تعرف أن التنبؤ بـ«ليس احتيالًا» "
          "دائمًا يعطي 95% أيضًا.")
comparison_table([
    {"المسألة": "Classification", "Baseline": "DummyClassifier(strategy='most_frequent' | 'prior' | 'stratified')",
     "ما يكشفه": "هل المقياس مضلل بسبب عدم التوازن؟"},
    {"المسألة": "Regression", "Baseline": "DummyRegressor(strategy='mean' | 'median')", "ما يكشفه": "R² = 0 مرجعيًا"},
    {"المسألة": "Time series", "Baseline": "Naive (y_t = y_{t-1}) أو Seasonal naive (y_t = y_{t-7})",
     "ما يكشفه": "هل النموذج يتفوق على «الأمس»؟"},
    {"المسألة": "أي مسألة", "Baseline": "نموذج خطي بسيط", "ما يكشفه": "هل التعقيد مبرر؟"},
    {"المسألة": "أي مسألة", "Baseline": "قاعدة الخبير الحالية", "ما يكشفه": "هل نتفوق على الممارسة القائمة؟"},
])

st.markdown("## مختبر: Baseline على بيانات غير متوازنة")
X, y = xy("imbalanced")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


@st.cache_data(show_spinner=False)
def _clf_table():
    rows = []
    models = {
        "Dummy most_frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy stratified": DummyClassifier(strategy="stratified", random_state=0),
        "Logistic Regression": LogisticRegression(max_iter=2000),
    }
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        rows.append({"model": name, "accuracy": (p == y_te).mean(), "balanced accuracy": balanced_accuracy_score(y_te, p),
                     "F1 (fraud)": f1_score(y_te, p, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_te, proba) if len(np.unique(proba)) > 1 else 0.5})
    return pd.DataFrame(rows)


tab = _clf_table()
st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns if c != "model"}), hide_index=True, width="stretch")
st.markdown(f"انتشار الاحتيال في الاختبار = **{y_te.mean():.1%}**. نموذج «لا احتيال دائمًا» يحقق Accuracy = "
            f"{1 - y_te.mean():.1%} وF1 = 0: الـAccuracy وحدها **لا تقول شيئًا** هنا.")
why("قارن دائمًا بمقياسين على الأقل وبـBaseline.",
    "المقياس الذي لا يميّز بين Baseline والنموذج الحقيقي مقياس غير مناسب للمسألة.")

st.markdown("## Baseline للسلاسل الزمنية: Naive forecast")
ts = load_dataset("timeseries")
s = ts["sales"].to_numpy()
test = slice(len(s) - 60, len(s))
naive = s[test.start - 1: test.stop - 1]
seasonal = s[test.start - 7: test.stop - 7]
mae_naive = np.mean(np.abs(s[test] - naive))
mae_seas = np.mean(np.abs(s[test] - seasonal))
plot(lines(np.arange(60), {"actual": s[test], "naive (yesterday)": naive, "seasonal naive (last week)": seasonal},
           title=f"Last 60 days · MAE naive = {mae_naive:.2f}, seasonal naive = {mae_seas:.2f}", xaxis="day",
           yaxis="sales", dash={"naive (yesterday)": "dot", "seasonal naive (last week)": "dash"}), height=340)
st.caption("أي نموذج سلاسل زمنية لا يتفوق على هذين الخطين لا يستحق التعقيد. (وحدة Time-Series ML تبني نموذجًا يتفوق عليهما.)")

st.markdown("## Baseline للانحدار في كود")
Xr, yr = xy("regression")


def _code(p):
    return ("from sklearn.dummy import DummyRegressor\nfrom sklearn.linear_model import LinearRegression\n\n"
            f"base = DummyRegressor(strategy='{p['strategy']}').fit(X_train, y_train)\n"
            "lin = LinearRegression().fit(X_train, y_train)\n"
            "print(mean_absolute_error(y_test, base.predict(X_test)), r2_score(y_test, base.predict(X_test)))\n"
            "print(mean_absolute_error(y_test, lin.predict(X_test)), r2_score(y_test, lin.predict(X_test)))")


def _run(strategy):
    a, b, c, d = train_test_split(Xr, yr, test_size=0.3, random_state=0)
    base = DummyRegressor(strategy=strategy).fit(a, c)
    lin = LinearRegression().fit(a, c)
    return pd.DataFrame([
        {"model": f"Dummy ({strategy})", "MAE": mean_absolute_error(d, base.predict(b)), "R²": r2_score(d, base.predict(b))},
        {"model": "LinearRegression", "MAE": mean_absolute_error(d, lin.predict(b)), "R²": r2_score(d, lin.predict(b))},
    ]).round(3)


code_lab("baseline_reg", "Dummy vs Linear on the regression dataset", _code, _run,
         lambda: {"strategy": st.segmented_control("strategy", ["mean", "median"], default="mean", key="bl_s",
                                                   required=True)},
         explanation="R² للـDummy قريب من 0 (قد يكون سالبًا قليلًا على الاختبار)؛ هذا تعريف R² نفسه: المقارنة مع التنبؤ بالمتوسط.")

if at_least("advanced"):
    st.markdown("## متقدم: Baseline كأداة تشخيص")
    st.markdown("- نموذج معقد **أسوأ** من Baseline ⇒ خطأ في الـPipeline أو تسرب معكوس أو مقياس خاطئ.\n"
                "- نموذج أفضل **بشكل مريب** من خبير المجال ⇒ ابحث عن تسرب.\n"
                "- الفجوة بين Baseline الخطي والنموذج غير الخطي تقيس قيمة اللاخطية/التفاعلات.")
if at_least("research"):
    researcher_note(["أبلغ عن Baselines في كل جدول نتائج؛ المراجعون يرفضون «التحسين» دون مرجع.",
                     "استخدم نفس الطيات لكل النماذج بما فيها Baseline لتكون المقارنات مزدوجة (Paired)."])
real_world(["ما القاعدة التي يستخدمها الناس اليوم لاتخاذ القرار؟", "ما أداء «لا تفعل شيئًا»؟"])
page_footer("baselines",
            takeaways=["كل نتيجة تحتاج Baseline.", "Dummy يكشف المقاييس المضللة.",
                       "للسلاسل الزمنية: Naive وSeasonal naive إلزاميان."],
            mistakes=["إعلان نجاح بـAccuracy مع بيانات غير متوازنة.", "تجاهل قاعدة الخبير الحالية.",
                      "مقارنة نماذج على طيات مختلفة."])
