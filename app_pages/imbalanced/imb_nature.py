import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.imbalance import imbalance_ratio, make_imbalanced_2d
from utils.plotting import lines, plot

page_header("imb_nature")

st.markdown("عدم التوازن ليس مشكلة واحدة. «5% موجبة» قد تكون مسألة سهلة تمامًا أو مستحيلة تقريبًا حسب **كم** مثالًا نادرًا لدينا، "
            "و**أين** تقع الأمثلة النادرة بالنسبة للأغلبية. هذه الصفحة تفكك المشكلة قبل أي علاج (He & Garcia, 2009).")

st.markdown("## المقاييس الوصفية")
formula(r"\mathrm{IR} = \frac{n_{\text{majority}}}{n_{\text{minority}}},\qquad \pi = \frac{n_1}{n}",
        title="Imbalance ratio والانتشار",
        symbols={r"\mathrm{IR}": "نسبة عدم التوازن (1 = متوازن)", r"\pi": "الانتشار (Prevalence) = نسبة الموجبات"},
        example="3000 معاملة منها 150 احتيالًا: π = 0.05 وIR = 2850/150 = 19.")
definition("Absolute vs relative rarity",
           "**الندرة النسبية**: الأقلية صغيرة **مقارنة** بالأغلبية (π صغير). **الندرة المطلقة**: عدد أمثلة الأقلية نفسه صغير "
           "(مثلًا 20 مثالًا). 1% من 10 ملايين = 100,000 مثال — تعلّم ممكن جدًا؛ 1% من 2,000 = 20 مثالًا — المشكلة الحقيقية "
           "هنا قلة المعلومات لا النسبة.")

st.markdown("## مصادر الصعوبة")
comparison_table([
    {"المصدر": "الندرة المطلقة", "ماذا يحدث": "تقدير حدود الأقلية من أمثلة قليلة ⇒ تباين عالٍ", "المؤشر": "عدد الموجبات في التدريب"},
    {"المصدر": "تداخل الفئات (Overlap)", "ماذا يحدث": "في منطقة التداخل ترجّح الأولوية القرار للأغلبية",
     "المؤشر": "نسبة جيران الأقلية من الأغلبية"},
    {"المصدر": "الجزر الصغيرة (Small disjuncts)", "ماذا يحدث": "الأقلية موزعة على عناقيد صغيرة يتجاهلها النموذج",
     "المؤشر": "عناقيد في الأقلية؛ أداء متفاوت عبرها"},
    {"المصدر": "ضجيج التسميات", "ماذا يحدث": "بضعة أخطاء تسمية تشكّل نسبة كبيرة من الأقلية",
     "المؤشر": "أمثلة أقلية معزولة وسط الأغلبية"},
    {"المصدر": "تحوّل الانتشار (Prior shift)", "ماذا يحدث": "π في النشر ≠ π في التدريب ⇒ احتمالات وعتبات خاطئة",
     "المؤشر": "مراقبة نسبة التنبؤات الموجبة في الإنتاج"},
])

st.markdown("## لماذا تنحاز النماذج نحو الأغلبية؟ قاعدة بايز")
formula(r"P(y=1\mid x) = \frac{\pi\, f_1(x)}{\pi\, f_1(x) + (1-\pi)\, f_0(x)}"
        r"\ \ \Longrightarrow\ \ P(y=1\mid x) \ge 0.5 \iff \frac{f_1(x)}{f_0(x)} \ge \frac{1-\pi}{\pi}",
        symbols={"f_1, f_0": "كثافة x داخل كل فئة", r"\pi": "الانتشار"},
        intuition="مع π = 0.01 لا يتنبأ المصنف الأمثل (لخسارة 0-1) بـ«موجب» إلا حيث تكون كثافة الأقلية أكبر بـ99 مرة من كثافة "
                  "الأغلبية. هذا ليس خطأ في النموذج: إنه **القرار الصحيح لدقة Accuracy**. المشكلة أن Accuracy ليس هدفنا.",
        example="π = 0.05 ⇒ نسبة الكثافة المطلوبة = 0.95/0.05 = 19.")
warning("انهيار Recall عند العتبة 0.5 مع بيانات غير متوازنة **ليس عيبًا في الاحتمالات** بل في اختيار العتبة لهدف غير الهدف "
        "الحقيقي. العلاج الأول: عتبة وتكلفة مناسبتان (صفحة التعلم الحساس للتكلفة)، لا إعادة العيّنة تلقائيًا.")

st.markdown("## مختبر هندسة عدم التوازن · Imbalance Geometry Lab")
c1, c2, c3, c4 = st.columns(4)
frac = c1.select_slider("نسبة الأقلية", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], value=0.05, key="nat_frac")
sep = c2.slider("المسافة (تداخل أقل ←)", 0.5, 4.0, 2.0, 0.5, key="nat_sep")
clusters = c3.segmented_control("عناقيد الأقلية", [1, 2, 3, 4], default=1, key="nat_cl") or 1
model_name = c4.selectbox("النموذج", ["Logistic regression", "kNN (k=15)", "Decision tree (depth 4)"], key="nat_model")
X, y = make_imbalanced_2d(n=1000, minority_frac=frac, separation=sep, n_clusters=clusters, seed=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
model = {"Logistic regression": LogisticRegression(max_iter=2000), "kNN (k=15)": KNeighborsClassifier(15),
         "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0)}[model_name].fit(X_tr, y_tr)
pred = model.predict(X_te)
p = model.predict_proba(X_te)[:, 1]
left, right = st.columns([1.5, 1])
with left:
    boundary_chart(model, X, y, title=f"{model_name}: decision regions at threshold 0.5 (IR = {imbalance_ratio(y):.0f})",
                   height=440)
with right:
    st.metric("Accuracy", f"{accuracy_score(y_te, pred):.3f}", border=True)
    st.metric("Accuracy لمصنف «الأغلبية دائمًا»", f"{1 - y_te.mean():.3f}", border=True)
    st.metric("Recall الأقلية @0.5", f"{recall_score(y_te, pred, zero_division=0):.3f}", border=True)
    st.metric("Precision @0.5", f"{precision_score(y_te, pred, zero_division=0):.3f}", border=True)
    st.metric("Balanced accuracy", f"{balanced_accuracy_score(y_te, pred):.3f}", border=True)
    st.metric("PR-AUC (مستقل عن العتبة)", f"{average_precision_score(y_te, p):.3f}", border=True)
st.caption("جرّب: نسبة 1% مع مسافة 1.0 ⇒ Accuracy ممتاز وRecall قريب من الصفر. ثم زد المسافة: المشكلة تختفي رغم أن النسبة لم "
           "تتغير — الصعوبة من التداخل لا من النسبة وحدها. ثم جرّب 3 عناقيد: جزر صغيرة يصعب على النموذج الخطي التقاطها.")

st.markdown("## الندرة المطلقة: نفس النسبة، أعداد مختلفة")
st.caption("π ثابتة = 5% والتداخل ثابت؛ نغيّر حجم البيانات فيتغير عدد الموجبات المطلق. Logistic regression، اختبار ثابت كبير.")


@st.cache_data(show_spinner="منحنى التعلم حسب عدد الموجبات…")
def _absolute_rarity() -> pd.DataFrame:
    X_big, y_big = make_imbalanced_2d(n=20000, minority_frac=0.05, separation=1.5, n_clusters=3, seed=7)
    X_tr, X_te, y_tr, y_te = train_test_split(X_big, y_big, test_size=0.5, random_state=0, stratify=y_big)
    rng = np.random.default_rng(0)
    rows = []
    for n in (100, 200, 400, 1000, 2000, 5000, 10000):
        aps = []
        for r in range(10):
            idx = rng.choice(len(y_tr), n, replace=False)
            if y_tr[idx].sum() < 2:
                continue
            m = KNeighborsClassifier(min(15, int(y_tr[idx].sum()) * 2 + 1)).fit(X_tr[idx], y_tr[idx])
            aps.append(average_precision_score(y_te, m.predict_proba(X_te)[:, 1]))
        rows.append({"n": n, "positives": int(round(0.05 * n)), "PR-AUC mean": np.mean(aps), "PR-AUC SD": np.std(aps)})
    return pd.DataFrame(rows)


ar = _absolute_rarity()
plot(lines(ar["positives"], {"PR-AUC (mean of 10 draws)": ar["PR-AUC mean"], "SD across draws": ar["PR-AUC SD"]},
           title="Same prevalence (5%), more positive examples", xaxis="number of positive training examples",
           yaxis="PR-AUC on a large test set", log_x=True, markers=True), height=340)
intuition("بعد عدد كافٍ من الموجبات يستقر الأداء — ولا يتغير الانتشار أبدًا في هذا المختبر. جمع أمثلة نادرة إضافية (أو تسميات "
          "أفضل) غالبًا أنفع من أي خدعة إعادة عيّنة.")

if at_least("advanced"):
    st.markdown("## متقدم: المصنف الأمثل تحت تكلفة")
    formula(r"\hat y(x) = 1 \iff P(y=1\mid x) \ge \frac{C_{FP}}{C_{FP}+C_{FN}}",
            intuition="الانتشار يدخل في P(y=1|x) والتكلفة تدخل في العتبة. الفصل بينهما هو أساس كل ما يأتي في هذا المسار.")
if at_least("research"):
    researcher_note([
        "التداخل والجزر الصغيرة كثيرًا ما تفسر صعوبة المسألة أكثر من IR نفسه؛ أبلغ عن مؤشرات تعقيد البيانات لا عن IR وحده.",
        "في البيانات الطبية، تصحيح عدم التوازن قد يضر المعايرة دون مكسب في التمييز (van den Goorbergh et al., 2022).",
    ])
why("شخّص قبل أن تعالج: كم موجبًا لديك؟ هل تتداخل الفئات؟ هل الأقلية عناقيد؟",
    "كل تشخيص يقود إلى علاج مختلف: بيانات أكثر، نموذج أكثر مرونة، عتبة مختلفة، أو تنظيف الضجيج.")
st.markdown("### المراجع")
st.markdown(cite("he2009", "elkan2001", "vandengoorbergh2022"))
mistakes(["الحكم على الصعوبة من IR وحده.", "الاحتفال بـAccuracy 95% مع 5% موجبات.",
          "لوم الاحتمالات بدل العتبة.", "تجاهل أن عدد الموجبات المطلق هو ما يحدد ما يمكن تعلّمه."])
page_footer("imb_nature",
            takeaways=["IR والانتشار وصف، لا تشخيص.", "الصعوبة من الندرة المطلقة والتداخل والجزر الصغيرة والضجيج.",
                       "قاعدة بايز تفسر انحياز العتبة 0.5 نحو الأغلبية."])
