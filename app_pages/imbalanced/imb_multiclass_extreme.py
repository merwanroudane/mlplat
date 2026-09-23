import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.registry import missing_notice, optional
from core.state import at_least
from utils.plotting import confusion_heatmap, overlay_hist, plot

page_header("imb_multiclass_extreme")
imb = optional("imblearn")

st.markdown("## 1. عدم التوازن متعدد الفئات")
st.markdown("مع K فئات قد تكون عدة فئات نادرة بدرجات مختلفة. الأسئلة نفسها (المقياس، التكلفة، الاحتمالات) تصبح لكل فئة.")
formula(r"\text{Macro-}F_1 = \frac1K\sum_{k=1}^K F_{1,k},\qquad \text{Weighted-}F_1 = \sum_{k=1}^K \frac{n_k}{n}\,F_{1,k}",
        intuition="Macro يعطي كل فئة الوزن نفسه (يكشف إهمال الفئات النادرة)؛ Weighted يُغرقها في الفئة الكبيرة. Balanced "
                  "accuracy = متوسط Recall لكل فئة.")
st.code("""# per-class targets instead of a single ratio
SMOTE(sampling_strategy={1: 600, 2: 600})          # grow classes 1 and 2 to 600 rows each (training folds only)
RandomUnderSampler(sampling_strategy={0: 800})     # shrink class 0 to 800 rows
LogisticRegression(class_weight="balanced")         # w_k = n / (K n_k) for every class""", language="python")


@st.cache_data(show_spinner="5-fold CV لثلاث استراتيجيات متعددة الفئات…")
def _multiclass(has_imb: bool):
    X, y = make_classification(n_samples=3000, n_features=10, n_informative=6, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, weights=[0.85, 0.12, 0.03], class_sep=0.9, random_state=3)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = lambda **k: LogisticRegression(max_iter=3000, **k)  # noqa: E731
    models = {"plain": make_pipeline(StandardScaler(), lr()),
              "class_weight='balanced'": make_pipeline(StandardScaler(), lr(class_weight="balanced"))}
    if has_imb:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import make_pipeline as imb_pipe
        models["SMOTE {1: 1000, 2: 1000}"] = imb_pipe(StandardScaler(), SMOTE(sampling_strategy={1: 1000, 2: 1000},
                                                                             random_state=0), lr())
    rows, cms = [], {}
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, cv=cv)
        rec = recall_score(y, pred, average=None)
        rows.append({"strategy": name, "recall class 0": rec[0], "recall class 1": rec[1], "recall class 2": rec[2],
                     "macro-F1": f1_score(y, pred, average="macro"), "weighted-F1": f1_score(y, pred, average="weighted"),
                     "balanced acc": balanced_accuracy_score(y, pred), "accuracy": (pred == y).mean()})
        cms[name] = confusion_matrix(y, pred)
    return pd.DataFrame(rows), cms, np.bincount(y)


mc, cms, counts = _multiclass(imb is not None)
st.caption(f"أعداد الفئات: {counts.tolist()} (85% / 12% / 3%).")
st.dataframe(mc.round(3), hide_index=True, width="stretch")
choice = st.segmented_control("مصفوفة الالتباس", list(cms), default="plain", key="mc_cm") or "plain"
plot(confusion_heatmap(cms[choice], ["0", "1", "2"], title=f"Confusion matrix: {choice}"), height=360)
plain_row, cw_row = mc.iloc[0], mc.iloc[1]
st.caption(f"لاحظ: Recall الفئة 2 النادرة يرتفع من {plain_row['recall class 2']:.2f} إلى {cw_row['recall class 2']:.2f} مع "
           f"class_weight، بينما تنخفض Accuracy ({plain_row['accuracy']:.3f} → {cw_row['accuracy']:.3f}) وWeighted-F1 لأنهما "
           "يكافئان الفئة الكبيرة. Balanced accuracy يلتقط التحسن للفئات النادرة؛ Macro-F1 يوازن ذلك مع Precision المنخفضة.")
if imb is None:
    missing_notice("imblearn")

st.markdown("## 2. الندرة القصوى: عشرات الموجبات فقط")
st.markdown("حين تكون الموجبات 0.5% أو أقل ولدينا بضع عشرات منها، يتغير السؤال:")
comparison_table([
    {"الإطار": "تصنيف موجّه", "يستخدم": "تسميات الموجبات القليلة", "يفترض": "الموجبات القادمة تشبه الموجبات المعروفة",
     "الخطر": "تباين هائل؛ لا يعمم على أنماط جديدة"},
    {"الإطار": "كشف شذوذ (IsolationForest، LOF، One-class SVM)", "يستخدم": "الأغلبية فقط (أو دون تسميات)",
     "يفترض": "الموجبات «غريبة» عن الأغلبية", "الخطر": "الغريب ليس دائمًا المهم (ضجيج)"},
    {"الإطار": "هجين", "يستخدم": "درجة الشذوذ كخاصية داخل مصنف موجّه", "يفترض": "كلاهما مفيد", "الخطر": "تعقيد التحقق"},
])
definition("Novelty vs outlier detection",
           "Novelty: ندرّب على بيانات «نظيفة» (سالبة فقط) ونكشف الجديد المختلف. Outlier: ندرّب على بيانات فيها شواذ غير مُعلَّمة.")
page_link("anomaly_detection")

st.markdown("### مختبر الندرة القصوى · Extreme Rarity Lab")
st.caption("4000 حالة، 20 موجبة (0.5%) تقع بعيدًا عن الأغلبية. Repeated stratified 5-fold × 4. نقارن توزيع PR-AUC عبر الطيات.")


@st.cache_data(show_spinner="20 طية × ثلاثة نماذج…")
def _extreme():
    rng = np.random.default_rng(11)
    X0 = rng.normal(size=(3980, 6))
    X1 = rng.normal(loc=[2.2, -2.0, 1.5, 0, 0, 0], scale=0.9, size=(20, 6))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(3980, int), np.ones(20, int)]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    res = {"Logistic (class_weight)": [], "RandomForest (balanced_subsample)": [], "IsolationForest (no labels)": []}
    for tr, te in cv.split(X, y):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
        rf = RandomForestClassifier(200, class_weight="balanced_subsample", min_samples_leaf=2, random_state=0,
                                    n_jobs=1).fit(X[tr], y[tr])
        iso = IsolationForest(n_estimators=200, random_state=0).fit(X[tr])
        res["Logistic (class_weight)"].append(average_precision_score(y[te], lr.predict_proba(X[te])[:, 1]))
        res["RandomForest (balanced_subsample)"].append(average_precision_score(y[te], rf.predict_proba(X[te])[:, 1]))
        res["IsolationForest (no labels)"].append(average_precision_score(y[te], -iso.score_samples(X[te])))
    return res


if st.button("شغّل المختبر", key="ext_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["ext_done"] = True
if st.session_state.get("ext_done"):
    ex = _extreme()
    summary = pd.DataFrame([{"model": k, "PR-AUC mean": np.mean(v), "SD across folds": np.std(v), "min": np.min(v),
                             "max": np.max(v)} for k, v in ex.items()])
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
    plot(overlay_hist(ex, title="PR-AUC across 20 folds (each fold has only 4 positives)", nbins=20), height=360)
    sup_best = max(np.mean(ex["Logistic (class_weight)"]), np.mean(ex["RandomForest (balanced_subsample)"]))
    iso_mean = np.mean(ex["IsolationForest (no labels)"])
    st.markdown("مع 4 موجبات في كل طية، يقفز PR-AUC بين الطيات قفزات كبيرة (انظر min/max): فرق متوسطين أصغر من هذا التشتت "
                "لا معنى له. "
                + (f"هنا IsolationForest (بلا تسميات) = {iso_mean:.2f} مقابل {sup_best:.2f} لأفضل نموذج موجّه: الموجبات "
                   "تختلف عن الأغلبية في 3 أبعاد فقط من 6، بينما كثير من حالات الأغلبية في أطراف التوزيع «غريبة» بالقدر نفسه. "
                   "الشذوذ ≠ الأهمية؛ حتى 20 تسمية تحمل معلومات لا يملكها كشف الشذوذ."
                   if iso_mean < 0.7 * sup_best else
                   f"هنا IsolationForest (بلا تسميات) = {iso_mean:.2f} ينافس النموذج الموجّه ({sup_best:.2f}) لأن الموجبات "
                   "شاذة حقًا — افتراض يجب التحقق منه في بياناتك."))
warning("مع الندرة القصوى، التقسيم الطبقي إلزامي (وإلا قد تخلو طية من الموجبات)، وRepeated CV ضروري لتقدير التباين، والتقييم "
        "النهائي يحتاج بيانات أحدث أو أطول زمنيًا قدر الإمكان.")
intuition("كل موجب إضافي يساوي ذهبًا: التسمية النشطة (Active learning) وجمع بيانات أطول زمنيًا أنفع من أي خوارزمية.")

if at_least("advanced"):
    st.markdown("## متقدم: الفئات النادرة في Multi-class")
    st.markdown("- `class_weight='balanced'` يعطي كل فئة وزن n/(K·n_k).\n"
                "- في imbalanced-learn، `sampling_strategy='not majority'` يوازن كل الفئات مع الأكبر، و`'minority'` الأصغر فقط.\n"
                "- لضبط العتبات لكل فئة: حوّل إلى One-vs-Rest واضبط عتبة لكل فئة، أو اضرب الاحتمالات في أوزان تكلفة ثم argmax.")
if at_least("research"):
    researcher_note([
        "مع موجبات قليلة جدًا، أبلغ عن عدد الموجبات في كل طية وعن توزيع المقياس لا متوسطه فقط.",
        "Positive-unlabeled learning بديل حين تكون السالبة غير مؤكدة (حالات لم تُفحص).",
        "درجات الشذوذ كخصائص إضافية داخل مصنف موجّه غالبًا تجمع مزايا الإطارين.",
    ])
why("اسأل: هل الموجبات القادمة تشبه الموجبات المعروفة؟", "إن كانت أنماط الاحتيال تتغير، كشف الشذوذ أو الهجين أكثر متانة من مصنف "
                                                             "حفظ عشرين مثالًا.")
st.markdown("### المراجع")
st.markdown(cite("he2009", "liu2008", "lemaitre2017"))
mistakes(["Weighted-F1 لتقييم الفئات النادرة.", "KFold غير طبقي مع 20 موجبًا.", "مقارنة متوسطات CV دون النظر للتشتت.",
          "افتراض أن كل شاذ مهم وأن كل مهم شاذ."])
page_footer("imb_multiclass_extreme",
            takeaways=["Macro-F1 وBalanced accuracy وRecall لكل فئة لمتعدد الفئات.",
                       "sampling_strategy كقاموس لموازنة كل فئة.",
                       "مع الندرة القصوى: Repeated stratified CV، وكشف الشذوذ بديل أو مكمل."])
