import json

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least, log_experiment
from utils.datasets import xy
from utils.validation import package_versions

page_header("reproducibility")

comparison_table([
    {"العنصر": "Random seed", "لماذا": "العشوائية في التقسيم والنموذج والمحاكاة", "كيف": "random_state في كل مكوّن + np.random.default_rng(seed)"},
    {"العنصر": "Package versions", "لماذا": "الافتراضيات والخوارزميات تتغير بين الإصدارات", "كيف": "requirements مثبتة + تسجيل الإصدارات"},
    {"العنصر": "Dataset version", "لماذا": "البيانات تتحدث", "كيف": "Hash أو تاريخ لقطة أو DVC"},
    {"العنصر": "Split strategy", "لماذا": "جزء من تعريف النتيجة", "كيف": "نوع المقسِّم وk والبذرة والمجموعات"},
    {"العنصر": "CV", "لماذا": "عدد الطيات والتكرار يغيران الأرقام", "كيف": "سجّلها"},
    {"العنصر": "Hyperparameters", "لماذا": "بما فيها الافتراضية", "كيف": "get_params() كاملًا"},
    {"العنصر": "HPO seed & budget", "لماذا": "نتائج البحث عشوائية", "كيف": "sampler seed وعدد المحاولات"},
    {"العنصر": "Runtime notes", "لماذا": "الخيوط، GPU، نظام التشغيل", "كيف": "n_jobs، device، platform"},
])

st.markdown("## أثر البذرة: مصدر تباين يجب الإبلاغ عنه")
X, y = xy("classification")


@st.cache_data(show_spinner="10 بذور × 5 طيات…")
def _seeds():
    rows = []
    for s in range(10):
        sc = cross_val_score(RandomForestClassifier(100, random_state=s, n_jobs=1), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="roc_auc")
        rows.append({"seed": s, "ROC-AUC": sc.mean()})
    return pd.DataFrame(rows)


seeds = _seeds()
c1, c2 = st.columns([2, 1])
c1.dataframe(seeds.round(4), hide_index=True, width="stretch")
c2.metric("مدى AUC عبر البذور", f"{seeds['ROC-AUC'].min():.4f} – {seeds['ROC-AUC'].max():.4f}")
c2.metric("SD عبر البذور", f"{seeds['ROC-AUC'].std():.4f}")
why("أبلغ عن المتوسط والتشتت عبر عدة بذور، لا عن أفضل بذرة.", "اختيار البذرة الأفضل = Seed hacking؛ الفرق بين نموذجين أصغر من "
    "تشتت البذور ليس نتيجة.")

st.markdown("## البيئة الحالية")
st.dataframe(pd.DataFrame(list(package_versions().items()), columns=["package", "version"]), hide_index=True, width="stretch")

st.markdown("## سجل التجارب (Experiment Log)")
st.caption("تضيف المختبرات (Playground، Pipeline، Regularization، GD، XGBoost، Optuna، PLR، Monte Carlo، ...) سجلات هنا حين تضغط «سجّل». "
           "السجل في جلستك فقط؛ صدّره كـJSON أو CSV.")
log = st.session_state.get("experiment_log", [])
if st.button("أضف تجربة مثال (RandomForest seed 0)", key="rep_example", icon=":material/add:"):
    log_experiment("Reproducibility page", "RandomForestClassifier(100)", {"n_estimators": 100, "random_state": 0},
                   {"roc_auc": float(seeds.iloc[0]["ROC-AUC"])}, seed=0, dataset="classification", split="StratifiedKFold(5, shuffle, rs=0)",
                   notes="example entry")
    log = st.session_state["experiment_log"]
if log:
    flat = [{"time": e["time"], "lab": e["lab"], "model": e["model"], "dataset": e["dataset"], "split": e["split"], "seed": e["seed"],
             "params": json.dumps(e["params"], ensure_ascii=False), "scores": json.dumps(e["scores"]),
             "sklearn": e["versions"].get("scikit-learn")} for e in log]
    st.dataframe(pd.DataFrame(flat), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.download_button("تصدير JSON", json.dumps(log, ensure_ascii=False, indent=2), "experiment_log.json", "application/json",
                       icon=":material/download:")
    c2.download_button("تصدير CSV", pd.DataFrame(flat).to_csv(index=False).encode("utf-8-sig"), "experiment_log.csv", "text/csv",
                       icon=":material/download:")
    if c3.button("امسح السجل", key="rep_clear", icon=":material/delete:"):
        st.session_state["experiment_log"] = []
        st.rerun()
else:
    st.info("السجل فارغ. جرّب «سجّل» في أي مختبر أو أضف مثالًا.", icon=":material/history:")
intuition("السجل الجيد يسمح لشخص آخر (أو لك بعد 6 أشهر) بإعادة الرقم نفسه: البيانات + الكود + الإصدارات + المعاملات + البذور.")

if at_least("advanced"):
    st.markdown("## متقدم: قائمة إعادة الإنتاج")
    st.code("""python -m venv .venv && pip install -r requirements.txt   # pinned versions
git rev-parse HEAD > run_commit.txt                          # code version
sha256sum data/snapshot.parquet > data_hash.txt              # data version
python train.py --seed 0 --config configs/model.yaml          # everything else in a config file""", language="bash")
if at_least("research"):
    researcher_note(["الحتمية الكاملة قد تتطلب ضبط عدد الخيوط (BLAS) — بعض العمليات المتوازية غير حتمية.",
                     "في DML: n_rep وبذرة التقسيم جزء من النتيجة؛ أبلغ عنهما."])
mistakes(["بذرة واحدة «محظوظة».", "عدم تثبيت الإصدارات.", "نسيان random_state في أحد مكونات الـPipeline."])
page_footer("reproducibility",
            takeaways=["سجّل البذور والإصدارات والبيانات والتقسيم والمعاملات.", "أبلغ عن التشتت عبر البذور.",
                       "صدّر سجل التجارب مع النتائج."])
