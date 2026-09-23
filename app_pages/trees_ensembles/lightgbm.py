import numpy as np
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from config import MAX_TREES_LAB
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.plotting import lines, plot
from utils.preprocessing import ordinal_categoricals

page_header("lightgbm")
algorithm_profile("lightgbm")

st.markdown("## Leaf-wise مقابل Level-wise")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Level-wise (depth-wise)** — XGBoost الافتراضي")
    mermaid("flowchart TD\n  R((root)) --> A((L)) & B((R))\n  A --> A1((·)) & A2((·))\n  B --> B1((·)) & B2((·))")
with c2:
    st.markdown("**Leaf-wise (best-first)** — LightGBM")
    mermaid("flowchart TD\n  R((root)) --> A((L)) & B((R: leaf))\n  A --> A1((·)) & A2((·))\n  A1 --> A11((·)) & A12((·))")
intuition("Leaf-wise يوسّع دائمًا الورقة ذات أكبر انخفاض في الخسارة؛ يصل إلى خسارة أقل بعدد الأوراق نفسه، لكنه قد ينمو "
          "أشجارًا عميقة وغير متوازنة ⇒ خطر Overfitting مع بيانات قليلة. لذلك num_leaves هو المتحكم الأساسي.")
comparison_table([
    {"sklearn API": "num_leaves", "Native": "num_leaves", "الدور": "المتحكم الأساسي في التعقيد"},
    {"sklearn API": "max_depth", "Native": "max_depth", "الدور": "حماية (−1 = بلا حد)؛ اجعل num_leaves < 2^max_depth"},
    {"sklearn API": "min_child_samples", "Native": "min_data_in_leaf", "الدور": "أهم معامل ضد Overfitting مع Leaf-wise"},
    {"sklearn API": "colsample_bytree", "Native": "feature_fraction", "الدور": "خصائص عشوائية لكل شجرة"},
    {"sklearn API": "subsample + subsample_freq", "Native": "bagging_fraction + bagging_freq", "الدور": "Bagging (يحتاج freq > 0)"},
    {"sklearn API": "reg_alpha / reg_lambda", "Native": "lambda_l1 / lambda_l2", "الدور": "تنظيم الأوراق"},
    {"sklearn API": "max_bin (عبر kwargs)", "Native": "max_bin (255)", "الدور": "دقة المدرجات"},
])
hyperparameter_table("LGBMClassifier")
st.caption("GOSS (أخذ عينات حسب حجم التدرج) وEFB (دمج الخصائص الحصرية المتفرقة) من أفكار LightGBM لتسريع البيانات الكبيرة.")

st.markdown("## مختبر LightGBM")
lgb = optional("lightgbm")
if lgb is None:
    missing_notice("lightgbm")
else:
    st.caption(f"lightgbm {installed_version('lightgbm')} مثبتة.")
    X, y = xy("mixed")
    X = ordinal_categoricals(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)
    c1, c2, c3, c4 = st.columns(4)
    leaves = c1.slider("num_leaves", 2, 255, 31, key="lgb_leaves")
    mcs = c2.select_slider("min_child_samples", [1, 5, 20, 50, 100], value=20, key="lgb_mcs")
    lr = c3.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.3], value=0.1, key="lgb_lr")
    ff = c4.slider("colsample_bytree", 0.3, 1.0, 1.0, 0.1, key="lgb_ff")

    @st.cache_data(show_spinner="يدرّب LightGBM…", max_entries=32)
    def _fit(leaves, mcs, lr, ff):
        m = lgb.LGBMClassifier(n_estimators=MAX_TREES_LAB, num_leaves=leaves, min_child_samples=mcs, learning_rate=lr,
                               colsample_bytree=ff, random_state=0, n_jobs=1, verbose=-1)
        # LightGBM 4.7: eval_set is deprecated in favour of eval_X / eval_y (verified on the installed version)
        m.fit(Xa, ya, eval_X=(Xa, Xv), eval_y=(ya, yv), eval_names=["train", "valid"], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(30, verbose=False)])
        ev = m.evals_result_
        return (ev["train"]["binary_logloss"], ev["valid"]["binary_logloss"], m.best_iteration_,
                roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

    tr, va, best, auc = _fit(leaves, mcs, lr, ff)
    fig = lines(np.arange(1, len(tr) + 1), {"train": tr, "validation": va}, title="LightGBM log loss (early stopping 30)",
                xaxis="iteration", yaxis="binary log loss")
    fig.add_vline(x=best, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text=f"best {best}")
    plot(fig, height=340)
    with st.container(horizontal=True):
        st.metric("best_iteration_", best, border=True)
        st.metric("Test ROC-AUC", f"{auc:.4f}", border=True)
        st.metric("train − valid loss gap", f"{va[best - 1] - tr[best - 1]:.3f}", border=True)
    if leaves > 100 and mcs <= 5:
        warning("num_leaves كبير مع min_child_samples صغير على 1200 صف تدريب: وصفة Overfitting لـLeaf-wise.")
    st.caption("الفئات: أعمدة pandas من نوع category تُعالج أصليًا (تقسيم الفئات إلى مجموعتين).")
why("ابدأ بـnum_leaves معتدل وmin_child_samples ≥ 20 وlearning_rate صغير مع early stopping.",
    "هذه الثلاثة تحكم معظم سلوك LightGBM؛ الباقي تحسينات ثانوية.")
st.code("""model = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1)
model.fit(X_tr, y_tr, eval_X=(X_val,), eval_y=(y_val,), callbacks=[lgb.early_stopping(100)])  # LightGBM ≥ 4.7
# older versions: model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], ...)""", language="python")

if at_least("research"):
    researcher_note(["Ke et al. (2017): GOSS يحتفظ بالعينات ذات التدرج الكبير ويأخذ عينة من الصغيرة مع تصحيح وزنها.",
                     "LightGBM شائع كمتعلم إزعاج في DML لسرعته؛ اضبط min_child_samples لتجنب إزعاج مفرط الملاءمة."])
    st.markdown(cite("ke2017"))
template_checklist({5: "Leaf-wise vs Level-wise", 9: "جدول المعاملات + الأسماء البديلة", 13: "category dtype",
                    22: "LGBMClassifier", 23: "مختبر LightGBM", 24: "منزلقات المختبر"})
mistakes(["num_leaves كبير جدًا مع بيانات صغيرة.", "subsample دون subsample_freq (لا أثر).", "الخلط بين الأسماء البديلة."])
page_footer("lightgbm",
            takeaways=["Leaf-wise: خسارة أقل بسرعة وخطر Overfitting.", "num_leaves وmin_child_samples أهم المعاملات.",
                       "الفئات وNaN مدعومة أصليًا."])
