import streamlit as st

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import flow
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.registry import optional
from core.state import at_least

page_header("mlops")

st.markdown("## المكونات")
comparison_table([
    {"المكوّن": "Experiment tracking", "السؤال": "ما الذي جرّبناه وبأي نتيجة؟", "أمثلة أدوات": "MLflow Tracking، سجل تجارب المنصة"},
    {"المكوّن": "Model registry", "السؤال": "أي نموذج في الإنتاج؟ ما إصداره وحالته؟", "أمثلة أدوات": "MLflow Model Registry"},
    {"المكوّن": "Data validation", "السؤال": "هل البيانات الجديدة تطابق العقد؟", "أمثلة أدوات": "Pandera، Great Expectations (منصة DSplat)"},
    {"المكوّن": "Model validation", "السؤال": "هل النموذج الجديد أفضل وآمن للنشر؟", "أمثلة أدوات": "اختبارات أداء + عدالة + معايرة"},
    {"المكوّن": "CI/CD", "السؤال": "هل التغيير يمر بالاختبارات آليًا؟", "أمثلة أدوات": "GitHub Actions + pytest"},
    {"المكوّن": "Deployment", "السؤال": "كيف يصل النموذج للمستخدمين؟", "أمثلة أدوات": "Batch job، REST API، Shadow/Canary"},
    {"المكوّن": "Monitoring", "السؤال": "هل الأداء والبيانات مستقرة؟", "أمثلة أدوات": "PSI/KS، أداء متأخر، تنبيهات"},
    {"المكوّن": "Lineage & reproducibility", "السؤال": "من أي بيانات وكود نتج هذا النموذج؟", "أمثلة أدوات": "Git + نسخ البيانات + الإصدارات"},
])
flow(["Code + data versioned", "CI: tests + validation", "Train & track", "Register candidate", "Validate vs champion",
      "Deploy (shadow → canary → full)", "Monitor", "Retrain / rollback"], direction="LR")

st.markdown("## بوابة التحقق قبل النشر (Champion vs Challenger)")
st.code("""def promote(challenger, champion, X_val, y_val):
    checks = {
        "auc_not_worse": auc(challenger) >= auc(champion) - 0.005,
        "calibration_ok": brier(challenger) <= brier(champion) * 1.05,
        "subgroup_recall_gap": max_gap(recall_by_group(challenger)) <= 0.05,   # fairness guardrail
        "latency_ms_p95": p95_latency(challenger) <= 50,
        "schema_matches_contract": validate_schema(challenger.feature_names_in_),
    }
    return all(checks.values()), checks""", language="python")
why("اجعل معايير الترقية مكتوبة وآلية.", "قرار «النموذج الجديد أفضل» بالعين يتأثر بالرغبة؛ البوابة المكتوبة قابلة للتدقيق والتكرار.")

st.markdown("## MLflow (اختياري)")
st.code("""import mlflow
mlflow.set_experiment("loan-approval")
with mlflow.start_run():
    mlflow.log_params({"learning_rate": 0.1, "max_leaf_nodes": 31})
    mlflow.log_metric("val_auc", 0.84)
    mlflow.sklearn.log_model(pipeline, name="model")   # artifact + environment""", language="python")
mlflow = optional("mlflow")
st.caption(("mlflow مثبتة هنا." if mlflow is not None else "mlflow غير مثبتة في هذه البيئة (اختيارية؛ الإصدار المتحقق من PyPI: 3.16.1). ") +
           " تحقق من توقيعات الدوال في توثيق الإصدار المثبت؛ هذا المقتطف توضيحي.")
page_link("reproducibility", "جرّب سجل التجارب المدمج في المنصة", ":material/history:")
intuition("مستويات النضج: (0) كل شيء يدوي في Notebook ← (1) Pipeline تدريب مؤتمت ← (2) CI/CD للـPipeline نفسه مع مراقبة وإعادة "
          "تدريب آلية.")

if at_least("advanced"):
    st.markdown("## أنماط النشر الآمن")
    comparison_table([
        {"النمط": "Shadow", "الوصف": "النموذج الجديد يتنبأ بالتوازي دون أن يؤثر في القرار"},
        {"النمط": "Canary", "الوصف": "نسبة صغيرة من الحركة للنموذج الجديد ثم التوسيع"},
        {"النمط": "A/B test", "الوصف": "تجربة عشوائية لقياس الأثر على مقياس الأعمال (سببي!)"},
        {"النمط": "Blue/green", "الوصف": "بيئتان كاملتان والتبديل الفوري للتراجع"},
    ])
if at_least("research"):
    researcher_note(["اختبار A/B للنموذج الجديد هو تجربة عشوائية: التقدير الصحيح لأثره على الأعمال سببي بطبيعته.",
                     "Feedback loops: النموذج يؤثر في البيانات التي سيتعلم منها لاحقًا (Performative prediction)."])
    st.markdown(cite("sculley2015", "mlcc"))
mistakes(["نشر دون بوابة تحقق.", "لا سجل لما نُشر ومتى.", "إعادة تدريب آلية دون تحقق من البيانات الجديدة."])
page_footer("mlops",
            takeaways=["MLOps = نسخ، تتبع، تحقق، نشر آمن، مراقبة، إعادة تدريب.", "بوابات ترقية مكتوبة وآلية.",
                       "Shadow/Canary قبل الاستبدال الكامل."])
