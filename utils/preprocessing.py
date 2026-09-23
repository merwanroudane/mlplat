"""Preprocessing builders and the ML-readiness checker (sections 12 and 20)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = df.select_dtypes(include="number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return num, cat


def tabular_preprocessor(scale: bool = True, onehot: bool = True, min_frequency: float | int | None = 0.01) -> ColumnTransformer:
    """Impute → (scale) numeric; impute → one-hot categoricals. Columns chosen by dtype at fit time."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent"))]
    if onehot:
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                                  sparse_output=False)))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), make_column_selector(dtype_include="number")),
        ("cat", Pipeline(cat_steps), make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)


def ordinal_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to pandas 'category' dtype (native support in HGB / LightGBM / XGBoost)."""
    out = df.copy()
    for c in out.select_dtypes(exclude="number").columns:
        out[c] = out[c].astype("category")
    return out


# ---------------------------------------------------------------- readiness
@dataclass
class Check:
    item: str
    status: str  # "ok" | "warn" | "fail" | "manual"
    evidence: str
    action: str


def readiness_report(df: pd.DataFrame, target: str, id_col: str | None = None, time_col: str | None = None,
                     group_col: str | None = None) -> list[Check]:
    """Automated part of the 13-item ML Readiness Checklist.

    Items that need domain knowledge (availability at prediction time, unit of
    analysis) are returned as "manual" with the question to ask.
    """
    checks: list[Check] = []
    y = df[target]
    X = df.drop(columns=[target])

    # 1. target definition
    if y.isna().any():
        checks.append(Check("Target definition", "fail", f"{int(y.isna().sum())} قيم مفقودة في الهدف",
                            "عرّف الهدف لكل صف أو استبعد الصفوف غير المعرّفة (لا تعوّض الهدف)."))
    else:
        kind = "classification" if (y.dtype == object or y.nunique() <= 15) else "regression"
        checks.append(Check("Target definition", "ok", f"{y.nunique()} قيمة مميزة ⇒ غالبًا {kind}",
                            "تأكد أن الهدف يقيس ما تريد التنبؤ به فعلًا وفي الأفق الزمني الصحيح."))
    # 2. unit of analysis
    if id_col and id_col in df:
        dup_ids = int(df[id_col].duplicated().sum())
        status = "warn" if dup_ids else "ok"
        checks.append(Check("Unit of analysis", status, f"{dup_ids} صفوف تكرر المعرّف `{id_col}`",
                            "إن تكرر الكيان فاستخدم GroupKFold بالمعرّف." if dup_ids else "صف واحد لكل كيان."))
    else:
        checks.append(Check("Unit of analysis", "manual", "لا يوجد عمود معرّف محدد",
                            "ما الذي يمثله الصف: عميل؟ معاملة؟ زيارة؟ هل تتكرر الكيانات؟"))
    # 3. availability at prediction time
    checks.append(Check("Availability at prediction time", "manual", "يتطلب معرفة المجال",
                        "لكل عمود: هل قيمته معروفة لحظة التنبؤ؟ استبعد كل ما يُسجَّل بعد الحدث."))
    # 4. missingness
    miss = X.isna().mean().sort_values(ascending=False)
    worst = miss[miss > 0]
    if worst.empty:
        checks.append(Check("Missingness", "ok", "لا قيم مفقودة", "—"))
    else:
        status = "fail" if worst.iloc[0] > 0.4 else "warn"
        checks.append(Check("Missingness", status, ", ".join(f"{c}: {v:.0%}" for c, v in worst.head(4).items()),
                            "عوّض داخل الـPipeline (SimpleImputer) أو استخدم نموذجًا يدعم NaN أصليًا."))
    # 5. outliers (robust z > 5 on numeric features)
    num = X.select_dtypes(include="number")
    if not num.empty:
        med = num.median()
        mad = (num - med).abs().median().replace(0, np.nan)
        rz = ((num - med).abs() / (1.4826 * mad)).fillna(0)
        share = float((rz > 5).any(axis=1).mean())
        checks.append(Check("Outliers", "warn" if share > 0.02 else "ok", f"{share:.1%} من الصفوف فيها |robust z| > 5",
                            "افحصها: خطأ إدخال أم حالة حقيقية نادرة؟ فكّر في نموذج/Loss متين."))
    # 6. feature types
    n_num, n_cat = num.shape[1], X.shape[1] - num.shape[1]
    high_card = [c for c in X.columns if c not in num.columns and X[c].nunique() > 20]
    checks.append(Check("Feature types", "warn" if high_card else "ok",
                        f"{n_num} عددية، {n_cat} فئوية" + (f"؛ عالية الكاردينالية: {', '.join(high_card)}" if high_card else ""),
                        "استخدم OneHotEncoder مع min_frequency أو TargetEncoder (داخل CV) أو CatBoost."))
    # 7. leakage (suspiciously predictive single feature)
    suspicious = []
    if y.nunique() <= 15:
        for c in num.columns:
            s = num[c].fillna(num[c].median())
            if s.nunique() > 1:
                corr = abs(np.corrcoef(s, pd.factorize(y)[0])[0, 1])
                if corr > 0.9:
                    suspicious.append(f"{c} (|r|={corr:.2f})")
    checks.append(Check("Leakage", "fail" if suspicious else "manual",
                        "خصائص شبه مطابقة للهدف: " + ", ".join(suspicious) if suspicious else "لا مؤشر آلي واضح",
                        "خاصية تتنبأ بالهدف «أفضل من اللازم» غالبًا تسرب. راجع وحدة Leakage."))
    # 8. duplicates
    dups = int(df.duplicated().sum())
    checks.append(Check("Duplicates", "warn" if dups else "ok", f"{dups} صفوف مكررة تمامًا",
                        "أزل التكرار الخاطئ قبل التقسيم وإلا ظهرت الصفوف نفسها في التدريب والاختبار."))
    # 9. imbalance
    if y.nunique() <= 15:
        prev = y.value_counts(normalize=True)
        minority = float(prev.min())
        checks.append(Check("Imbalance", "warn" if minority < 0.1 else "ok", f"أصغر فئة: {minority:.1%}",
                            "استخدم Stratified split، ومقاييس مثل PR-AUC وBalanced accuracy، واضبط العتبة."))
    # 10. temporal order
    checks.append(Check("Temporal order", "warn" if time_col else "manual",
                        f"عمود زمني: `{time_col}`" if time_col else "لم يُحدَّد عمود زمني",
                        "إن كان الهدف مستقبليًا فاستخدم TimeSeriesSplit وتقسيمًا زمنيًا للاختبار."))
    # 11. grouped observations
    checks.append(Check("Grouped observations", "warn" if group_col else "manual",
                        f"مجموعات: `{group_col}`" if group_col else "لم تُحدَّد مجموعات",
                        "مرضى/عملاء/مدارس متكررة ⇒ GroupKFold أو StratifiedGroupKFold."))
    # 12. sample size
    n, p = X.shape
    status = "fail" if n < 10 * max(p, 1) and n < 200 else ("warn" if n < 20 * max(p, 1) else "ok")
    checks.append(Check("Sample size", status, f"n = {n}، p = {p}، n/p = {n / max(p, 1):.1f}",
                        "مع n/p صغير: نماذج بسيطة وتنظيم قوي وCV متكرر، وتحفّظ في التفسير."))
    # 13. shift risk
    checks.append(Check("Shift risk", "manual", "يتطلب معرفة سياق النشر",
                        "هل ستختلف بيانات النشر (زمن، سوق، سياسة، جهاز قياس)؟ خطط للمراقبة (وحدة Drift)."))
    return checks


def readiness_frame(checks: list[Check]) -> pd.DataFrame:
    icon = {"ok": "✅ جاهز", "warn": "⚠️ انتبه", "fail": "⛔ مشكلة", "manual": "🧭 يحتاج حكمك"}
    return pd.DataFrame([{"البند": c.item, "الحالة": icon[c.status], "الدليل": c.evidence, "الإجراء المقترح": c.action}
                         for c in checks])
