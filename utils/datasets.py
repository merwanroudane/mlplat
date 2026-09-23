"""Dataset registry (section 72).

Every dataset is synthetic, generated deterministically from a seed, so the
app ships no data files, loads instantly and keeps the *ground truth* (true
coefficients, true treatment effect, true anomalies...) whenever it is
educational. ``load_dataset(name)`` returns a DataFrame; ``DATASET_INFO``
documents purpose, design, variables and truth (used by docs/datasets.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons, make_regression

from config import RANDOM_SEED


@dataclass(frozen=True)
class DatasetInfo:
    title: str
    task: str
    target: str
    purpose: str
    design: str
    variables: dict[str, str] = field(default_factory=dict)
    truth: str = ""


DATASET_INFO: dict[str, DatasetInfo] = {
    "regression": DatasetInfo(
        "انحدار اصطناعي بمعاملات معروفة", "regression", "y",
        "تعليم الانحدار الخطي والتنظيم ومقاييس الانحدار مع معرفة المعاملات الحقيقية.",
        "Cross-sectional, n=600, 8 خصائص منها 4 مفيدة فقط، ضجيج غاوسي، وعلاقة غير خطية واحدة.",
        {"x1..x8": "خصائص عددية معيارية", "y": "الهدف"},
        "y = 3·x1 − 2·x2 + 1.5·x3 + 2·sin(2·x4) + ε, ε~N(0,1); x5..x8 بلا أثر."),
    "classification": DatasetInfo(
        "تصنيف ثنائي متوازن", "binary", "y",
        "المصنّفات، حدود القرار، المقاييس، المعايرة، وضبط المعاملات.",
        "make_classification: n=1000، 10 خصائص (5 مفيدة، 2 مكررة)، فئتان متوازنتان تقريبًا.",
        {"f0..f9": "خصائص عددية", "y": "الفئة 0/1"}),
    "imbalanced": DatasetInfo(
        "تصنيف غير متوازن (احتيال مُحاكى)", "binary", "fraud",
        "عدم التوازن، مصفوفة الالتباس، PR vs ROC، العتبة، SMOTE وأوزان الفئات.",
        "n=3000، انتشار الفئة الموجبة ≈ 5%، 8 خصائص.",
        {"f0..f7": "خصائص عددية", "fraud": "1 = احتيال"}, "الانتشار الحقيقي ≈ 5%"),
    "moons": DatasetInfo(
        "الهلالان", "binary", "y",
        "حدود القرار غير الخطية: kNN وSVM والأشجار وMLP وDBSCAN.",
        "make_moons: n=400، ضجيج 0.25، خاصيتان.", {"x1, x2": "إحداثيات", "y": "الفئة"}),
    "circles": DatasetInfo(
        "الدوائر المتحدة المركز", "binary", "y",
        "فشل الحدود الخطية وKernel trick وKernel PCA.",
        "make_circles: n=400، factor=0.45، ضجيج 0.08.", {"x1, x2": "إحداثيات", "y": "الدائرة"}),
    "multiclass": DatasetInfo(
        "تصنيف متعدد الفئات", "multiclass", "y",
        "OvR وOvO وSoftmax ومصفوفة الالتباس متعددة الفئات.",
        "make_classification: n=900، 3 فئات، خاصيتان مفيدتان للعرض.", {"x1, x2": "خصائص", "y": "0/1/2"}),
    "high_dim": DatasetInfo(
        "أبعاد عالية ومتفرقة", "regression", "y",
        "التنظيم L1/L2، اختيار الخصائص، PCA، وتسرّب اختيار الخصائص.",
        "n=200، p=100، 10 معاملات غير صفرية فقط، خصائص مترابطة جزئيًا.",
        {"x000..x099": "خصائص عددية", "y": "الهدف"}, "10 معاملات حقيقية غير صفرية (x000..x009)."),
    "mixed": DatasetInfo(
        "بيانات مختلطة: موافقة القروض", "binary", "approved",
        "Pipelines وColumnTransformer وCatBoost وHistGradientBoosting والقيم المفقودة والترميز.",
        "n=2000، أعمدة عددية وفئوية (منها عالية الكاردينالية) مع قيم مفقودة مقصودة.",
        {"age": "العمر", "income": "الدخل السنوي (مفقود أحيانًا)", "loan_amount": "مبلغ القرض",
         "employment": "نوع التوظيف", "region": "المنطقة", "city": "المدينة (عالية الكاردينالية)",
         "credit_score": "درجة الائتمان (مفقود أحيانًا)", "n_accounts": "عدد الحسابات", "approved": "الهدف"}),
    "text": DatasetInfo(
        "مراجعات منتجات قصيرة", "binary", "label",
        "Bag of Words وTF-IDF وN-grams والمصنفات الخطية.",
        "600 جملة إنجليزية قصيرة مولَّدة من قوالب لتقييم إيجابي/سلبي مع كلمات محايدة ونفي.",
        {"text": "النص", "label": "1 = إيجابي"}),
    "timeseries": DatasetInfo(
        "مبيعات يومية", "forecasting", "sales",
        "خصائص Lag وRolling، Walk-forward، والتسرب الزمني.",
        "730 يومًا: اتجاه + موسمية أسبوعية وسنوية + عطلات + ضجيج AR(1).",
        {"date": "التاريخ", "sales": "المبيعات", "promo": "عرض ترويجي 0/1"}),
    "panel": DatasetInfo(
        "Panel بتبنٍّ متدرّج للمعالجة", "panel", "y",
        "GroupKFold وبنية Panel وDiD متعدد الفترات.",
        "300 وحدة × 8 فترات؛ مجموعات تبنٍّ في 4 و6 وغير معالجة أبدًا؛ أثر ديناميكي معروف.",
        {"id": "الوحدة", "t": "الفترة", "g": "فترة أول معالجة (0 = أبدًا)", "d": "معالَج الآن", "x1, x2": "متغيرات",
         "y": "النتيجة"}, "ATT(g,t) = 1 + 0.5·(t − g) للوحدات المعالجة عند t ≥ g."),
    "causal": DatasetInfo(
        "بيانات سببية بمعالجة ثنائية", "causal", "y",
        "الإرباك، ATE/ATT/CATE، IRM، والآثار غير المتجانسة.",
        "n=2000؛ الميل للمعالجة يعتمد على X (إرباك)؛ أثر غير متجانس يعتمد على x1.",
        {"x1..x5": "متغيرات مشتركة", "d": "المعالجة 0/1", "y": "النتيجة", "tau": "الأثر الفردي الحقيقي (للتعليم فقط)"},
        "τ(x) = 1 + x1 ⇒ ATE الحقيقي ≈ 1."),
    "dml_nonlinear": DatasetInfo(
        "DGP غير خطي لـPLR", "causal", "y",
        "مقارنة الساذج والمرن وDML مع θ₀ معروف.",
        "Y = θ₀D + g₀(X) + ζ، D = m₀(X) + V، مع g₀ وm₀ غير خطيتين (على نمط Chernozhukov et al., 2018).",
        {"x1..x10": "متغيرات مشتركة", "d": "المعالجة المستمرة", "y": "النتيجة"}, "θ₀ = 0.5"),
    "blobs": DatasetInfo(
        "عناقيد غاوسية", "clustering", "cluster_true",
        "K-Means والتجميع الهرمي وGMM مع التسميات الحقيقية للمقارنة.",
        "make_blobs: 4 مراكز، تباين مختلف قليلًا.", {"x1, x2": "إحداثيات", "cluster_true": "العنقود الحقيقي"}),
    "anomalies": DatasetInfo(
        "بيانات مع شذوذ معروف", "anomaly", "is_anomaly",
        "Isolation Forest وLOF وOne-Class SVM وElliptic Envelope.",
        "500 نقطة طبيعية من عنقودين + 25 نقطة شاذة منتظمة التوزيع.",
        {"x1, x2": "إحداثيات", "is_anomaly": "الحقيقة (للتقييم فقط)"}, "25 شذوذًا من 525"),
}


# ------------------------------------------------------------------ builders
def _regression(n: int = 600, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + 2 * np.sin(2 * X[:, 3]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(8)])
    df["y"] = y
    return df


def _classification(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, flip_y=0.03,
                               class_sep=0.9, random_state=seed)
    # redundant features are exact linear combinations; measurement noise makes them strongly (not perfectly) collinear
    X = X + np.random.default_rng(seed).normal(scale=0.1, size=X.shape)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["y"] = y
    return df


def _imbalanced(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_classification(n_samples=3000, n_features=8, n_informative=5, n_redundant=1, weights=[0.95],
                               flip_y=0.01, class_sep=0.8, random_state=seed)
    X = X + np.random.default_rng(seed + 1).normal(scale=0.1, size=X.shape)  # avoid exact collinearity
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["fraud"] = y
    return df


def _moons(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_moons(n_samples=400, noise=0.25, random_state=seed)
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})


def _circles(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_circles(n_samples=400, factor=0.45, noise=0.08, random_state=seed)
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})


def _multiclass(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_classification(n_samples=900, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, class_sep=1.2, random_state=seed)
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})


def _high_dim(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, p = 200, 100
    base = rng.normal(size=(n, 20))
    X = np.hstack([base, base @ rng.normal(scale=0.3, size=(20, p - 20)) + rng.normal(size=(n, p - 20))])
    beta = np.zeros(p)
    beta[:10] = [4, -3, 3, -2, 2, 1.5, -1.5, 1, -1, 1]
    y = X @ beta + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame(X, columns=[f"x{i:03d}" for i in range(p)])
    df["y"] = y
    return df


def _mixed(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2000
    age = rng.integers(21, 70, n)
    employment = rng.choice(["salaried", "self_employed", "public", "unemployed"], n, p=[0.5, 0.2, 0.2, 0.1])
    region = rng.choice(["north", "south", "east", "west", "center"], n)
    cities = [f"city_{i:02d}" for i in range(40)]
    city = rng.choice(cities, n, p=np.r_[np.full(10, 0.06), np.full(30, 0.4 / 30)])
    income = np.exp(rng.normal(10.3, 0.5, n)) * np.where(employment == "unemployed", 0.3, 1.0)
    loan = np.exp(rng.normal(9.6, 0.6, n))
    score = np.clip(rng.normal(650, 70, n) + 0.8 * (age - 40), 300, 850)
    n_acc = rng.poisson(3, n)
    city_effect = {c: e for c, e in zip(cities, rng.normal(0, 0.6, len(cities)))}
    logit = (-1.0 + 0.012 * (score - 650) + 0.9 * np.log(income / loan) + np.vectorize(city_effect.get)(city)
             - 1.2 * (employment == "unemployed") + 0.3 * (employment == "public") + 0.02 * (age - 40))
    approved = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"age": age, "income": income.round(0), "loan_amount": loan.round(0), "employment": employment,
                       "region": region, "city": city, "credit_score": score.round(0), "n_accounts": n_acc,
                       "approved": approved})
    df.loc[rng.random(n) < 0.08, "income"] = np.nan
    df.loc[rng.random(n) < 0.05, "credit_score"] = np.nan
    return df


_POS = ["great", "excellent", "love", "amazing", "perfect", "fast delivery", "works well", "highly recommend",
        "good value", "very happy"]
_NEG = ["terrible", "broken", "waste of money", "poor quality", "stopped working", "very slow", "disappointed",
        "awful", "refund", "bad support"]
_NEU = ["the product", "this item", "the package", "my order", "the battery", "the screen", "the color", "the size"]


def _text(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(600):
        label = int(rng.random() < 0.5)
        words = list(rng.choice(_POS if label else _NEG, size=rng.integers(1, 3), replace=False))
        if rng.random() < 0.15:  # negation flips the surface sentiment of one phrase
            words.append("not " + str(rng.choice(_NEG if label else _POS)))
        if rng.random() < 0.1:  # label noise: realistic ambiguity
            words.append(str(rng.choice(_NEG if label else _POS)))
        text = f"{rng.choice(_NEU)} is {' and '.join(words)}"
        rows.append((text, label))
    return pd.DataFrame(rows, columns=["text", "label"])


def _timeseries(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 730
    t = np.arange(n)
    date = pd.date_range("2024-01-01", periods=n, freq="D")
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + rng.normal(scale=4)
    promo = (rng.random(n) < 0.1).astype(int)
    sales = (100 + 0.05 * t + 12 * np.sin(2 * np.pi * t / 7) + 20 * np.sin(2 * np.pi * t / 365.25)
             + 15 * promo + ar)
    return pd.DataFrame({"date": date, "sales": sales.round(2), "promo": promo})


def _panel(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_units, T = 300, 8
    g_unit = rng.choice([0, 4, 6], n_units, p=[0.4, 0.3, 0.3])
    alpha = rng.normal(0, 1, n_units)
    x1 = rng.normal(size=n_units)
    rows = []
    for i in range(n_units):
        for t in range(1, T + 1):
            g = g_unit[i]
            d = int(g > 0 and t >= g)
            att = (1 + 0.5 * (t - g)) if d else 0.0
            x2 = rng.normal()
            y = alpha[i] + 0.3 * t + 0.5 * x1[i] + 0.3 * x2 + att + rng.normal(scale=0.5)
            rows.append((i, t, g, d, x1[i], x2, y, att))
    return pd.DataFrame(rows, columns=["id", "t", "g", "d", "x1", "x2", "y", "true_att"])


def _causal(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2000
    X = rng.normal(size=(n, 5))
    ps = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2])))
    d = (rng.random(n) < ps).astype(int)
    tau = 1 + X[:, 0]
    y = 2 * X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 3] + tau * d + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(5)])
    df["d"], df["y"], df["tau"], df["true_ps"] = d, y, tau, ps
    return df


def make_plr(n: int = 1000, p: int = 10, theta: float = 0.5, confounding: float = 1.0, nonlinearity: float = 1.0,
             noise: float = 1.0, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Partially linear DGP with known θ₀ (inspired by the design in Chernozhukov et al., 2018).

    ``confounding`` scales how strongly X drives both D and Y; ``nonlinearity``
    interpolates between linear (0) and strongly nonlinear (1) nuisance functions.
    """
    rng = np.random.default_rng(seed)
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    lin_m = X[:, 0] + 0.25 * X[:, 2]
    lin_g = X[:, 0] + 0.25 * X[:, 2]
    nl_m = X[:, 0] + 0.25 * np.exp(X[:, 2]) / (1 + np.exp(X[:, 2]))
    nl_g = np.exp(X[:, 0]) / (1 + np.exp(X[:, 0])) + 0.25 * X[:, 2] + 0.5 * np.cos(2 * X[:, 1])
    m0 = confounding * ((1 - nonlinearity) * lin_m + nonlinearity * nl_m)
    g0 = confounding * ((1 - nonlinearity) * lin_g + nonlinearity * 2 * nl_g)
    d = m0 + rng.normal(size=n)
    y = theta * d + g0 + noise * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(p)])
    df["d"], df["y"] = d, y
    return df


def _blobs(seed: int = RANDOM_SEED) -> pd.DataFrame:
    X, y = make_blobs(n_samples=500, centers=4, cluster_std=[0.9, 1.1, 0.7, 1.3], random_state=seed)
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "cluster_true": y})


def _anomalies(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = np.vstack([rng.normal([0, 0], 0.7, size=(300, 2)), rng.normal([3, 3], 0.5, size=(200, 2))])
    outl = rng.uniform(-4, 7, size=(25, 2))
    X = np.vstack([normal, outl])
    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "is_anomaly": np.r_[np.zeros(500, int), np.ones(25, int)]})


_BUILDERS = {
    "regression": _regression, "classification": _classification, "imbalanced": _imbalanced, "moons": _moons,
    "circles": _circles, "multiclass": _multiclass, "high_dim": _high_dim, "mixed": _mixed, "text": _text,
    "timeseries": _timeseries, "panel": _panel, "causal": _causal, "dml_nonlinear": lambda: make_plr(),
    "blobs": _blobs, "anomalies": _anomalies,
}


@st.cache_data(show_spinner=False, max_entries=32)
def load_dataset(name: str) -> pd.DataFrame:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown dataset: {name!r}")
    return _BUILDERS[name]()


def xy(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target for a supervised dataset (drops teaching-only truth columns)."""
    df = load_dataset(name)
    info = DATASET_INFO[name]
    drop = [info.target] + [c for c in ("tau", "true_ps", "true_att", "is_anomaly", "cluster_true") if c in df]
    return df.drop(columns=list(dict.fromkeys(drop))), df[info.target]


def toy_2d(kind: str = "moons", n: int = 300, noise: float = 0.25, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Small 2-D sets used by boundary labs (generated fresh from widget values)."""
    if kind == "moons":
        return make_moons(n_samples=n, noise=noise, random_state=seed)
    if kind == "circles":
        return make_circles(n_samples=n, factor=0.45, noise=noise / 2, random_state=seed)
    if kind == "linear":
        X, y = make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                                   class_sep=1.4 - noise, flip_y=0.02, random_state=seed)
        return X, y
    if kind == "xor":
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X = X + rng.normal(scale=noise / 3, size=X.shape)
        return X, y
    if kind == "blobs":
        return make_blobs(n_samples=n, centers=3, cluster_std=0.6 + noise * 2, random_state=seed)
    raise KeyError(kind)


def toy_regression_1d(n: int = 60, noise: float = 0.3, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """1-D nonlinear regression problem with truth f(x) = sin(2πx) on [0, 1]."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + rng.normal(scale=noise, size=n)


def true_f_1d(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


__all__ = ["DATASET_INFO", "load_dataset", "xy", "toy_2d", "toy_regression_1d", "true_f_1d", "make_plr",
           "make_regression"]
