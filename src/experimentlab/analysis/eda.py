"""Вспомогательные функции для EDA. Здесь нет текста SQL и нет интерфейса."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.exc import SQLAlchemyError
from statsmodels.stats.proportion import proportion_confint

from experimentlab.config import PROJECT_ROOT, SQL_DIR, load_config
from experimentlab.data import generator as g
from experimentlab.db.connection import get_engine
from experimentlab.db.queries import query_file

GROUPS = ["control", "treatment"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
CART_SEGMENTS = ["low", "mid", "high"]
ORDERS = {
    "group_name": GROUPS,
    "device_type": g.DEVICES,
    "country": g.COUNTRIES,
    "traffic_source": g.SOURCES,
    "age_group": AGE_GROUPS,
    "cart_segment": CART_SEGMENTS,
}


def _derive_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Повторяет вычисляемые поля user_analytics, если БД недоступна."""
    df = df.rename(columns={"group": "group_name"})
    df["signup_ts"] = pd.to_datetime(df["signup_ts"])
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    df["age_group"] = pd.cut(
        df["age"], [0, 25, 35, 45, 55, 200], right=False, labels=AGE_GROUPS
    ).astype(str)
    df["cart_segment"] = pd.cut(
        df["cart_value"], [0, 30, 80, np.inf], right=False, labels=CART_SEGMENTS
    ).astype(str)
    hours = (df["purchase_ts"] - df["signup_ts"]).dt.total_seconds() / 3600
    df["hours_to_purchase"] = hours.round(2)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Единые типы: даты, числа (NUMERIC из БД приходит как Decimal), категории."""
    df = df.copy()
    for col in ("signup_ts", "purchase_ts", "signup_date"):
        df[col] = pd.to_datetime(df[col])
    for col in ("revenue", "cart_value", "hours_to_purchase"):
        df[col] = pd.to_numeric(df[col]).astype(float)
    for col, order in ORDERS.items():
        df[col] = pd.Categorical(df[col], categories=order)
    return df.sort_values("user_id").reset_index(drop=True)


def load_analytics_frame() -> tuple[pd.DataFrame, str]:
    """Читает user_analytics из PostgreSQL; при ошибке или пустой таблице берёт CSV."""
    reason = ""
    try:
        engine = get_engine()
        try:
            df = query_file(engine, SQL_DIR / "05_eda_extract.sql")
        finally:
            engine.dispose()
        if not df.empty:
            return _finalize(df), "PostgreSQL: таблица user_analytics"
        reason = "таблица user_analytics пуста"
    except SQLAlchemyError:
        reason = "PostgreSQL недоступен"
    cfg = load_config()
    path = PROJECT_ROOT / cfg["paths"]["raw_csv"]
    if not path.exists():
        raise FileNotFoundError(f"{reason}, и файла {path} нет: выполните scripts/generate_data.py")
    return load_csv_frame(path), f"CSV: {cfg['paths']['raw_csv']} ({reason})"


def iqr_summary(s: pd.Series, k: float = 1.5) -> dict:
    """Выбросы по правилу Тьюки: за пределами [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    n_out = int(((s < low) | (s > high)).sum())
    return {
        "n_outliers": n_out,
        "share": n_out / len(s),
        "lower_fence": low,
        "upper_fence": high,
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
    }


def balance_table(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    group_col: str = "group_name",
) -> pd.DataFrame:
    """Баланс контрольной и тестовой групп по признакам, известным ДО эксперимента.

    SMD (стандартизованная разница средних) не зависит от размера выборки.
    Ориентир: |SMD| < 0.1 означает хорошо сбалансировано.
    Для категорий берётся максимум |SMD| по уровням.
    """
    ctrl = df[df[group_col] == "control"]
    treat = df[df[group_col] == "treatment"]
    rows = []
    for col in numeric:
        a, b = ctrl[col].astype(float), treat[col].astype(float)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        smd = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        rows.append(
            {
                "variable": col,
                "kind": "numeric",
                "smd": smd,
                "p_value": stats.mannwhitneyu(a, b).pvalue,
            }
        )
    for col in categorical:
        table = pd.crosstab(df[group_col], df[col])
        shares = table.div(table.sum(axis=1), axis=0)
        smds = []
        for level in shares.columns:
            pc, pt = shares.loc["control", level], shares.loc["treatment", level]
            pooled = np.sqrt((pc * (1 - pc) + pt * (1 - pt)) / 2)
            smds.append((pt - pc) / pooled if pooled > 0 else 0.0)
        rows.append(
            {
                "variable": col,
                "kind": "categorical",
                "smd": max(smds, key=abs),
                "p_value": stats.chi2_contingency(table)[1],
            }
        )
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    out["balanced"] = out["abs_smd"] < 0.1
    return out.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def conversion_ci(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Конверсия и 95% интервал Уилсона. Это описание выборки, а не тест гипотезы."""
    agg = df.groupby(by, observed=True)["converted"].agg(["sum", "count"])
    low, high = proportion_confint(agg["sum"], agg["count"], alpha=0.05, method="wilson")
    agg["rate"] = agg["sum"] / agg["count"]
    agg["ci_low"] = low
    agg["ci_high"] = high
    return agg


def load_csv_frame(path) -> pd.DataFrame:
    """Читает CSV датасета и приводит к виду user_analytics (dashboard и демо-эксперименты)."""
    return _finalize(_derive_from_csv(pd.read_csv(path)))
