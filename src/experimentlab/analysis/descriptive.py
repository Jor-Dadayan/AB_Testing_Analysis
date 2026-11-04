"""Описательная статистика по группам. Только вычисления: без графиков, SQL и интерфейса."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _clean(s) -> pd.Series:
    x = pd.Series(s).dropna().astype(float)
    if x.empty:
        raise ValueError("пустая выборка")
    return x


def describe_series(s) -> dict[str, float]:
    """Среднее, медиана, мода, дисперсия, std, min, max, IQR и перцентили."""
    x = _clean(s)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    out = {
        "count": float(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "mode": float(x.mode().iloc[0]),  # для непрерывных величин мода малоинформативна
        "variance": float(x.var(ddof=1)),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        "iqr": float(q3 - q1),
    }
    for p, v in x.quantile(list(QUANTILES)).items():
        out[f"p{int(round(p * 100)):02d}"] = float(v)
    return out


def describe_by_group(
    df: pd.DataFrame, columns: list[str], group_col: str = "group_name"
) -> pd.DataFrame:
    """Таблица статистик: строки (variable, group), столбцы - показатели."""
    rows = {}
    for col in columns:
        for group, part in df.groupby(group_col, observed=True):
            rows[(col, str(group))] = describe_series(part[col])
    out = pd.DataFrame(rows).T
    out.index.names = ["variable", "group"]
    return out


def distribution_diagnostics(s) -> dict[str, float]:
    """Форма распределения: асимметрия, эксцесс, доля нулей, скос положительной части."""
    x = _clean(s)
    median = x.median()
    out = {
        "skewness": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
        "zero_share": float((x == 0).mean()),
        "mean_to_median": float(x.mean() / median) if median != 0 else float("nan"),
    }
    positive = x[x > 0]
    if len(positive) > 2:
        out["positive_skewness"] = float(stats.skew(positive))
        out["log_positive_skewness"] = float(stats.skew(np.log(positive)))
    return out
