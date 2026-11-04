"""Проверка входной таблицы перед анализом: понятная ошибка вместо тихого искажения."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("group_name", "converted", "revenue", "signup_date")
VALID_GROUPS = {"control", "treatment"}


def validate_analysis_frame(df: pd.DataFrame) -> None:
    """Бросает ValueError, если данные нельзя безопасно анализировать."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"во входных данных нет столбцов: {missing}")
    if df.empty:
        raise ValueError("входная таблица пуста")
    nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise ValueError(f"NULL в обязательных столбцах: {nulls.to_dict()}")
    unknown = set(df["group_name"].astype(str)) - VALID_GROUPS
    if unknown:
        raise ValueError(f"неизвестные группы: {sorted(unknown)}")
    if not set(df["converted"].unique()) <= {0, 1}:
        raise ValueError("converted должен содержать только 0 и 1")
    if (df["revenue"] < 0).any():
        raise ValueError("revenue содержит отрицательные значения")
    if ((df["revenue"] > 0) != (df["converted"] == 1)).any():
        raise ValueError("revenue > 0 допустима только у покупателей")
