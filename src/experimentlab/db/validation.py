"""Проверка CSV перед загрузкой: дубликаты, NULL, типы, значения, логика."""
from __future__ import annotations

import pandas as pd

DB_COLUMNS = [
    "user_id", "group", "signup_ts", "device_type", "country", "traffic_source",
    "is_new_user", "age", "sessions_prior_30d", "cart_items", "cart_value",
    "converted", "revenue", "purchase_ts",
]
NOT_NULL_COLUMNS = [c for c in DB_COLUMNS if c != "purchase_ts"]
NUMERIC_COLUMNS = [
    "user_id", "is_new_user", "age", "sessions_prior_30d", "cart_items",
    "cart_value", "converted", "revenue",
]
ALLOWED_VALUES = {
    "group": {"control", "treatment"},
    "device_type": {"mobile", "desktop", "tablet"},
    "country": {"US", "IN", "UK", "DE", "BR", "FR", "CA", "OTHER"},
    "traffic_source": {"organic", "paid_search", "direct", "email", "social"},
    "converted": {0, 1},
    "is_new_user": {0, 1},
}


def validate(df: pd.DataFrame) -> list[str]:
    """Возвращает список проблем. Пустой список означает, что данные чистые."""
    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing:
        return [f"нет столбцов: {missing}"]
    if df.empty:
        return ["датасет пуст"]

    errors: list[str] = []
    dup = int(df["user_id"].duplicated().sum())
    if dup:
        errors.append(f"дубликаты user_id: {dup}")
    nulls = df[NOT_NULL_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        errors.append(f"NULL в обязательных столбцах: {nulls.to_dict()}")
    bad_types = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_types:
        errors.append(f"неверный тип (ожидалось число): {bad_types}")
    signup = pd.to_datetime(df["signup_ts"], errors="coerce")
    purchase = pd.to_datetime(df["purchase_ts"], errors="coerce")
    if signup.isna().any():
        errors.append("signup_ts: не удалось распознать дату")
    if (df["purchase_ts"].notna() & purchase.isna()).any():
        errors.append("purchase_ts: не удалось распознать дату")
    if errors:  # дальнейшие проверки требуют корректных типов и отсутствия NULL
        return errors

    for col, allowed in ALLOWED_VALUES.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            examples = sorted(df.loc[bad, col].astype(str).unique())[:5]
            errors.append(f"{col}: недопустимые значения {examples}")
    if not df["age"].between(18, 75).all():
        errors.append("age: значения вне диапазона 18-75")
    if (df["sessions_prior_30d"] < 0).any():
        errors.append("sessions_prior_30d: отрицательные значения")
    if (df["cart_items"] < 1).any():
        errors.append("cart_items: значения меньше 1")
    if (df["cart_value"] <= 0).any():
        errors.append("cart_value: значения не больше 0")
    if (df["revenue"] < 0).any():
        errors.append("revenue: отрицательные значения")

    buyers = df["converted"] == 1
    if ((df["revenue"] > 0) != buyers).any():
        errors.append("converted и revenue не согласованы (revenue > 0 только у купивших)")
    if (df["purchase_ts"].notna() != buyers).any():
        errors.append("converted и purchase_ts не согласованы")
    if (purchase[buyers] < signup[buyers]).any():
        errors.append("purchase_ts раньше signup_ts")
    return errors
