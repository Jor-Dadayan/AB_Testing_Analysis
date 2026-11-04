"""Генератор синтетического A/B-датасета на уровне пользователя."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("experimentlab.generator")

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_P = [0.58, 0.35, 0.07]
COUNTRIES = ["US", "IN", "UK", "DE", "BR", "FR", "CA", "OTHER"]
COUNTRY_P = [0.32, 0.14, 0.10, 0.09, 0.08, 0.07, 0.07, 0.13]
SOURCES = ["organic", "paid_search", "direct", "email", "social"]
SOURCE_P_BY_DEVICE = {
    "mobile": [0.30, 0.20, 0.15, 0.08, 0.27],
    "desktop": [0.35, 0.28, 0.22, 0.10, 0.05],
    "tablet": [0.35, 0.20, 0.25, 0.10, 0.10],
}
AGE_MEAN = {"mobile": 32, "desktop": 41, "tablet": 45}
P_NEW_BY_SOURCE = {"organic": 0.45, "paid_search": 0.60, "direct": 0.20,
                   "email": 0.10, "social": 0.55}
COUNTRY_CART_MULT = {"US": 1.2, "UK": 1.1, "DE": 1.1, "FR": 1.05, "CA": 1.1,
                     "IN": 0.6, "BR": 0.7, "OTHER": 0.9}
DEVICE_CART_MULT = {"desktop": 1.1, "mobile": 0.95, "tablet": 1.0}

# Вклады в логит вероятности покупки
DEVICE_EFFECT = {"desktop": 0.25, "mobile": -0.15, "tablet": 0.0}
SOURCE_EFFECT = {"email": 0.5, "direct": 0.2, "organic": 0.0,
                 "paid_search": -0.1, "social": -0.35}
NEW_USER_EFFECT = -0.40


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lookup(values: np.ndarray, mapping: dict) -> np.ndarray:
    return pd.Series(values).map(mapping).to_numpy(dtype=float)


def _calibrate_shift(score: np.ndarray, target: float) -> float:
    """Подбирает сдвиг логита так, чтобы средняя вероятность равнялась target."""
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(score + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def config_fingerprint(cfg: dict) -> str:
    """Хеш параметров генерации: одинаковые параметры дают одинаковый хеш."""
    keys = ["seed", "n_users", "scenario", "start_date", "n_days",
            "treatment_share", "baseline_conversion"]
    payload = {k: cfg[k] for k in keys}
    payload["scenario_params"] = cfg["scenarios"][cfg["scenario"]]
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_users(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Возвращает (датасет, скрытая правда). Одна строка = один пользователь."""
    n = int(cfg["n_users"])
    n_days = int(cfg["n_days"])
    rng = np.random.default_rng(cfg["seed"])
    sc = cfg["scenarios"][cfg["scenario"]]
    base = float(cfg["baseline_conversion"])

    # --- Время ---
    dates = pd.date_range(cfg["start_date"], periods=n_days)
    day_w = np.where(dates.day_of_week >= 5, 1.2, 1.0)
    day_idx = rng.choice(n_days, n, p=day_w / day_w.sum())
    h = np.arange(24)
    hour_w = np.exp(-((h - 20) ** 2) / 50) + np.exp(-((h - 12) ** 2) / 40) + 0.1
    hour = rng.choice(24, n, p=hour_w / hour_w.sum())
    secs = hour * 3600 + rng.integers(0, 3600, n)
    signup_ts = (pd.DatetimeIndex(dates.to_numpy()[day_idx])
                 + pd.to_timedelta(secs, unit="s")).to_numpy()

    # --- Группа: фиксированная доля, случайный порядок ---
    n_treat = int(round(n * cfg["treatment_share"]))
    group = np.array(["control"] * (n - n_treat) + ["treatment"] * n_treat)
    rng.shuffle(group)

    # --- Профиль пользователя ---
    device = rng.choice(DEVICES, n, p=DEVICE_P)
    country = rng.choice(COUNTRIES, n, p=COUNTRY_P)
    source = np.empty(n, dtype=object)
    for dev, probs in SOURCE_P_BY_DEVICE.items():
        m = device == dev
        source[m] = rng.choice(SOURCES, int(m.sum()), p=probs)
    is_new = (rng.random(n) < _lookup(source, P_NEW_BY_SOURCE)).astype(int)
    age = np.clip(np.round(rng.normal(_lookup(device, AGE_MEAN), 11)), 18, 75).astype(int)
    sessions = np.where(is_new == 1, rng.poisson(0.3, n), rng.poisson(4.0, n))

    # --- Корзина (правосторонний разброс, зависит от страны и устройства) ---
    cart_items = 1 + rng.poisson(1.0 + 0.3 * (device == "desktop"), n)
    log_cart = (np.log(45.0) + np.log(_lookup(country, COUNTRY_CART_MULT))
                + np.log(_lookup(device, DEVICE_CART_MULT))
                + 0.15 * (cart_items - 1) + rng.normal(0, 0.45, n))
    cart_value = np.clip(np.exp(log_cart), 1.0, None).round(2)

    # --- Вероятность покупки (логит) ---
    is_weekend = (pd.DatetimeIndex(signup_ts).day_of_week >= 5).astype(int)
    score = (_lookup(device, DEVICE_EFFECT) + _lookup(source, SOURCE_EFFECT)
             + NEW_USER_EFFECT * is_new + 0.06 * np.minimum(sessions, 10)
             - 0.2 * np.log(cart_value / 45.0) + 0.05 * is_weekend
             + rng.normal(0, 0.3, n))
    shift0 = _calibrate_shift(score, base)
    shift1 = _calibrate_shift(score, base + sc["conversion_delta"])
    p0, p1 = _sigmoid(score + shift0), _sigmoid(score + shift1)
    is_treat = group == "treatment"
    p = np.where(is_treat, p1, p0)
    converted = rng.random(n) < p

    # --- Выручка: нули у не купивших, лог-нормальный хвост у купивших ---
    aov_noise = rng.lognormal(0, 0.2, n)
    mult = np.where(is_treat, sc["aov_multiplier"], 1.0)
    revenue = np.where(converted, cart_value * aov_noise * mult, 0.0).round(2)
    delay_s = np.round(rng.exponential(20, n) * 60 + 60)

    df = pd.DataFrame({
        "group": group,
        "signup_ts": signup_ts,
        "device_type": device,
        "country": country,
        "traffic_source": source,
        "is_new_user": is_new,
        "age": age,
        "sessions_prior_30d": sessions,
        "cart_items": cart_items,
        "cart_value": cart_value,
        "converted": converted.astype(int),
        "revenue": revenue,
    })
    df["purchase_ts"] = (df["signup_ts"] + pd.to_timedelta(delay_s, unit="s")).where(converted)
    df = df.sort_values("signup_ts", kind="stable").reset_index(drop=True)
    df.insert(0, "user_id", np.arange(1, n + 1))
    df["signup_date"] = df["signup_ts"].dt.date
    df["signup_hour"] = df["signup_ts"].dt.hour
    df["day_of_week"] = df["signup_ts"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    truth = {
        "scenario": cfg["scenario"],
        "seed": cfg["seed"],
        "fingerprint": config_fingerprint(cfg),
        "n_users": n,
        "true_conversion_control": float(p0.mean()),
        "true_conversion_treatment": float(p1.mean()),
        "true_conversion_delta": float(sc["conversion_delta"]),
        "true_log_odds_ratio": float(shift1 - shift0),
        "true_aov_multiplier": float(sc["aov_multiplier"]),
    }
    return df, truth


def run_generation(cfg: dict, csv_path: Path, truth_path: Path,
                   force: bool = False) -> str:
    """Генерирует и сохраняет данные. При тех же параметрах повторно не пересоздаёт."""
    csv_path, truth_path = Path(csv_path), Path(truth_path)
    fp = config_fingerprint(cfg)
    if not force and csv_path.exists() and truth_path.exists():
        old = json.loads(truth_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fp:
            log.info("Параметры не изменились, пропуск")
            return "skipped"
    df, truth = generate_users(cfg)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    log.info("Сгенерировано %s строк", len(df))
    return "generated"
