"""Подготовка данных для визуализаций. Тесты делегируются модулям analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from experimentlab.analysis import eda, power
from experimentlab.analysis.conversion import two_proportion_ztest
from experimentlab.analysis.revenue import welch_ttest

METRICS = {
    "conversion": ("Конверсия", "share"),
    "arpu": ("ARPU (выручка на пользователя)", "money"),
    "aov": ("AOV (средний чек покупателей)", "money"),
}
GROUPS = ("control", "treatment")


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    label: str
    unit: str  # share / money
    test: str
    n_control: int
    n_treatment: int
    control: float
    treatment: float
    diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    note: str


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = df["group_name"].astype(str)
    return df[g == "control"], df[g == "treatment"]


def compare_metric(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> MetricComparison:
    """Сравнение групп по выбранной метрике тестом, подходящим для её типа."""
    if metric not in METRICS:
        raise ValueError(f"неизвестная метрика: {metric}")
    label, unit = METRICS[metric]
    c, t = _split(df)
    if metric == "conversion":
        r = two_proportion_ztest(
            int(c["converted"].sum()), len(c), int(t["converted"].sum()), len(t), alpha=alpha
        )
        return MetricComparison(
            metric,
            label,
            unit,
            "Z-тест двух долей",
            len(c),
            len(t),
            r.p_control,
            r.p_treatment,
            r.diff,
            r.rel_lift,
            r.ci_low,
            r.ci_high,
            r.p_value,
            alpha,
            "Первичная метрика: по ней принимается решение.",
        )
    note = "Вторичная метрика (контроль выручки)."
    if metric == "aov":
        c, t = c[c["converted"] == 1], t[t["converted"] == 1]
        note = (
            "Защитная метрика. Покупатели отбираются после назначения группы, "
            "поэтому это описание, а не независимое сравнение."
        )
    w = welch_ttest(c["revenue"].to_numpy(float), t["revenue"].to_numpy(float), alpha)
    rel = w.diff / w.mean_control if w.mean_control else float("nan")
    return MetricComparison(
        metric,
        label,
        unit,
        "Тест Уэлча",
        len(c),
        len(t),
        w.mean_control,
        w.mean_treatment,
        w.diff,
        rel,
        w.ci_low,
        w.ci_high,
        w.p_value,
        alpha,
        note,
    )


def group_metric_ci(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """Значение метрики в каждой группе с интервалом (Уилсон для долей, t для средних)."""
    cols = ["group", "value", "ci_low", "ci_high", "n"]
    if metric == "conversion":
        ci = eda.conversion_ci(df, "group_name").reset_index()
        return pd.DataFrame(
            {
                "group": ci["group_name"].astype(str),
                "value": ci["rate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n": ci["count"],
            }
        )[cols]
    d = df.assign(group_name=df["group_name"].astype(str))
    if metric == "aov":
        d = d[d["converted"] == 1]
    rows = []
    for g in GROUPS:
        x = d.loc[d["group_name"] == g, "revenue"].to_numpy(float)
        if x.size < 2:
            continue
        m = float(x.mean())
        half = float(stats.t.ppf(1 - alpha / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
        rows.append(
            {"group": g, "value": m, "ci_low": m - half, "ci_high": m + half, "n": int(x.size)}
        )
    return pd.DataFrame(rows, columns=cols)


def daily_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Значение метрики по дням и накопленное, отдельно для каждой группы."""
    d = df.assign(
        group_name=df["group_name"].astype(str),
        signup_date=pd.to_datetime(df["signup_date"]).dt.normalize(),
    )
    agg = (
        d.groupby(["signup_date", "group_name"])
        .agg(users=("user_id", "size"), buyers=("converted", "sum"), revenue=("revenue", "sum"))
        .reset_index()
        .sort_values(["group_name", "signup_date"])
    )
    for col in ("users", "buyers", "revenue"):
        agg[f"cum_{col}"] = agg.groupby("group_name")[col].cumsum()
    num, den = {
        "conversion": ("buyers", "users"),
        "arpu": ("revenue", "users"),
        "aov": ("revenue", "buyers"),
    }[metric]
    agg["daily"] = agg[num] / agg[den].replace(0, np.nan)
    agg["cumulative"] = agg[f"cum_{num}"] / agg[f"cum_{den}"].replace(0, np.nan)
    return agg.reset_index(drop=True)


def cumulative_conversion_difference(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Накопленная разница конверсий с интервалом на каждый день (z-тест на данных до этого дня)."""
    cols = ["signup_date", "diff", "ci_low", "ci_high", "p_value", "n_total"]
    daily = daily_metrics(df, "conversion")
    wide = (
        daily.pivot(index="signup_date", columns="group_name", values=["users", "buyers"])
        .fillna(0)
        .sort_index()
        .cumsum()
    )
    rows = []
    for date, r in wide.iterrows():
        try:
            res = two_proportion_ztest(
                int(r[("buyers", "control")]),
                int(r[("users", "control")]),
                int(r[("buyers", "treatment")]),
                int(r[("users", "treatment")]),
                alpha=alpha,
            )
        except (ValueError, KeyError):
            continue
        rows.append(
            {
                "signup_date": date,
                "diff": res.diff,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
                "n_total": res.n_control + res.n_treatment,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def revenue_histogram(df: pd.DataFrame, n_bins: int = 40) -> pd.DataFrame:
    """Доля покупателей по логарифмическим интервалам чека, отдельно для групп."""
    cols = ["group", "mid", "share", "count"]
    d = df.assign(group_name=df["group_name"].astype(str))
    d = d[(d["converted"] == 1) & (d["revenue"] > 0)]
    if len(d) < 2:
        return pd.DataFrame(columns=cols)
    rev = d["revenue"].to_numpy(float)
    edges = np.geomspace(rev.min(), rev.max(), n_bins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    for g in GROUPS:
        x = d.loc[d["group_name"] == g, "revenue"].to_numpy(float)
        if not x.size:
            continue
        counts, _ = np.histogram(x, bins=edges)
        rows += [
            {"group": g, "mid": float(m), "share": float(c / x.size), "count": int(c)}
            for m, c in zip(mids, counts, strict=True)
        ]
    return pd.DataFrame(rows, columns=cols)


@dataclass(frozen=True)
class PowerScenario:
    p_control: float
    p_treatment: float
    alpha: float
    power: float
    n_control: int
    n_total: int
    days_raw: int
    days_full_weeks: int


def power_scenario(
    p_control: float,
    delta: float,
    alpha: float,
    target_power: float,
    daily_users: float,
    ratio: float = 1.0,
) -> PowerScenario:
    """Сценарий «что если»: сколько пользователей и дней нужно для заданного эффекта."""
    p_t = p_control + delta
    if not 0 < p_control < 1 or not 0 < p_t < 1:
        raise ValueError("конверсии должны быть строго между 0 и 1")
    n = power.sample_size_two_proportions(p_control, p_t, alpha, target_power, ratio)
    return PowerScenario(
        p_control,
        p_t,
        alpha,
        target_power,
        n,
        n + math.ceil(n * ratio),
        power.estimate_duration_days(n, daily_users, ratio, round_to_full_weeks=False),
        power.estimate_duration_days(n, daily_users, ratio),
    )


def power_curve(
    p_control: float,
    deltas,
    n_max: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    n_points: int = 60,
) -> pd.DataFrame:
    """Мощность z-теста в зависимости от размера контрольной группы для нескольких эффектов."""
    grid = np.unique(np.linspace(200, max(n_max, 400), n_points).round().astype(int))
    rows = []
    for d in deltas:
        p_t = p_control + d
        if not 0 < p_t < 1:
            continue
        rows += [
            {
                "n_control": int(n),
                "delta": float(d),
                "power": power.power_two_proportions(p_control, p_t, float(n), alpha, ratio),
            }
            for n in grid
        ]
    return pd.DataFrame(rows, columns=["n_control", "delta", "power"])


def mde_curve(
    p_control: float,
    n_max: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    ratio: float = 1.0,
    n_points: int = 40,
) -> pd.DataFrame:
    """Минимальный обнаружимый эффект в зависимости от размера контрольной группы."""
    grid = np.unique(np.linspace(500, max(n_max, 1000), n_points).round().astype(int))
    rows = [
        {
            "n_control": int(n),
            "mde": power.mde_two_proportions(p_control, float(n), alpha, target_power, ratio),
        }
        for n in grid
    ]
    return pd.DataFrame(rows).dropna()


def guardrail_normalized(guards) -> pd.DataFrame:
    """Эффект и граница интервала в единицах допуска. Вред сверх допуска это значения меньше -1."""
    rows = []
    for g in guards:
        if (
            g.status == "NOT_EVALUATED"
            or not np.isfinite(g.margin_abs)
            or g.margin_abs <= 0
            or not np.isfinite(g.bound)
        ):
            continue
        sign = 1.0 if g.direction == "higher_is_better" else -1.0
        rows.append(
            {
                "name": g.name,
                "label": g.label,
                "status": g.status,
                "effect": sign * g.diff / g.margin_abs,
                "bound": sign * g.bound / g.margin_abs,
            }
        )
    return pd.DataFrame(rows, columns=["name", "label", "status", "effect", "bound"])


def srm_time_series(srm_segments: pd.DataFrame) -> pd.DataFrame:
    """Доля treatment по дням (строки SRM-проверки по датам регистрации)."""
    if srm_segments.empty:
        return srm_segments
    out = srm_segments[srm_segments["dimension"] == "signup_date"].copy()
    out["date"] = pd.to_datetime(out["level"])
    return out.sort_values("date").reset_index(drop=True)
