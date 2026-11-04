"""Мощность, MDE, размер выборки и длительность теста."""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

ALTERNATIVES = ("two-sided", "greater", "less")


def _z_alpha(alpha: float, alternative: str) -> float:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative должен быть одним из {ALTERNATIVES}")
    return float(norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha))


def power_two_proportions(
    p_control: float,
    p_treatment: float,
    n_control: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> float:
    """Мощность z-теста долей. ratio = n_treatment / n_control.

    Под H0 дисперсия объединённая (SE0), под альтернативой раздельная (SE1):
    power = Phi((|d| - z_a * SE0) / SE1) (+ вторая кость для двустороннего теста).
    """
    n_t = ratio * n_control
    delta = p_treatment - p_control
    p_bar = (n_control * p_control + n_t * p_treatment) / (n_control + n_t)
    se0 = math.sqrt(p_bar * (1 - p_bar) * (1 / n_control + 1 / n_t))
    se1 = math.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_t)
    z = _z_alpha(alpha, alternative)
    if alternative == "two-sided":
        return float(norm.cdf((delta - z * se0) / se1) + norm.cdf((-delta - z * se0) / se1))
    if alternative == "greater":
        return float(norm.cdf((delta - z * se0) / se1))
    return float(norm.cdf((-delta - z * se0) / se1))


def sample_size_two_proportions(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """Минимальный размер КОНТРОЛЬНОЙ группы (treatment = ratio * это число)."""
    if p_control == p_treatment:
        raise ValueError("разница долей равна нулю: размер выборки не определён")
    for p in (p_control, p_treatment):
        if not 0 < p < 1:
            raise ValueError("доли должны быть строго между 0 и 1")

    def gap(n: float) -> float:
        return power_two_proportions(p_control, p_treatment, n, alpha, ratio, alternative) - power

    lo, hi = 2.0, 1e10
    if gap(lo) >= 0:
        return int(lo)
    if gap(hi) < 0:
        raise ValueError("целевая мощность недостижима при такой альтернативе")
    return math.ceil(brentq(gap, lo, hi))


def mde_two_proportions(
    p_control: float,
    n_control: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> float:
    """Минимальный обнаружимый РОСТ конверсии (абсолютный) при заданном размере выборки."""
    lo, hi = p_control + 1e-9, 1 - 1e-9

    def gap(p: float) -> float:
        return power_two_proportions(p_control, p, n_control, alpha, ratio, alternative) - power

    if gap(hi) < 0:
        return float("nan")
    if gap(lo) >= 0:
        return lo - p_control
    return float(brentq(gap, lo, hi) - p_control)


def sample_size_two_means(
    sd: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """Размер контрольной группы для разницы средних: n = sd^2 (1 + 1/r) (z_a + z_b)^2 / d^2."""
    if sd <= 0 or delta == 0:
        raise ValueError("sd должно быть > 0, delta не равна 0")
    z_sum = _z_alpha(alpha, alternative) + float(norm.ppf(power))
    return math.ceil(sd**2 * (1 + 1 / ratio) * z_sum**2 / delta**2)


def mde_two_means(
    sd: float,
    n_control: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> float:
    """Минимальная обнаружимая разница средних: d = (z_a + z_b) * sd * sqrt((1 + 1/r) / n)."""
    z_sum = _z_alpha(alpha, alternative) + float(norm.ppf(power))
    return float(z_sum * sd * math.sqrt((1 + 1 / ratio) / n_control))


def estimate_duration_days(
    n_control: float,
    daily_users: float,
    ratio: float = 1.0,
    round_to_full_weeks: bool = True,
) -> int:
    """Дни до набора выборки. Округление до недель выравнивает дни недели в обеих группах."""
    if daily_users <= 0:
        raise ValueError("daily_users должен быть больше 0")
    days = math.ceil(n_control * (1 + ratio) / daily_users)
    return math.ceil(days / 7) * 7 if round_to_full_weeks else days
