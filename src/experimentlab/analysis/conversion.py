"""Анализ конверсии: z-тест двух долей, доверительные интервалы, проверка SRM."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

ALTERNATIVES = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class ProportionTestResult:
    x_control: int
    n_control: int
    x_treatment: int
    n_treatment: int
    p_control: float
    p_treatment: float
    diff: float  # p_treatment - p_control
    rel_lift: float  # p_treatment / p_control - 1
    se_pooled: float  # SE при H0 (для z-статистики)
    se_unpooled: float  # SE без предположения H0 (для доверительного интервала)
    z: float
    p_value: float
    alternative: str
    alpha: float
    ci_low: float
    ci_high: float
    rel_lift_ci_low: float
    rel_lift_ci_high: float
    assumption_issues: tuple[str, ...]


@dataclass(frozen=True)
class SRMResult:
    n_control: int
    n_treatment: int
    expected_treatment_share: float
    chi2: float
    p_value: float
    alpha: float
    passed: bool


def _check_counts(name: str, x: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"{name}: размер группы должен быть больше 0")
    if not 0 <= x <= n:
        raise ValueError(f"{name}: число успехов должно быть от 0 до n")


def p_value_from_z(z: float, alternative: str = "two-sided") -> float:
    """p-value для стандартной нормальной статистики."""
    if alternative == "two-sided":
        return float(2 * stats.norm.sf(abs(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    if alternative == "less":
        return float(stats.norm.cdf(z))
    raise ValueError(f"alternative должен быть одним из {ALTERNATIVES}")


def proportion_assumption_issues(
    x_c: int, n_c: int, x_t: int, n_t: int, min_count: int = 10
) -> tuple[str, ...]:
    """Нормальное приближение надёжно, если в каждой группе >= 10 успехов и >= 10 неудач."""
    issues = []
    for name, x, n in (("control", x_c, n_c), ("treatment", x_t, n_t)):
        if x < min_count or n - x < min_count:
            issues.append(
                f"{name}: успехов или неудач меньше {min_count}, "
                "нормальное приближение ненадёжно (нужен точный тест)"
            )
    return tuple(issues)


def two_proportion_ztest(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> ProportionTestResult:
    """Z-тест двух долей. Эффект = treatment - control.

    z = (p_T - p_C) / SE0, где SE0 использует ОБЪЕДИНЁННУЮ долю (под H0 доли равны).
    ДИ строится с раздельной SE1, потому что под альтернативой доли различаются.
    """
    _check_counts("control", x_control, n_control)
    _check_counts("treatment", x_treatment, n_treatment)
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative должен быть одним из {ALTERNATIVES}")
    if not 0 < alpha < 1:
        raise ValueError("alpha должен быть в интервале (0, 1)")

    p_c = x_control / n_control
    p_t = x_treatment / n_treatment
    diff = p_t - p_c
    p_pool = (x_control + x_treatment) / (n_control + n_treatment)
    if p_pool in (0.0, 1.0):
        raise ValueError("все исходы одинаковы: тест не определён")

    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    z = diff / se_pooled
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    if x_control > 0 and x_treatment > 0:  # метод Катца: ДИ для отношения долей через логарифм
        log_rr = math.log(p_t / p_c)
        se_log = math.sqrt((1 - p_t) / x_treatment + (1 - p_c) / x_control)
        rel = p_t / p_c - 1
        rel_lo = math.exp(log_rr - z_crit * se_log) - 1
        rel_hi = math.exp(log_rr + z_crit * se_log) - 1
    else:
        rel = rel_lo = rel_hi = float("nan")

    return ProportionTestResult(
        x_control=int(x_control),
        n_control=int(n_control),
        x_treatment=int(x_treatment),
        n_treatment=int(n_treatment),
        p_control=p_c,
        p_treatment=p_t,
        diff=diff,
        rel_lift=rel,
        se_pooled=se_pooled,
        se_unpooled=se_unpooled,
        z=z,
        p_value=p_value_from_z(z, alternative),
        alternative=alternative,
        alpha=alpha,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
        rel_lift_ci_low=rel_lo,
        rel_lift_ci_high=rel_hi,
        assumption_issues=proportion_assumption_issues(
            x_control, n_control, x_treatment, n_treatment
        ),
    )


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_treatment_share: float = 0.5,
    alpha: float = 0.001,
) -> SRMResult:
    """Sample Ratio Mismatch: хи-квадрат сравнивает фактические размеры групп с планом."""
    if not 0 < expected_treatment_share < 1:
        raise ValueError("expected_treatment_share должна быть строго между 0 и 1")
    if n_control < 0 or n_treatment < 0 or n_control + n_treatment == 0:
        raise ValueError("размеры групп неотрицательны, их сумма должна быть больше 0")
    n = n_control + n_treatment
    exp_t = n * expected_treatment_share
    exp_c = n - exp_t
    chi2 = (n_control - exp_c) ** 2 / exp_c + (n_treatment - exp_t) ** 2 / exp_t
    p = float(stats.chi2.sf(chi2, df=1))
    return SRMResult(
        n_control, n_treatment, expected_treatment_share, float(chi2), p, alpha, bool(p >= alpha)
    )
