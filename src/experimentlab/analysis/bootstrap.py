"""Bootstrap: разница конверсий, relative lift, ARPU и произвольная статистика."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from experimentlab.analysis.revenue import BootstrapResult, bootstrap_mean_diff_ci


def _check(x: int, n: int, name: str) -> None:
    if n <= 0 or not 0 <= x <= n:
        raise ValueError(f"{name}: нужно 0 <= x <= n и n > 0")


def _binomial_draws(x_c, n_c, x_t, n_t, n_boot, seed):
    _check(x_c, n_c, "control")
    _check(x_t, n_t, "treatment")
    rng = np.random.default_rng(seed)
    p_c = rng.binomial(n_c, x_c / n_c, n_boot) / n_c
    p_t = rng.binomial(n_t, x_t / n_t, n_boot) / n_t
    return p_c, p_t


def bootstrap_conversion_diff(
    x_control: int, n_control: int, x_treatment: int, n_treatment: int,
    n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> BootstrapResult:
    """Bootstrap-интервал для разницы конверсий (treatment - control)."""
    p_c, p_t = _binomial_draws(x_control, n_control, x_treatment, n_treatment, n_boot, seed)
    lo, hi = np.quantile(p_t - p_c, [alpha / 2, 1 - alpha / 2])
    point = x_treatment / n_treatment - x_control / n_control
    return BootstrapResult(float(point), float(lo), float(hi), n_boot, seed, alpha)


def bootstrap_conversion_relative_lift(
    x_control: int, n_control: int, x_treatment: int, n_treatment: int,
    n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> BootstrapResult:
    """Bootstrap-интервал для relative lift = p_T / p_C - 1 (поле diff содержит lift)."""
    if x_control == 0:
        raise ValueError("relative lift не определён при нулевой конверсии control")
    p_c, p_t = _binomial_draws(x_control, n_control, x_treatment, n_treatment, n_boot, seed)
    ok = p_c > 0
    lo, hi = np.quantile(p_t[ok] / p_c[ok] - 1, [alpha / 2, 1 - alpha / 2])
    point = (x_treatment / n_treatment) / (x_control / n_control) - 1
    return BootstrapResult(float(point), float(lo), float(hi), n_boot, seed, alpha)


def bootstrap_arpu_diff(
    control, treatment, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> BootstrapResult:
    """Bootstrap-интервал для разницы ARPU (среднее по ВСЕМ пользователям, включая нули)."""
    return bootstrap_mean_diff_ci(control, treatment, n_boot, alpha, seed)


def bootstrap_statistic_diff(
    control,
    treatment,
    statistic: Callable[[np.ndarray], np.ndarray] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    max_cells: int = 5_000_000,
) -> BootstrapResult:
    """Bootstrap для разницы любой статистики. statistic получает матрицу (k, n), возвращает (k,)."""
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    if c.size < 2 or t.size < 2:
        raise ValueError("нужно минимум 2 наблюдения в каждой группе")
    stat = statistic or (lambda m: m.mean(axis=1))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    chunk = max(1, max_cells // max(c.size, t.size))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        s_c = stat(c[rng.integers(0, c.size, size=(k, c.size))])
        s_t = stat(t[rng.integers(0, t.size, size=(k, t.size))])
        diffs[done:done + k] = s_t - s_c
        done += k
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point = float(stat(t[None, :])[0] - stat(c[None, :])[0])
    return BootstrapResult(point, float(lo), float(hi), n_boot, seed, alpha)


def excludes_zero(res: BootstrapResult) -> bool:
    """Интервал не содержит 0: bootstrap-аналог «значимо»."""
    return not (res.ci_low <= 0.0 <= res.ci_high)
