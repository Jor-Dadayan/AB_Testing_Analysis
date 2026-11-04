"""Анализ выручки: ARPU/ARPPU, тест Уэлча, Манна-Уитни, bootstrap."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class WelchResult:
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    diff: float  # mean_treatment - mean_control
    se: float
    t: float
    df: float  # степени свободы Уэлча-Саттертуэйта
    p_value: float
    alternative: str
    alpha: float
    ci_low: float
    ci_high: float
    assumption_issues: tuple[str, ...]


@dataclass(frozen=True)
class MannWhitneyResult:
    n_control: int
    n_treatment: int
    u_treatment: float
    p_value: float
    prob_superiority: float  # P(Y_T > Y_C) + 0.5 * P(Y_T = Y_C)
    alternative: str


@dataclass(frozen=True)
class BootstrapResult:
    diff: float
    ci_low: float
    ci_high: float
    n_boot: int
    seed: int
    alpha: float


def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 2:
        raise ValueError("нужна одномерная выборка минимум из 2 значений")
    if not np.isfinite(a).all():
        raise ValueError("в выборке есть NaN или бесконечности")
    return a


def arpu(x) -> float:
    """Average Revenue Per User: выручка / ВСЕ пользователи (включая нули)."""
    return float(_as_array(x).mean())


def arppu(x) -> float:
    """Average Revenue Per Paying User: выручка / только покупатели."""
    a = _as_array(x)
    paying = a[a > 0]
    return float(paying.mean()) if paying.size else float("nan")


def total_revenue(x) -> float:
    return float(_as_array(x).sum())


def revenue_summary(control, treatment) -> dict[str, dict[str, float]]:
    """ARPU, ARPPU, общая выручка, число покупателей по группам."""
    out = {}
    for name, x in (("control", control), ("treatment", treatment)):
        a = _as_array(x)
        payers = int((a > 0).sum())
        out[name] = {
            "users": int(a.size),
            "payers": payers,
            "payer_share": payers / a.size,
            "total": total_revenue(a),
            "arpu": arpu(a),
            "arppu": arppu(a),
        }
    return out


def clt_issues(a: np.ndarray, name: str) -> list[str]:
    """Проверка предпосылки теста Уэлча: среднее выборки должно быть почти нормальным.

    Ориентир (Cochran / Sugden и др.): n >= 28 + 25 * skew^2. Это эвристика, не закон.
    """
    issues = []
    if a.size < 30:
        issues.append(f"{name}: меньше 30 наблюдений")
    needed = 28 + 25 * float(stats.skew(a)) ** 2
    if a.size < needed:
        issues.append(
            f"{name}: n={a.size} меньше ориентира {needed:.0f} для данной асимметрии; "
            "нормальность среднего сомнительна, опирайтесь на bootstrap"
        )
    return issues


def _t_p_value(t_stat: float, df: float, alternative: str) -> float:
    if alternative == "two-sided":
        return float(2 * stats.t.sf(abs(t_stat), df))
    if alternative == "greater":
        return float(stats.t.sf(t_stat, df))
    if alternative == "less":
        return float(stats.t.cdf(t_stat, df))
    raise ValueError("alternative должен быть 'two-sided', 'greater' или 'less'")


def welch_ttest(
    control, treatment, alpha: float = 0.05, alternative: str = "two-sided"
) -> WelchResult:
    """Тест Уэлча (не предполагает равенства дисперсий). Эффект = treatment - control.

    t = (m_T - m_C) / sqrt(s_C^2/n_C + s_T^2/n_T)
    df = (a + b)^2 / (a^2/(n_C-1) + b^2/(n_T-1)),  a = s_C^2/n_C,  b = s_T^2/n_T
    """
    c, t = _as_array(control), _as_array(treatment)
    a, b = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    se = math.sqrt(a + b)
    if se == 0:
        raise ValueError("нулевая дисперсия в обеих группах: тест не определён")
    diff = float(t.mean() - c.mean())
    t_stat = diff / se
    df = (a + b) ** 2 / (a**2 / (c.size - 1) + b**2 / (t.size - 1))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    return WelchResult(
        n_control=int(c.size),
        n_treatment=int(t.size),
        mean_control=float(c.mean()),
        mean_treatment=float(t.mean()),
        diff=diff,
        se=se,
        t=t_stat,
        df=float(df),
        p_value=_t_p_value(t_stat, df, alternative),
        alternative=alternative,
        alpha=alpha,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        assumption_issues=tuple(clt_issues(c, "control") + clt_issues(t, "treatment")),
    )


def mann_whitney(control, treatment, alternative: str = "two-sided") -> MannWhitneyResult:
    """Манн-Уитни. Проверяет СТОХАСТИЧЕСКОЕ превосходство, а не равенство средних."""
    c, t = _as_array(control), _as_array(treatment)
    res = stats.mannwhitneyu(t, c, alternative=alternative, method="asymptotic")
    return MannWhitneyResult(
        n_control=int(c.size),
        n_treatment=int(t.size),
        u_treatment=float(res.statistic),
        p_value=float(res.pvalue),
        prob_superiority=float(res.statistic / (t.size * c.size)),
        alternative=alternative,
    )


def bootstrap_mean_diff_ci(
    control,
    treatment,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    max_cells: int = 5_000_000,
) -> BootstrapResult:
    """Percentile-bootstrap ДИ для разницы средних. Пересчёт порциями ограничивает память."""
    c, t = _as_array(control), _as_array(treatment)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    chunk = max(1, max_cells // max(c.size, t.size))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        mean_c = c[rng.integers(0, c.size, size=(k, c.size))].mean(axis=1)
        mean_t = t[rng.integers(0, t.size, size=(k, t.size))].mean(axis=1)
        diffs[done : done + k] = mean_t - mean_c
        done += k
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(t.mean() - c.mean()), float(lo), float(hi), n_boot, seed, alpha)


def methods_agree(welch: WelchResult, boot: BootstrapResult) -> bool:
    """Заранее заданное правило: Уэлч и bootstrap должны давать одно решение и один знак."""
    welch_sig = welch.p_value < welch.alpha
    boot_sig = not (boot.ci_low <= 0.0 <= boot.ci_high)
    if welch_sig != boot_sig:
        return False
    return (not welch_sig) or bool(np.sign(welch.diff) == np.sign(boot.diff))
