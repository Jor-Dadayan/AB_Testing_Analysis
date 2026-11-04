"""Анализ чувствительности: разные методы, влияние выбросов, регрессионная поправка."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from experimentlab.analysis import revenue
from experimentlab.analysis.bootstrap import (
    bootstrap_arpu_diff,
    bootstrap_conversion_diff,
    excludes_zero,
)
from experimentlab.analysis.conversion import two_proportion_ztest

COVARIATES_CAT = ["device_type", "country", "traffic_source"]
COVARIATES_NUM = ["is_new_user", "age", "sessions_prior_30d", "cart_items"]


@dataclass(frozen=True)
class MethodEstimate:
    metric: str
    method: str
    in_family: bool  # входит ли в вердикт об устойчивости
    estimate: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    significant: bool
    note: str = ""


@dataclass
class SensitivityResult:
    estimates: list
    conversion_robust: bool
    conversion_reason: str
    arpu_robust: bool
    arpu_reason: str
    influence: dict


def adjusted_effect(df: pd.DataFrame, alpha: float = 0.05, group_col: str = "group_name"):
    """Линейная вероятностная модель с ковариатами ДО эксперимента. Возвращает (tau, low, high, p)."""
    d = pd.DataFrame({
        "y": df["converted"].astype(float).to_numpy(),
        "t": df[group_col].astype(str).eq("treatment").astype(float).to_numpy(),
    })
    terms = ["t"]
    for c in COVARIATES_CAT:
        if c in df.columns:
            d[c] = df[c].astype(str).to_numpy()
            terms.append(f"C({c})")
    for c in COVARIATES_NUM:
        if c in df.columns:
            d[c] = df[c].astype(float).to_numpy()
            terms.append(c)
    if "cart_value" in df.columns:
        d["log_cart_value"] = np.log(df["cart_value"].astype(float).to_numpy())
        terms.append("log_cart_value")
    fit = smf.ols("y ~ " + " + ".join(terms), d).fit(cov_type="HC1")
    lo, hi = fit.conf_int(alpha=alpha).loc["t"]
    return float(fit.params["t"]), float(lo), float(hi), float(fit.pvalues["t"])


def conversion_estimates(df, alpha, n_boot, seed, group_col="group_name") -> list[MethodEstimate]:
    is_t = df[group_col].astype(str).eq("treatment")
    y = df["converted"].astype(int)
    x_c, n_c = int(y[~is_t].sum()), int((~is_t).sum())
    x_t, n_t = int(y[is_t].sum()), int(is_t.sum())
    z = two_proportion_ztest(x_c, n_c, x_t, n_t, alpha=alpha)
    b = bootstrap_conversion_diff(x_c, n_c, x_t, n_t, n_boot, alpha, seed)
    tau, lo, hi, p = adjusted_effect(df, alpha, group_col)
    return [
        MethodEstimate("conversion", "Z-тест долей (первичный)", True, z.diff, z.ci_low,
                       z.ci_high, z.p_value, bool(z.p_value < alpha)),
        MethodEstimate("conversion", "Bootstrap (биномиальный)", True, b.diff, b.ci_low,
                       b.ci_high, None, excludes_zero(b)),
        MethodEstimate("conversion", "Регрессия с ковариатами (HC1)", True, tau, lo, hi, p,
                       bool(p < alpha), "эффект на вероятность покупки"),
    ]


def revenue_estimates(df, alpha, n_boot, seed, winsor_percentiles, group_col="group_name"):
    is_t = df[group_col].astype(str).eq("treatment")
    rc = df.loc[~is_t, "revenue"].to_numpy(float)
    rt = df.loc[is_t, "revenue"].to_numpy(float)
    out = []
    w = revenue.welch_ttest(rc, rt, alpha)
    out.append(MethodEstimate("arpu", "Тест Уэлча (вторичный)", True, w.diff, w.ci_low,
                              w.ci_high, w.p_value, bool(w.p_value < alpha)))
    b = bootstrap_arpu_diff(rc, rt, n_boot, alpha, seed)
    out.append(MethodEstimate("arpu", "Bootstrap по среднему", True, b.diff, b.ci_low,
                              b.ci_high, None, excludes_zero(b)))
    positive = np.concatenate([rc[rc > 0], rt[rt > 0]])
    for q in winsor_percentiles:
        cap = float(np.quantile(positive, q))
        ww = revenue.welch_ttest(np.minimum(rc, cap), np.minimum(rt, cap), alpha)
        n_cap = int((rc > cap).sum() + (rt > cap).sum())
        out.append(MethodEstimate(
            "arpu", f"Уэлч после винзоризации на {q:.1%} (потолок {cap:,.0f})", True,
            ww.diff, ww.ci_low, ww.ci_high, ww.p_value, bool(ww.p_value < alpha),
            f"ограничено значений: {n_cap}"))
    mw = revenue.mann_whitney(rc, rt)
    out.append(MethodEstimate("arpu", "Манн-Уитни (другой вопрос)", False,
                              mw.prob_superiority - 0.5, None, None, mw.p_value,
                              bool(mw.p_value < alpha), "оценка = P(Y_T > Y_C) + 0.5·P(=) − 0.5"))
    lw = revenue.welch_ttest(np.log1p(rc), np.log1p(rt), alpha)
    out.append(MethodEstimate("arpu", "Уэлч на ln(1 + выручка) (другой вопрос)", False,
                              lw.diff, lw.ci_low, lw.ci_high, lw.p_value,
                              bool(lw.p_value < alpha), "оценка в лог-единицах"))
    return out


def robustness(estimates: list[MethodEstimate], metric: str) -> tuple[bool, str]:
    """Заранее заданное правило: все методы одного вопроса дают одно решение и один знак."""
    fam = [e for e in estimates if e.metric == metric and e.in_family]
    if not fam:
        return False, "нет оценок"
    sig = [e.method for e in fam if e.significant]
    non = [e.method for e in fam if not e.significant]
    if sig and non:
        return False, (f"расхождение по значимости: значим — {', '.join(sig)}; "
                       f"не значим — {', '.join(non)}")
    if len({int(np.sign(e.estimate)) for e in fam}) > 1:
        return False, "оценки имеют разные знаки"
    verdict = "значим" if sig else "не значим"
    return True, f"все {len(fam)} метода(ов) дают одинаковый вывод ({verdict}) и один знак"


def outlier_influence(df: pd.DataFrame, group_col: str = "group_name") -> dict:
    """Какую долю выручки дают самые крупные клиенты и один максимальный чек."""
    out = {}
    for g in ("control", "treatment"):
        r = np.sort(df.loc[df[group_col].astype(str) == g, "revenue"].to_numpy(float))[::-1]
        total = float(r.sum())
        top = max(1, int(np.ceil(0.01 * r.size)))
        out[g] = {
            "top1pct_users_share": float(r[:top].sum() / total) if total > 0 else float("nan"),
            "max_order": float(r[0]),
            "max_order_share": float(r[0] / total) if total > 0 else float("nan"),
        }
    return out


def run_sensitivity(df, alpha, n_boot_revenue, n_boot_conversion, seed, winsor_percentiles,
                    group_col="group_name") -> SensitivityResult:
    est = conversion_estimates(df, alpha, n_boot_conversion, seed, group_col)
    est += revenue_estimates(df, alpha, n_boot_revenue, seed, winsor_percentiles, group_col)
    c_ok, c_why = robustness(est, "conversion")
    a_ok, a_why = robustness(est, "arpu")
    return SensitivityResult(est, c_ok, c_why, a_ok, a_why, outlier_influence(df, group_col))
