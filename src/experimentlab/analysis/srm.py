"""Расширенная проверка SRM: отчёт с предупреждением и проверка по сегментам и дням."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from experimentlab.analysis.conversion import srm_check


@dataclass(frozen=True)
class SRMReport:
    n_control: int
    n_treatment: int
    expected_control: float
    expected_treatment: float
    expected_treatment_share: float
    observed_treatment_share: float
    chi2: float
    df: int
    p_value: float
    alpha: float
    passed: bool
    warning: str | None


def srm_report(
    n_control: int,
    n_treatment: int,
    expected_treatment_share: float = 0.5,
    alpha: float = 0.001,
) -> SRMReport:
    """Chi-square goodness-of-fit: фактические размеры групп против плана."""
    base = srm_check(n_control, n_treatment, expected_treatment_share, alpha)
    n = n_control + n_treatment
    exp_t = n * expected_treatment_share
    warning = None
    if not base.passed:
        warning = (
            f"ВНИМАНИЕ: обнаружен SRM. Ожидалось {exp_t:,.0f} (treatment) и {n - exp_t:,.0f} "
            f"(control), получено {n_treatment:,} и {n_control:,} "
            f"(χ² = {base.chi2:.1f}, p = {base.p_value:.2g} < {alpha}). "
            "Результаты НЕЛЬЗЯ интерпретировать, пока не найдена причина: ошибка назначения "
            "групп, потеря или фильтрация данных в одной группе, боты, разная скорость "
            "загрузки версий."
        )
    return SRMReport(
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        expected_control=n - exp_t,
        expected_treatment=exp_t,
        expected_treatment_share=expected_treatment_share,
        observed_treatment_share=n_treatment / n,
        chi2=base.chi2,
        df=1,
        p_value=base.p_value,
        alpha=alpha,
        passed=base.passed,
        warning=warning,
    )


def srm_by_segment(
    df: pd.DataFrame,
    columns: list[str],
    expected_treatment_share: float = 0.5,
    family_alpha: float = 0.01,
    group_col: str = "group_name",
) -> pd.DataFrame:
    """SRM внутри каждого уровня каждого признака. Порог делится на число проверок (Bonferroni)."""
    cols = [
        "dimension",
        "level",
        "n_control",
        "n_treatment",
        "share_treatment",
        "chi2",
        "p_value",
        "alpha_adj",
        "flagged",
    ]
    rows = []
    for col in columns:
        counts = pd.crosstab(df[col].astype(str), df[group_col].astype(str))
        for level, r in counts.iterrows():
            n_c, n_t = int(r.get("control", 0)), int(r.get("treatment", 0))
            n = n_c + n_t
            exp_t = n * expected_treatment_share
            if min(exp_t, n - exp_t) < 5:  # хи-квадрат при малых ожидаемых частотах ненадёжен
                continue
            res = srm_check(n_c, n_t, expected_treatment_share, family_alpha)
            label = str(level)[:10] if col.endswith("date") else str(level)
            rows.append(
                {
                    "dimension": col,
                    "level": label,
                    "n_control": n_c,
                    "n_treatment": n_t,
                    "share_treatment": n_t / n,
                    "chi2": res.chi2,
                    "p_value": res.p_value,
                }
            )
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out["alpha_adj"] = family_alpha / len(out)
        out["flagged"] = out["p_value"] < out["alpha_adj"]
    return out
