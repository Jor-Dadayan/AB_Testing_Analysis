"""Сегментный анализ конверсии: заранее заданные сегменты, поправки, омнибус-тест."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from experimentlab.analysis.conversion import two_proportion_ztest
from experimentlab.analysis.hypothesis import holm_adjust
from experimentlab.analysis.multiple_testing import compare_corrections

LEVEL_LABELS = {"is_new_user": {0: "returning", 1: "new"}}
HET_COLUMNS = ["dimension", "n_levels", "lr_stat", "df", "p_value", "p_holm", "significant_holm"]
TABLE_COLUMNS = [
    "dimension",
    "level",
    "n_control",
    "n_treatment",
    "x_control",
    "x_treatment",
    "p_control",
    "p_treatment",
    "diff",
    "ci_low",
    "ci_high",
    "z",
    "p_value",
    "p_bonferroni",
    "p_holm",
    "q_bh",
    "sig_raw",
    "sig_bonferroni",
    "sig_holm",
    "sig_bh",
    "ci_low_bonf",
    "ci_high_bonf",
]


@dataclass
class SegmentAnalysis:
    table: pd.DataFrame
    heterogeneity: pd.DataFrame
    skipped: list
    n_tests: int
    alpha: float
    fdr_q: float
    expected_false_positives: float


def segment_labels(df: pd.DataFrame, dimension: str) -> pd.Series:
    """Человекочитаемые уровни сегмента (строки)."""
    s = df[dimension]
    if dimension in LEVEL_LABELS:
        return s.astype(int).map(LEVEL_LABELS[dimension])
    return s.astype(str)


def analyze_segments(
    df: pd.DataFrame,
    dimensions: list[str],
    alpha: float = 0.05,
    min_users: int = 500,
    fdr_q: float = 0.05,
    group_col: str = "group_name",
    run_heterogeneity: bool = True,
) -> SegmentAnalysis:
    """Эффект на конверсию по каждому уровню каждого заранее заданного признака."""
    is_t = df[group_col].astype(str).eq("treatment")
    y = df["converted"].astype(int)
    rows, skipped, labels_by_dim = [], [], {}
    for dim in dimensions:
        labels = segment_labels(df, dim)
        labels_by_dim[dim] = labels
        for level in sorted(labels.unique()):
            in_seg = labels.eq(level)
            c, t = in_seg & ~is_t, in_seg & is_t
            n_c, n_t = int(c.sum()), int(t.sum())
            x_c, x_t = int(y[c].sum()), int(y[t].sum())
            reason = None
            if min(n_c, n_t) < min_users:
                reason = f"меньше {min_users} пользователей в группе"
            elif min(x_c, n_c - x_c, x_t, n_t - x_t) < 10:
                reason = "меньше 10 успехов или неудач"
            if reason:
                skipped.append(
                    {
                        "dimension": dim,
                        "level": level,
                        "n_control": n_c,
                        "n_treatment": n_t,
                        "reason": reason,
                    }
                )
                continue
            r = two_proportion_ztest(x_c, n_c, x_t, n_t, alpha=alpha)
            rows.append(
                {
                    "dimension": dim,
                    "level": level,
                    "n_control": n_c,
                    "n_treatment": n_t,
                    "x_control": x_c,
                    "x_treatment": x_t,
                    "p_control": r.p_control,
                    "p_treatment": r.p_treatment,
                    "diff": r.diff,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "z": r.z,
                    "p_value": r.p_value,
                }
            )

    table = pd.DataFrame(rows)
    m = len(table)
    if m:
        corr = compare_corrections(
            table["dimension"] + ": " + table["level"], table["p_value"], alpha, fdr_q
        )
        table = pd.concat([table, corr.drop(columns=["label", "p_raw"])], axis=1)
        adj = [
            two_proportion_ztest(
                r.x_control, r.n_control, r.x_treatment, r.n_treatment, alpha=alpha / m
            )
            for r in table.itertuples()
        ]
        table["ci_low_bonf"] = [a.ci_low for a in adj]
        table["ci_high_bonf"] = [a.ci_high for a in adj]
    else:
        table = pd.DataFrame(columns=TABLE_COLUMNS)

    het_rows = []
    if run_heterogeneity and m:
        for dim in dimensions:
            kept = set(table.loc[table["dimension"] == dim, "level"])
            if len(kept) < 2:
                continue
            labels = labels_by_dim[dim]
            mask = labels.isin(kept)
            d = pd.DataFrame(
                {
                    "y": y[mask].to_numpy(),
                    "t": is_t[mask].astype(int).to_numpy(),
                    "s": labels[mask].to_numpy(),
                }
            )
            m0 = smf.logit("y ~ t + C(s)", d).fit(disp=0)
            m1 = smf.logit("y ~ t * C(s)", d).fit(disp=0)
            lr = max(0.0, 2 * (m1.llf - m0.llf))
            dfree = int(round(m1.df_model - m0.df_model))
            het_rows.append(
                {
                    "dimension": dim,
                    "n_levels": len(kept),
                    "lr_stat": lr,
                    "df": dfree,
                    "p_value": float(stats.chi2.sf(lr, dfree)),
                }
            )
    het = pd.DataFrame(het_rows, columns=HET_COLUMNS)
    if len(het):
        het["p_holm"] = holm_adjust(het["p_value"].to_numpy())
        het["significant_holm"] = het["p_holm"] < alpha

    return SegmentAnalysis(table, het, skipped, m, alpha, fdr_q, m * alpha)


def significant_levels(
    table: pd.DataFrame, rule: str = "holm", direction: str = "negative"
) -> list[str]:
    """Сегменты со значимой (после поправки rule) разницей нужного знака."""
    if table.empty:
        return []
    col = {"raw": "sig_raw", "bonferroni": "sig_bonferroni", "holm": "sig_holm", "bh": "sig_bh"}[
        rule
    ]
    sign = table["diff"] < 0 if direction == "negative" else table["diff"] > 0
    mask = table[col].astype(bool) & sign
    return (table["dimension"] + ": " + table["level"])[mask].tolist()
