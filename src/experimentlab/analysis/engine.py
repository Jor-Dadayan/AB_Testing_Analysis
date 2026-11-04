"""Связывает модули анализа в один воспроизводимый прогон. Не содержит формул тестов."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, is_dataclass

import pandas as pd

from experimentlab.analysis import (
    conversion,
    descriptive,
    hypothesis,
    interpretation,
    power,
    revenue,
)
from experimentlab.analysis.input_checks import validate_analysis_frame

DESCRIBED = ["age", "sessions_prior_30d", "cart_items", "cart_value", "revenue"]


@dataclass
class AnalysisResult:
    meta: dict
    srm: conversion.SRMResult
    descriptive: pd.DataFrame
    revenue_distribution: dict
    conversion: conversion.ProportionTestResult
    conversion_verdict: interpretation.EffectVerdict
    revenue_summary: dict
    arpu_welch: revenue.WelchResult
    arpu_mwu: revenue.MannWhitneyResult
    arpu_bootstrap: revenue.BootstrapResult
    arpu_methods_agree: bool
    arpu_verdict: interpretation.EffectVerdict
    arppu_welch: revenue.WelchResult
    power: dict
    sufficiency: dict
    recommendation: str
    recommendation_text: str
    hypotheses: list

    def to_dict(self) -> dict:
        """Словарь для JSON (dashboard будет читать его, а не пересчитывать)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, pd.DataFrame):
                v = v.reset_index().to_dict(orient="records")
            elif is_dataclass(v):
                v = asdict(v)
            elif isinstance(v, list):
                v = [asdict(i) if is_dataclass(i) else i for i in v]
            out[f.name] = v
        return out


def run_analysis(df: pd.DataFrame, cfg: dict, source: str = "") -> AnalysisResult:
    """Полный анализ. df: одна строка на пользователя, столбцы group_name, converted, revenue."""
    validate_analysis_frame(df)
    a = cfg["analysis"]
    alpha, target_power = float(a["alpha"]), float(a["power"])
    thr_conv = float(a["practical_threshold_abs"])
    planned_mde = float(a["planned_mde_abs"])

    d = df.assign(group_name=df["group_name"].astype(str))
    ctrl, treat = d[d["group_name"] == "control"], d[d["group_name"] == "treatment"]
    if ctrl.empty or treat.empty:
        raise ValueError("в данных должны быть обе группы: control и treatment")
    n_c, n_t = len(ctrl), len(treat)
    payers = d[d["converted"] == 1]

    # 0. Корректность эксперимента
    srm = conversion.srm_check(n_c, n_t, float(cfg["treatment_share"]), float(a["srm_alpha"]))

    # 1. Описательная статистика
    desc = pd.concat(
        [
            descriptive.describe_by_group(d, DESCRIBED),
            descriptive.describe_by_group(payers, ["revenue"]).rename(
                index={"revenue": "revenue_payers"}, level="variable"
            ),
        ]
    )
    dist = {
        g: {
            "all": descriptive.distribution_diagnostics(part["revenue"]),
            "payers": descriptive.distribution_diagnostics(
                part.loc[part["converted"] == 1, "revenue"]
            ),
        }
        for g, part in d.groupby("group_name")
    }

    # 2. Конверсия (первичная метрика)
    conv = conversion.two_proportion_ztest(
        int(ctrl["converted"].sum()), n_c, int(treat["converted"].sum()), n_t, alpha=alpha
    )
    conv_verdict = interpretation.classify_effect(
        conv.diff,
        conv.ci_low,
        conv.ci_high,
        conv.p_value,
        alpha,
        thr_conv,
        fmt=interpretation.fmt_pp,
    )

    # 3. Выручка (вторичная метрика)
    rev_c = ctrl["revenue"].to_numpy(float)
    rev_t = treat["revenue"].to_numpy(float)
    summary = revenue.revenue_summary(rev_c, rev_t)
    welch = revenue.welch_ttest(rev_c, rev_t, alpha)
    mwu = revenue.mann_whitney(rev_c, rev_t)
    boot = revenue.bootstrap_mean_diff_ci(
        rev_c, rev_t, int(a["n_bootstrap"]), alpha, int(a["bootstrap_seed"])
    )
    agree = revenue.methods_agree(welch, boot)
    thr_arpu = float(a["practical_threshold_arpu_rel"]) * summary["control"]["arpu"]
    arpu_verdict = interpretation.classify_effect(
        welch.diff, welch.ci_low, welch.ci_high, welch.p_value, alpha, thr_arpu
    )
    arppu_welch = revenue.welch_ttest(rev_c[rev_c > 0], rev_t[rev_t > 0], alpha)

    # 4. Мощность и планирование
    share = float(cfg["treatment_share"])
    plan_ratio = share / (1 - share)
    baseline = float(cfg["baseline_conversion"])
    daily_users = len(d) / d["signup_date"].nunique()
    plan = []
    for key, label, delta in (
        ("planned_mde", "Плановый MDE", planned_mde),
        ("practical_threshold", "Практический порог", thr_conv),
    ):
        n_req = power.sample_size_two_proportions(
            baseline, baseline + delta, alpha, target_power, plan_ratio
        )
        plan.append(
            {
                "key": key,
                "label": label,
                "delta_abs": delta,
                "p_control": baseline,
                "p_treatment": baseline + delta,
                "n_control": n_req,
                "n_total": n_req + math.ceil(n_req * plan_ratio),
                "days_raw": power.estimate_duration_days(
                    n_req, daily_users, plan_ratio, round_to_full_weeks=False
                ),
                "days_full_weeks": power.estimate_duration_days(n_req, daily_users, plan_ratio),
            }
        )
    ratio_obs = n_t / n_c
    mde_actual = power.mde_two_proportions(conv.p_control, n_c, alpha, target_power, ratio_obs)
    revenue_sd = float(d["revenue"].std(ddof=1))
    mde_arpu = power.mde_two_means(revenue_sd, n_c, alpha, target_power, ratio_obs)
    power_info = {
        "plan": plan,
        "baseline_assumed": baseline,
        "daily_users": daily_users,
        "n_control_actual": n_c,
        "power_at_planned_mde": power.power_two_proportions(
            baseline, baseline + planned_mde, n_c, alpha, ratio_obs
        ),
        "mde_actual_abs": mde_actual,
        "mde_actual_rel": mde_actual / conv.p_control,
        "arpu_sd": revenue_sd,
        "mde_arpu_abs": mde_arpu,
        "mde_arpu_rel": mde_arpu / summary["control"]["arpu"],
    }
    suff = interpretation.sufficiency(n_c, plan[0]["n_control"], mde_actual, thr_conv, planned_mde)

    code, text = interpretation.recommend(
        srm.passed, conv_verdict, arpu_verdict, agree, suff["data_sufficient"]
    )

    meta = {
        "source": source,
        "n_users": len(d),
        "n_control": n_c,
        "n_treatment": n_t,
        "period_start": str(pd.to_datetime(d["signup_date"]).min().date()),
        "period_end": str(pd.to_datetime(d["signup_date"]).max().date()),
        "alpha": alpha,
        "power": target_power,
        **hypothesis.error_rates(alpha, target_power),
        "practical_threshold_abs": thr_conv,
        "practical_threshold_arpu": thr_arpu,
        "planned_mde_abs": planned_mde,
    }
    return AnalysisResult(
        meta=meta,
        srm=srm,
        descriptive=desc,
        revenue_distribution=dist,
        conversion=conv,
        conversion_verdict=conv_verdict,
        revenue_summary=summary,
        arpu_welch=welch,
        arpu_mwu=mwu,
        arpu_bootstrap=boot,
        arpu_methods_agree=agree,
        arpu_verdict=arpu_verdict,
        arppu_welch=arppu_welch,
        power=power_info,
        sufficiency=suff,
        recommendation=code,
        recommendation_text=text,
        hypotheses=hypothesis.build_hypotheses(),
    )
