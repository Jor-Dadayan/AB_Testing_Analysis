"""Качество эксперимента: баланс, достаточность выборки, длительность, стабильность, дизайн."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from experimentlab.analysis import eda
from experimentlab.analysis.conversion import p_value_from_z, two_proportion_ztest

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2, "ok": 3}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # critical / warning / info / ok
    message: str


@dataclass
class QualityReport:
    findings: list
    balance: pd.DataFrame
    stability: dict

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)


def check_balance(df, numeric, categorical, warn=0.10, critical=0.25):
    num = [c for c in numeric if c in df.columns]
    cat = [c for c in categorical if c in df.columns]
    table = eda.balance_table(df, num, cat)
    worst = table.iloc[0]
    if worst["abs_smd"] >= critical:
        sev, msg = "critical", (f"сильный дисбаланс групп: {worst['variable']} "
                                f"(|SMD| = {worst['abs_smd']:.2f}); рандомизация, вероятно, нарушена")
    elif worst["abs_smd"] >= warn:
        sev, msg = "warning", (f"заметный дисбаланс: {worst['variable']} "
                               f"(|SMD| = {worst['abs_smd']:.2f}); проверить назначение групп")
    else:
        sev, msg = "ok", f"группы сбалансированы: максимум |SMD| = {worst['abs_smd']:.3f} ({worst['variable']})"
    return [Finding("balance", sev, msg)], table


def check_sample(n_control, n_required, mde_actual, threshold, mde_arpu_rel, arpu_threshold_rel):
    out = []
    if n_control < n_required:
        out.append(Finding("sample_planned", "warning",
                           f"в control {n_control:,} пользователей, для планового MDE нужно {n_required:,}"))
    else:
        out.append(Finding("sample_planned", "ok",
                           f"объём достаточен для планового MDE: {n_control:,} ≥ {n_required:,}"))
    if mde_actual > threshold:
        out.append(Finding("sample_practical", "warning",
                           f"MDE на фактическом объёме {mde_actual * 100:.2f} п.п. больше практического "
                           f"порога {threshold * 100:.2f} п.п.: эффект порогового размера тест может не заметить"))
    if mde_arpu_rel > arpu_threshold_rel:
        out.append(Finding("sample_revenue", "info",
                           f"MDE по ARPU {mde_arpu_rel * 100:.1f}% выше порога {arpu_threshold_rel * 100:.0f}%: "
                           "выручка шумнее конверсии, малые изменения ARPU тест не заметит"))
    return out


def check_duration(dates, required_days: int, min_days: int = 14) -> list[Finding]:
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    span = int((d.max() - d.min()).days) + 1
    with_data = int(d.nunique())
    out = []
    if span % 7:
        out.append(Finding("duration_weeks", "warning",
                           f"длительность {span} дн. не кратна 7: дни недели представлены неравномерно"))
    if span < min_days:
        out.append(Finding("duration_min", "warning",
                           f"длительность {span} дн. меньше {min_days}: нет двух полных недельных циклов"))
    if span < required_days:
        out.append(Finding("duration_required", "warning",
                           f"длительность {span} дн. меньше расчётной {required_days} дн."))
    if with_data < span:
        out.append(Finding("duration_gaps", "warning",
                           f"данные есть только за {with_data} из {span} дней"))
    return out or [Finding("duration", "ok", f"{span} дн., без пропусков, кратно неделям")]


def check_stability(df, alpha=0.05, group_col="group_name"):
    """Один заранее заданный тест: одинаков ли эффект в первой и второй половине периода."""
    dates = pd.to_datetime(df["signup_date"]).dt.normalize()
    first = dates.min()
    span = int((dates.max() - first).days) + 1
    if span < 4:
        return Finding("stability", "info", "период слишком короткий для проверки стабильности"), {}
    mid = first + pd.Timedelta(days=span // 2)
    parts = []
    for mask in (dates < mid, dates >= mid):
        d = df[mask]
        is_t = d[group_col].astype(str).eq("treatment")
        y = d["converted"].astype(int)
        parts.append(two_proportion_ztest(int(y[~is_t].sum()), int((~is_t).sum()),
                                          int(y[is_t].sum()), int(is_t.sum())))
    early, late = parts
    z = (late.diff - early.diff) / math.sqrt(early.se_unpooled**2 + late.se_unpooled**2)
    p = p_value_from_z(z)
    info = {"diff_early": early.diff, "diff_late": late.diff, "z": z, "p_value": p}
    txt = (f"эффект в первой половине {early.diff * 100:+.2f} п.п., во второй {late.diff * 100:+.2f} п.п. "
           f"(p = {p:.3f})")
    if p < alpha and late.diff < early.diff:
        return Finding("stability", "warning", "эффект ослабевает со временем (возможен эффект новизны): " + txt), info
    if p < alpha:
        return Finding("stability", "info", "эффект усиливается со временем: " + txt), info
    return Finding("stability", "ok", "нет статистических признаков изменения эффекта (проверка маломощна): " + txt), info


def check_design(cfg: dict, p_control_observed: float) -> list[Finding]:
    out = []
    share = float(cfg["treatment_share"])
    if abs(share - 0.5) > 1e-9:
        out.append(Finding("design_allocation", "info",
                           f"доля treatment {share:.0%}: неравное разбиение снижает мощность"))
    base = float(cfg["baseline_conversion"])
    if abs(p_control_observed - base) > 0.01:
        out.append(Finding("design_baseline", "warning",
                           f"план мощности опирался на конверсию {base:.1%}, в control наблюдается "
                           f"{p_control_observed:.1%}: расчёты размера выборки нужно пересмотреть"))
    return out or [Finding("design", "ok", "допущения дизайна согласуются с данными")]


def assess_experiment_quality(df, cfg, base, srm) -> QualityReport:
    """base: AnalysisResult этапа 5, srm: SRMReport."""
    adv, a = cfg["advanced"], cfg["analysis"]
    findings = [Finding("srm", "ok", f"SRM пройдена (p = {srm.p_value:.3f})") if srm.passed
                else Finding("srm", "critical", srm.warning or "SRM")]
    bal_f, bal = check_balance(
        df, ["age", "sessions_prior_30d", "cart_items", "cart_value", "signup_hour"],
        ["country", "device_type", "traffic_source", "is_new_user", "age_group", "is_weekend"],
        float(adv["balance_smd_warn"]), float(adv["balance_smd_critical"]))
    findings += bal_f
    plan0, pw = base.power["plan"][0], base.power
    findings += check_sample(pw["n_control_actual"], plan0["n_control"], pw["mde_actual_abs"],
                             float(a["practical_threshold_abs"]), pw["mde_arpu_rel"],
                             float(a["practical_threshold_arpu_rel"]))
    findings += check_duration(df["signup_date"], plan0["days_full_weeks"], int(adv["min_duration_days"]))
    stab_f, stab = check_stability(df, float(adv["stability_alpha"]))
    findings.append(stab_f)
    findings += check_design(cfg, base.conversion.p_control)
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    return QualityReport(findings, bal, stab)
