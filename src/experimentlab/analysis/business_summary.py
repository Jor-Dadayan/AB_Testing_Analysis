"""Бизнес-вывод простым языком и оценка эффекта в заказах и деньгах."""

from __future__ import annotations

from dataclasses import dataclass

from experimentlab.analysis.advanced_engine import AdvancedResult
from experimentlab.analysis.report import VERDICT_RU, pct, pp

DECISION_LABELS = {
    "SHIP": ("Запускать", "good"),
    "SHIP_WITH_MONITORING": ("Запускать с мониторингом", "warn"),
    "NEED_MORE_DATA": ("Нужно больше данных", "warn"),
    "INVESTIGATE": ("Разобраться перед запуском", "warn"),
    "DO_NOT_SHIP": ("Не запускать", "bad"),
    "INVALID_EXPERIMENT": ("Эксперимент некорректен", "bad"),
}


@dataclass(frozen=True)
class ImpactEstimate:
    monthly_users: float
    extra_orders: float
    extra_orders_low: float
    extra_orders_high: float
    extra_revenue: float
    extra_revenue_low: float
    extra_revenue_high: float
    assumption: str


@dataclass(frozen=True)
class BusinessSummary:
    code: str
    headline: str
    findings: list
    risks: list
    next_steps: list
    impact: ImpactEstimate
    markdown: str


def project_impact(result: AdvancedResult, monthly_users: float) -> ImpactEstimate:
    """Перенос наблюдённого эффекта и его интервала на месячный охват (при запуске на 100%)."""
    c, w = result.base.conversion, result.base.arpu_welch
    m = float(monthly_users)
    return ImpactEstimate(
        m,
        c.diff * m,
        c.ci_low * m,
        c.ci_high * m,
        w.diff * m,
        w.ci_low * m,
        w.ci_high * m,
        "Оценка предполагает, что эффект эксперимента сохранится на всей аудитории и во времени "
        "(в эксперименте проверено 4 недели). Интервалы отражают только статистическую "
        "неопределённость.",
    )


def _next_steps(result: AdvancedResult) -> list[str]:
    code = result.decision.code
    plan = result.base.power["plan"]
    if code == "SHIP":
        return [
            "Запланировать запуск новой версии для всей аудитории.",
            "Первые 2-4 недели следить за конверсией, ARPU и средним чеком.",
            "Подготовить план отката на случай ухудшения защитных метрик.",
        ]
    if code == "SHIP_WITH_MONITORING":
        return [
            "Запускать, только если внедрение недорогое и обратимое.",
            "Настроить мониторинг и заранее определить условия отката.",
            "Измерить метрики, которых не хватило (возвраты, отказы от оформления).",
        ]
    if code == "NEED_MORE_DATA":
        extra = ""
        if len(plan) > 1:
            extra = (
                f" Для обнаружения эффекта размером с практический порог нужно около "
                f"{plan[1]['n_control']:,} пользователей на группу "
                f"(примерно {plan[1]['days_full_weeks']} дней)."
            )
        return ["Продолжить эксперимент, не принимая решения по промежуточным данным." + extra]
    if code == "INVESTIGATE":
        return [
            "Выяснить причину ухудшения защитной метрики, неустойчивости результата "
            "или вреда в сегменте.",
            "Решение о компромиссе принимают владельцы продукта.",
            "При необходимости провести отдельный тест на проблемном сегменте.",
        ]
    if code == "DO_NOT_SHIP":
        return [
            "Оставить текущую версию страницы.",
            "Разобрать гипотезы улучшения и спланировать следующий тест.",
        ]
    return [
        "Найти причину нарушения (назначение групп, потеря данных, боты).",
        "Повторить эксперимент после исправления.",
    ]


def build_business_summary(
    result: AdvancedResult, monthly_users: float | None = None
) -> BusinessSummary:
    base, dec = result.base, result.decision
    c, w, meta, pw = base.conversion, base.arpu_welch, base.meta, base.power
    lvl = f"{1 - meta['alpha']:.0%}"
    label, _ = DECISION_LABELS[dec.code]
    users = float(monthly_users) if monthly_users else pw["daily_users"] * 30
    impact = project_impact(result, users)

    findings = [
        f"Конверсия: {pct(c.p_treatment)} в новой версии против {pct(c.p_control)} в старой, "
        f"разница {pp(c.diff)} ({lvl} ДИ [{pp(c.ci_low)}; {pp(c.ci_high)}]). "
        f"Вывод: {VERDICT_RU[base.conversion_verdict.practical]}.",
        f"Выручка на пользователя (ARPU): {w.mean_treatment:.3f} против {w.mean_control:.3f} "
        f"({w.diff:+.3f}; {lvl} ДИ [{w.ci_low:+.3f}; {w.ci_high:+.3f}]). "
        f"Вывод: {VERDICT_RU[base.arpu_verdict.practical]}."
        + (
            ""
            if base.arpu_methods_agree
            else " Методы оценки выручки расходятся, поэтому вывод по выручке не делается."
        ),
    ]
    evaluated = [g for g in result.guardrails if g.status != "NOT_EVALUATED"]
    n_pass = sum(g.status == "PASS" for g in evaluated)
    not_measured = [g.label for g in result.guardrails if g.status == "NOT_EVALUATED"]
    guard = f"Защитные метрики: подтверждены {n_pass} из {len(evaluated)} измеренных."
    if not_measured:
        guard += " Не измерены: " + ", ".join(not_measured) + "."
    findings.append(guard)
    n_crit = sum(f.severity == "critical" for f in result.quality.findings)
    n_warn = sum(f.severity == "warning" for f in result.quality.findings)
    findings.append(
        f"Качество эксперимента: критических проблем {n_crit}, предупреждений {n_warn}. "
        f"Разбиение групп (SRM): {'в норме' if result.srm.passed else 'НАРУШЕНО'}."
    )

    risks: list[str] = []
    for k in dec.criteria:
        if k.status in ("fail", "warn"):
            risks.append(f"{k.name}: {k.detail}"[:260])
    risks += [x for x in dec.warnings if x not in risks]
    risks.append(
        "Данные синтетические; долгосрочный эффект и эффект новизны за 4 недели не оцениваются."
    )

    steps = _next_steps(result)
    md = [
        "# Бизнес-вывод по эксперименту\n",
        f"**Рекомендация: {label}** (`{dec.code}`)\n",
        dec.text + "\n",
        "## Что мы выяснили\n",
    ]
    md += [f"- {x}" for x in findings]
    md += [
        "",
        "## Ожидаемый эффект при запуске на всю аудиторию\n",
        f"Охват: {impact.monthly_users:,.0f} пользователей в месяц.\n",
        f"- Дополнительные заказы в месяц: {impact.extra_orders:+,.0f} "
        f"({lvl} ДИ [{impact.extra_orders_low:+,.0f}; {impact.extra_orders_high:+,.0f}]).",
        f"- Изменение выручки в месяц: {impact.extra_revenue:+,.0f} "
        f"({lvl} ДИ [{impact.extra_revenue_low:+,.0f}; {impact.extra_revenue_high:+,.0f}]).",
        f"- {impact.assumption}",
        "",
        "## Риски и ограничения\n",
    ]
    md += [f"- {x}" for x in risks]
    md += ["", "## Следующие шаги\n"] + [f"{i}. {x}" for i, x in enumerate(steps, 1)]
    return BusinessSummary(dec.code, label, findings, risks, steps, impact, "\n".join(md) + "\n")
