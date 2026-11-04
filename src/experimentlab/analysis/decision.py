"""Итоговое решение: успех = ВСЕ заранее заданные критерии, а не одна значимая метрика."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CRITERIA = {"require_sensitivity_robust": True, "block_on_confirmed_segment_harm": True}


@dataclass(frozen=True)
class Criterion:
    name: str
    status: str  # pass / fail / warn / na
    detail: str


@dataclass(frozen=True)
class FinalDecision:
    code: str
    text: str
    criteria: list
    warnings: list


def _guardrail_criterion(g) -> Criterion:
    name = f"Guardrail: {g.label}" + ("" if g.required else " (необязательный)")
    if g.status == "PASS":
        return Criterion(name, "pass", g.detail)
    if g.status == "FAIL":
        return Criterion(name, "fail", g.detail)
    if g.status == "INCONCLUSIVE":
        return Criterion(name, "warn", g.detail)
    return Criterion(name, "na", g.detail)


def decide(
    *,
    srm_passed,
    quality_findings,
    conversion,
    guardrails,
    conversion_robust,
    segment_harm_holm,
    segment_harm_bh,
    data_sufficient,
    criteria=None,
) -> FinalDecision:
    """conversion: EffectVerdict первичной метрики. Сегменты могут только остановить запуск."""
    cfg = {**DEFAULT_CRITERIA, **(criteria or {})}
    critical = [f for f in quality_findings if f.severity == "critical"]
    q_warn = [f for f in quality_findings if f.severity == "warning"]
    warns: list[str] = [f.message for f in q_warn]

    crit = [
        Criterion(
            "Корректность эксперимента (SRM)",
            "pass" if srm_passed else "fail",
            "соотношение групп соответствует плану" if srm_passed else "SRM обнаружен",
        ),
        Criterion(
            "Качество эксперимента",
            "fail" if critical else "warn" if q_warn else "pass",
            "; ".join(f.message for f in critical + q_warn) or "критических проблем нет",
        ),
        Criterion(
            "Первичная метрика: статистическая значимость",
            "pass" if conversion.significant and conversion.direction == "positive" else "fail",
            conversion.summary,
        ),
        Criterion(
            "Первичная метрика: практическая значимость",
            {"practical": "pass", "uncertain": "warn"}.get(conversion.practical, "fail"),
            f"классификация: {conversion.practical}",
        ),
        Criterion(
            "Устойчивость первичного результата",
            "pass"
            if conversion_robust
            else ("fail" if cfg["require_sensitivity_robust"] else "warn"),
            "методы согласуются" if conversion_robust else "методы расходятся",
        ),
    ]
    crit += [_guardrail_criterion(g) for g in guardrails]
    if segment_harm_holm:
        crit.append(
            Criterion(
                "Сегменты: подтверждённый вред",
                "fail" if cfg["block_on_confirmed_segment_harm"] else "warn",
                "значимое ухудшение после поправки Холма: " + "; ".join(segment_harm_holm),
            )
        )
    elif segment_harm_bh:
        crit.append(
            Criterion(
                "Сегменты: подтверждённый вред",
                "warn",
                "возможное ухудшение (только по BH, не по Холму): " + "; ".join(segment_harm_bh),
            )
        )
        warns.append("возможный вред в сегментах (BH): " + "; ".join(segment_harm_bh))
    else:
        crit.append(Criterion("Сегменты: подтверждённый вред", "pass", "значимого ухудшения нет"))

    guard_fail = [g for g in guardrails if g.status == "FAIL"]
    guard_open = [
        g for g in guardrails if g.required and g.status in ("INCONCLUSIVE", "NOT_EVALUATED")
    ]
    for g in guardrails:
        if g.status == "NOT_EVALUATED" and not g.required:
            warns.append(f"guardrail «{g.label}» не измерен: {g.detail}")

    p = conversion.practical
    if not srm_passed or critical:
        code, text = (
            "INVALID_EXPERIMENT",
            (
                "Эксперимент некорректен (SRM или критическая проблема качества). "
                "Результаты интерпретировать нельзя, пока не найдена причина."
            ),
        )
    elif p == "harmful":
        code, text = "DO_NOT_SHIP", "Новая версия значимо ухудшает конверсию."
    elif p in ("negligible", "no_practical_effect"):
        code, text = (
            "DO_NOT_SHIP",
            (
                "Практически значимого роста конверсии данные не поддерживают. "
                "Успехи в других метриках или сегментах этого не меняют."
            ),
        )
    elif p == "inconclusive":
        code, text = (
            "NEED_MORE_DATA",
            (
                "Результат по первичной метрике неопределённый"
                + ("" if data_sufficient else "; объём данных меньше планового")
                + ". Нужно больше данных."
            ),
        )
    elif guard_fail:
        code, text = (
            "INVESTIGATE",
            (
                "Конверсия выросла, но ухудшена guardrail-метрика: "
                + ", ".join(g.label for g in guard_fail)
                + ". Это компромисс, который решают владельцы продукта, а не автомат."
            ),
        )
    elif cfg["require_sensitivity_robust"] and not conversion_robust:
        code, text = (
            "INVESTIGATE",
            "Результат по конверсии неустойчив к выбору метода: разобраться до запуска.",
        )
    elif cfg["block_on_confirmed_segment_harm"] and segment_harm_holm:
        code, text = (
            "INVESTIGATE",
            (
                "Общий эффект положителен, но в сегментах подтверждён вред: "
                + "; ".join(segment_harm_holm)
            ),
        )
    else:
        code = "SHIP" if p == "practical" else "SHIP_WITH_MONITORING"
        text = (
            "Первичная метрика значима и практически значима, guardrail и проверки устойчивости пройдены."
            if code == "SHIP"
            else "Рост конверсии значим, но практическая величина не подтверждена: запуск оправдан, "
            "если он дёшев и обратим; следить за метрикой."
        )
        if code == "SHIP" and guard_open:
            code = "SHIP_WITH_MONITORING"
            text = (
                "Первичная метрика успешна, но обязательные guardrail не подтверждены ("
                + ", ".join(g.label for g in guard_open)
                + "): запуск только с мониторингом."
            )
    return FinalDecision(code, text, crit, warns)
