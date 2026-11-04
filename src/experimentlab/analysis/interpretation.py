"""Автоматические статистические выводы. Правила заданы заранее и не зависят от результата."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def fmt_pp(v: float) -> str:
    """Знаковый формат процентных пунктов."""
    return f"{v * 100:+.2f} п.п."


def fmt_pp_abs(v: float) -> str:
    return f"{v * 100:.2f} п.п."


def fmt_money(v: float) -> str:
    return f"{v:+.4f}"


@dataclass(frozen=True)
class EffectVerdict:
    significant: bool
    direction: str  # positive / negative / none
    ci_includes_zero: bool
    practical: (
        str  # practical / uncertain / negligible / harmful / no_practical_effect / inconclusive
    )
    summary: str


def p_value_statement(p_value: float, alpha: float) -> str:
    """Корректная формулировка смысла p-value."""
    verdict = "меньше" if p_value < alpha else "не меньше"
    return (
        f"p = {p_value:.4g} ({verdict} α = {alpha}). Это вероятность получить разницу не меньше "
        "наблюдаемой, ЕСЛИ H0 верна и допущения теста выполнены. "
        "Это не вероятность того, что H0 верна, и не вероятность того, что эффект случаен."
    )


def classify_effect(
    diff: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    alpha: float,
    threshold: float,
    fmt: Callable[[float], str] = fmt_money,
) -> EffectVerdict:
    """Разделяет статистическую и практическую значимость.

    Практическая значимость оценивается по ДОВЕРИТЕЛЬНОМУ ИНТЕРВАЛУ относительно порога,
    а не по точечной оценке: оценка может быть выше порога случайно.
    """
    significant = bool(p_value < alpha)
    includes_zero = bool(ci_low <= 0.0 <= ci_high)
    direction = "positive" if diff > 0 else "negative" if diff < 0 else "none"
    level = f"{1 - alpha:.0%}"
    size = f"оценка {fmt(diff)}, {level} ДИ [{fmt(ci_low)}; {fmt(ci_high)}]"

    if significant and diff > 0:
        if ci_low >= threshold:
            practical = "practical"
            text = (
                f"Эффект статистически значим и положителен ({size}). Весь интервал лежит "
                f"выше практического порога {fmt(threshold)}: эффект практически значим."
            )
        elif ci_high < threshold:
            practical = "negligible"
            text = (
                f"Эффект статистически значим и положителен ({size}), но даже верхняя граница "
                f"интервала ниже порога {fmt(threshold)}: практически он незначим."
            )
        else:
            practical = "uncertain"
            text = (
                f"Эффект статистически значим и положителен ({size}). Интервал включает "
                f"значения и выше, и ниже порога {fmt(threshold)}: практическая значимость "
                "не подтверждена."
            )
        if includes_zero:
            text += (
                " Замечание: p-value и интервал расходятся на границе значимости "
                "(тест использует объединённую дисперсию, интервал раздельную)."
            )
    elif significant and diff < 0:
        practical = "harmful"
        text = f"Эффект статистически значим и отрицателен ({size}): новая версия ухудшает метрику."
    elif -threshold < ci_low and ci_high < threshold:
        practical = "no_practical_effect"
        text = (
            f"Статистически значимого эффекта не обнаружено ({size}). Это не доказательство "
            f"равенства групп, но весь интервал лежит в пределах [{fmt(-threshold)}; "
            f"{fmt(threshold)}]: эффекты крупнее практического порога данными не поддерживаются."
        )
    else:
        practical = "inconclusive"
        text = (
            f"Статистически значимого эффекта не обнаружено ({size}). Интервал шире "
            f"практического порога ±{fmt(threshold).lstrip('+')}: данные не позволяют ни "
            "подтвердить, ни исключить практически значимый эффект. "
            "Отсутствие значимости не означает отсутствия эффекта."
        )
    return EffectVerdict(significant, direction, includes_zero, practical, text)


def sufficiency(
    n_control: int,
    n_required: int,
    mde_actual: float,
    threshold: float,
    planned_mde: float,
) -> dict:
    """Достаточно ли данных: по плановому MDE и по способности увидеть практический порог."""
    enough = n_control >= n_required
    can_detect = mde_actual <= threshold
    text = (
        f"Для планового MDE {fmt_pp_abs(planned_mde)} при заданных α и мощности нужно не менее "
        f"{n_required:,} пользователей на группу; в тесте {n_control:,}: данных "
        f"{'достаточно' if enough else 'недостаточно'}. "
        f"Минимальный эффект, обнаружимый на фактическом объёме с целевой мощностью, равен "
        f"{fmt_pp_abs(mde_actual)}; практический порог {fmt_pp_abs(threshold)}: "
        + (
            "тест способен заметить эффект такого размера."
            if can_detect
            else "тест может не заметить эффект такого размера."
        )
    )
    return {"data_sufficient": enough, "can_detect_threshold": can_detect, "text": text}


def _arpu_note(arpu: EffectVerdict, methods_agree: bool) -> str:
    if not methods_agree:
        return (
            "Вывод по выручке не делается: тест Уэлча и bootstrap дают разные решения, "
            "нужно разбираться с хвостом распределения."
        )
    if arpu.significant and arpu.direction == "positive":
        return "ARPU также значимо вырос."
    return "Значимого изменения ARPU не обнаружено (это не доказывает, что его нет)."


def recommend(
    srm_passed: bool,
    conversion: EffectVerdict,
    arpu: EffectVerdict,
    arpu_methods_agree: bool,
    data_sufficient: bool,
) -> tuple[str, str]:
    """Итоговая рекомендация по заранее заданным правилам."""
    if not srm_passed:
        return (
            "INVALID_EXPERIMENT",
            "Соотношение групп отличается от плана (SRM). Результаты нельзя интерпретировать, "
            "пока не найдена причина (ошибка назначения, потеря данных, фильтрация).",
        )
    p = conversion.practical
    note = _arpu_note(arpu, arpu_methods_agree)
    if p == "harmful":
        return "DO_NOT_SHIP", "Новая версия значимо снижает конверсию. Внедрять не следует."
    if p == "practical":
        if arpu.significant and arpu.direction == "negative" and arpu_methods_agree:
            return (
                "INVESTIGATE",
                "Конверсия растёт, но ARPU значимо падает: сначала разобраться, "
                "за счёт чего снижается средний чек.",
            )
        return "SHIP", f"Рост конверсии статистически и практически значим. {note}"
    if p == "uncertain":
        return (
            "SHIP_WITH_MONITORING",
            "Рост конверсии статистически значим, но его практическая величина не подтверждена. "
            f"Внедрение оправдано, если оно дёшево и обратимо; следить за метрикой. {note}",
        )
    if p == "negligible":
        return "DO_NOT_SHIP", "Эффект есть, но он слишком мал, чтобы окупить внедрение."
    if p == "no_practical_effect":
        return (
            "DO_NOT_SHIP",
            "Практически значимого эффекта данные не поддерживают: нет оснований внедрять "
            "ради роста конверсии.",
        )
    extra = "" if data_sufficient else " Объём данных меньше планового."
    return "NEED_MORE_DATA", f"Результат неопределённый: нужно больше данных.{extra}"
