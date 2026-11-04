"""Markdown-отчёт по AnalysisResult. Только форматирование, расчётов статистики здесь нет."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from scipy import stats

from experimentlab.analysis.conversion import p_value_from_z
from experimentlab.analysis.engine import AnalysisResult
from experimentlab.analysis.interpretation import p_value_statement
from experimentlab.config import PROJECT_ROOT

METHODS_PATH = PROJECT_ROOT / "docs" / "statistical_methods.md"

VERDICT_RU = {
    "practical": "значим статистически и практически",
    "uncertain": "значим статистически, практическая значимость не подтверждена",
    "negligible": "значим, но практически ничтожен",
    "harmful": "значимое ухудшение",
    "no_practical_effect": "значимости нет; практически значимый эффект не поддерживается",
    "inconclusive": "значимости нет; данных недостаточно для вывода",
}
DIRECTION_RU = {"positive": "положительный", "negative": "отрицательный", "none": "нулевой"}


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:.{digits}f}%"


def pp(v: float) -> str:
    return f"{v * 100:+.2f} п.п."


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _f(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.{digits}f}"


def _group_table(desc, variable: str, cols: list[str]) -> str:
    rows = []
    for g in ("control", "treatment"):
        rec = desc.loc[(variable, g)]
        rows.append([g] + [_f(rec[c], 0 if c == "count" else 2) for c in cols])
    return md_table(["группа"] + cols, rows)


def render_report(r: AnalysisResult, methods_path: Path = METHODS_PATH) -> str:
    L: list[str] = []
    add = L.append
    meta, c = r.meta, r.conversion
    cv, av = r.conversion_verdict, r.arpu_verdict
    w, mw, bt, ap = r.arpu_welch, r.arpu_mwu, r.arpu_bootstrap, r.arppu_welch
    alpha = meta["alpha"]
    lvl = f"{1 - alpha:.0%}"
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    sm = r.revenue_summary

    add("# ExperimentLab: отчёт статистического анализа\n")
    add(
        f"_Создан автоматически {datetime.now():%Y-%m-%d %H:%M}. Источник данных: {meta['source']}._\n"
    )
    add(
        f"Пользователей: {meta['n_users']:,} (control {meta['n_control']:,}, "
        f"treatment {meta['n_treatment']:,}). Период регистраций: "
        f"{meta['period_start']} - {meta['period_end']}.\n"
    )

    # ---- 1. Резюме
    add("## 1. Резюме\n")
    add(f"**Рекомендация: `{r.recommendation}`.** {r.recommendation_text}\n")
    arpu_flag = "" if r.arpu_methods_agree else " (методы расходятся, вывод не делается)"
    add(
        md_table(
            ["Метрика", "Control", "Treatment", "Разница (T - C)", f"{lvl} ДИ", "p-value", "Вывод"],
            [
                [
                    "Конверсия (первичная)",
                    pct(c.p_control),
                    pct(c.p_treatment),
                    pp(c.diff),
                    f"[{pp(c.ci_low)}; {pp(c.ci_high)}]",
                    fmt_p(c.p_value),
                    VERDICT_RU[cv.practical],
                ],
                [
                    "ARPU (вторичная)",
                    _f(w.mean_control, 4),
                    _f(w.mean_treatment, 4),
                    f"{w.diff:+.4f}",
                    f"[{w.ci_low:+.4f}; {w.ci_high:+.4f}]",
                    fmt_p(w.p_value),
                    VERDICT_RU[av.practical] + arpu_flag,
                ],
            ],
        )
    )
    add(
        "Решение принимается по первичной метрике (конверсия). ARPU служит контролем: "
        "рост конверсии не должен сопровождаться падением выручки на пользователя.\n"
    )

    # ---- 2. Гипотезы
    add("## 2. Гипотезы и параметры теста\n")
    add(
        md_table(
            ["Метрика", "Роль", "H0", "H1", "Тест"],
            [[h.metric, h.role, h.h0, h.h1, h.test] for h in r.hypotheses],
        )
    )
    add(
        f"- **Уровень значимости** α = {alpha}: заранее допустимая вероятность **ошибки I рода** "
        "(ложно найти эффект, которого нет)."
    )
    add(
        f"- **Целевая мощность** {meta['power']:.0%}, то есть вероятность **ошибки II рода** "
        f"β = {meta['type_2_error']:.2f} (пропустить реальный эффект планового размера)."
    )
    add(
        "- **Двусторонний тест** выбран заранее: новый дизайн может как улучшить, так и ухудшить "
        "метрику. Односторонний тест допустим только при заранее обоснованном направлении; "
        "выбирать его после просмотра данных нельзя."
    )
    add(
        "- Тесты не подбираются под результат: первичный тест один на метрику, остальные "
        "показаны как проверки устойчивости независимо от исхода.\n"
    )
    add("Для иллюстрации различия типов теста (на решение не влияет):\n")
    add(
        md_table(
            ["Альтернатива", "p-value"],
            [
                ["двусторонняя (p_T ≠ p_C), используется", fmt_p(p_value_from_z(c.z, "two-sided"))],
                ["односторонняя (p_T > p_C)", fmt_p(p_value_from_z(c.z, "greater"))],
                ["односторонняя (p_T < p_C)", fmt_p(p_value_from_z(c.z, "less"))],
            ],
        )
    )

    # ---- 3. SRM
    s = r.srm
    add("## 3. Проверка корректности: Sample Ratio Mismatch\n")
    add(
        f"План: доля treatment {s.expected_treatment_share:.0%}. Факт: control {s.n_control:,}, "
        f"treatment {s.n_treatment:,}. χ² = {s.chi2:.3f}, p = {fmt_p(s.p_value)} "
        f"(порог {s.alpha}). Результат: **{'пройдена' if s.passed else 'НЕ ПРОЙДЕНА'}**.\n"
    )

    # ---- 4. Описательная статистика
    add("## 4. Описательная статистика\n")
    add(
        "Показатели посчитаны отдельно для каждой группы. Мода у непрерывных величин "
        "малоинформативна, у выручки она равна 0 из-за большой доли нулей.\n"
    )
    centre = ["count", "mean", "median", "mode", "std", "variance", "min", "max"]
    spread = ["p01", "p05", "p25", "p75", "p95", "p99", "iqr"]
    for var, title in (
        ("age", "Возраст"),
        ("cart_value", "Стоимость корзины"),
        ("revenue", "Выручка, все пользователи"),
        ("revenue_payers", "Выручка, только покупатели"),
    ):
        add(f"### {title}\n")
        add(_group_table(r.descriptive, var, centre))
        add(_group_table(r.descriptive, var, spread))
    add("### Форма распределения выручки\n")
    rows = []
    for label, key in (("все пользователи", "all"), ("покупатели", "payers")):
        for g in ("control", "treatment"):
            dd = r.revenue_distribution[g][key]
            rows.append(
                [
                    label,
                    g,
                    _f(dd["skewness"]),
                    _f(dd["excess_kurtosis"]),
                    pct(dd["zero_share"], 1),
                    _f(dd["mean_to_median"]),
                    _f(dd.get("log_positive_skewness")),
                ]
            )
    add(
        md_table(
            [
                "выборка",
                "группа",
                "асимметрия",
                "эксцесс",
                "доля нулей",
                "среднее/медиана",
                "асимметрия ln(выручки>0)",
            ],
            rows,
        )
    )
    add(
        "Выручка по всем пользователям состоит из большой массы нулей и правого хвоста. "
        "У покупателей среднее больше медианы, а после логарифмирования асимметрия резко "
        "уменьшается: распределение близко к логнормальному. Поэтому для денег одного "
        "t-теста недостаточно, и результат подкрепляется bootstrap.\n"
    )

    # ---- 5. Конверсия
    add("## 5. Конверсия: z-тест двух долей\n")
    add("```text")
    add(f"p_C = {c.x_control:,} / {c.n_control:,} = {c.p_control:.5f}")
    add(f"p_T = {c.x_treatment:,} / {c.n_treatment:,} = {c.p_treatment:.5f}")
    add(f"Δ   = p_T - p_C = {c.diff:+.5f}")
    add(f"SE0 (объединённая доля, для теста) = {c.se_pooled:.6f}")
    add(f"z   = Δ / SE0 = {c.z:.4f}      p-value (двусторонний) = {fmt_p(c.p_value)}")
    add(f"SE1 (раздельные доли, для интервала) = {c.se_unpooled:.6f}")
    add(f"{lvl} ДИ = Δ ± {z_crit:.3f} × SE1 = [{c.ci_low:+.5f}; {c.ci_high:+.5f}]")
    add("```\n")
    add(
        md_table(
            ["Показатель", "Значение"],
            [
                [
                    "Конверсия control / treatment",
                    f"{pct(c.p_control, 3)} / {pct(c.p_treatment, 3)}",
                ],
                [
                    "Абсолютная разница",
                    f"{pp(c.diff)} ({lvl} ДИ {pp(c.ci_low)} ... {pp(c.ci_high)})",
                ],
                [
                    "Относительный прирост (relative lift)",
                    f"{c.rel_lift * 100:+.2f}% ({lvl} ДИ {c.rel_lift_ci_low * 100:+.2f}% ... "
                    f"{c.rel_lift_ci_high * 100:+.2f}%)",
                ],
                ["Стандартная ошибка разницы", f"{c.se_unpooled:.6f}"],
                ["z-статистика", f"{c.z:.4f}"],
                ["p-value", fmt_p(c.p_value)],
            ],
        )
    )
    add(
        "**Предпосылки теста:** "
        + (
            "независимость (одна строка на пользователя, группа назначена один раз) соблюдена; "
            "в каждой группе не менее 10 успехов и 10 неудач; SRM пройдена."
            if not c.assumption_issues and s.passed
            else "; ".join(c.assumption_issues) or "нарушена проверка SRM"
        )
    )
    add("")
    add("**Смысл p-value.** " + p_value_statement(c.p_value, alpha) + "\n")
    add("**Вывод.** " + cv.summary + "\n")

    # ---- 6. Выручка
    add("## 6. Выручка\n")
    add(
        md_table(
            ["Показатель", "Control", "Treatment"],
            [
                ["Пользователей", f"{sm['control']['users']:,}", f"{sm['treatment']['users']:,}"],
                ["Покупателей", f"{sm['control']['payers']:,}", f"{sm['treatment']['payers']:,}"],
                [
                    "Общая выручка",
                    f"{sm['control']['total']:,.2f}",
                    f"{sm['treatment']['total']:,.2f}",
                ],
                [
                    "ARPU (на всех пользователей)",
                    _f(sm["control"]["arpu"], 4),
                    _f(sm["treatment"]["arpu"], 4),
                ],
                [
                    "ARPPU (на покупателя)",
                    _f(sm["control"]["arppu"], 4),
                    _f(sm["treatment"]["arppu"], 4),
                ],
            ],
        )
    )
    add(
        "Общая выручка зависит от размера групп и сама по себе их не сравнивает. "
        "Сравниваются ARPU и ARPPU, пересчитанные на пользователя.\n"
    )
    add("### 6.1 Первичный тест для ARPU: тест Уэлча\n")
    add("```text")
    add(f"Δ  = mean_T - mean_C = {w.diff:+.5f}")
    add(f"SE = sqrt(s_C²/n_C + s_T²/n_T) = {w.se:.6f}")
    add(f"t  = Δ / SE = {w.t:.4f}, df (Уэлч-Саттертуэйт) = {w.df:,.1f}")
    add(f"p-value = {fmt_p(w.p_value)};  {lvl} ДИ = [{w.ci_low:+.5f}; {w.ci_high:+.5f}]")
    add("```\n")
    add(
        "Предпосылки: "
        + (
            "; ".join(w.assumption_issues)
            if w.assumption_issues
            else "объём каждой группы превышает ориентир Кохрана для данной асимметрии, "
            "среднее выборки можно считать близким к нормальному."
        )
        + "\n"
    )
    add("### 6.2 Проверка устойчивости: bootstrap\n")
    add(
        f"{bt.n_boot:,} пересэмплирований (seed {bt.seed}). Разница средних {bt.diff:+.5f}, "
        f"{lvl} bootstrap-ДИ [{bt.ci_low:+.5f}; {bt.ci_high:+.5f}]. "
        f"Правило заранее: тест Уэлча и bootstrap должны дать одно решение и один знак. "
        f"Результат: **{'согласуются' if r.arpu_methods_agree else 'РАСХОДЯТСЯ'}**.\n"
    )
    add("### 6.3 Диагностика: Манн-Уитни\n")
    add(
        f"U = {mw.u_treatment:,.0f}, p-value = {fmt_p(mw.p_value)}, вероятность превосходства "
        f"P(Y_T > Y_C) + 0.5·P(Y_T = Y_C) = {mw.prob_superiority:.4f} (0.5 означает отсутствие "
        "сдвига). Этот тест отвечает на другой вопрос (о сдвиге распределения, не о среднем), "
        "поэтому в решении не участвует. При 90% нулей он в основном реагирует на разницу "
        "долей покупателей.\n"
    )
    add("### 6.4 ARPPU (описательно)\n")
    add(
        f"Разница среднего чека покупателей {ap.diff:+.4f}, {lvl} ДИ "
        f"[{ap.ci_low:+.4f}; {ap.ci_high:+.4f}], p = {fmt_p(ap.p_value)}. "
        "Покупатели отбираются после назначения группы, поэтому группы покупателей не "
        "обязаны быть сравнимыми. Это объяснение, откуда берётся разница в ARPU, но не "
        "самостоятельный тест.\n"
    )
    add("**Смысл p-value для ARPU.** " + p_value_statement(w.p_value, alpha) + "\n")
    add("**Вывод по ARPU.** " + av.summary + "\n")

    # ---- 7. Мощность
    pw = r.power
    add("## 7. Мощность, размер выборки, длительность\n")
    add(
        f"Допущения планирования: базовая конверсия {pct(pw['baseline_assumed'], 1)}, "
        f"α = {alpha}, мощность {meta['power']:.0%}, трафик в среднем "
        f"{pw['daily_users']:,.0f} пользователей в день.\n"
    )
    add(
        md_table(
            ["Цель", "Эффект", "Нужно на группу", "Нужно всего", "Дней", "Дней (до целых недель)"],
            [
                [
                    p["label"],
                    pp(p["delta_abs"]),
                    f"{p['n_control']:,}",
                    f"{p['n_total']:,}",
                    p["days_raw"],
                    p["days_full_weeks"],
                ]
                for p in pw["plan"]
            ],
        )
    )
    add(f"- В тесте {pw['n_control_actual']:,} пользователей в контрольной группе.")
    add(
        f"- Мощность при плановом эффекте {pp(meta['planned_mde_abs'])} на фактическом объёме: "
        f"{pw['power_at_planned_mde']:.1%}."
    )
    add(
        f"- **MDE на фактическом объёме** (мощность {meta['power']:.0%}): "
        f"{pp(pw['mde_actual_abs'])} ({pw['mde_actual_rel'] * 100:.1f}% относительно control)."
    )
    add(
        f"- MDE для ARPU: {pw['mde_arpu_abs']:.4f} ({pw['mde_arpu_rel'] * 100:.1f}% от ARPU control). "
        "Выручка шумнее конверсии, поэтому ARPU обнаруживает только более крупные эффекты.\n"
    )
    add(r.sufficiency["text"] + "\n")
    add(
        "_Мощность по наблюдаемому эффекту не считается: она однозначно выводится из p-value "
        "и ничего не добавляет._\n"
    )

    # ---- 8. Итоговая интерпретация
    add("## 8. Статистический вывод\n")
    enough = "достаточно" if r.sufficiency["data_sufficient"] else "недостаточно"
    add(
        md_table(
            ["Вопрос", "Ответ"],
            [
                [
                    "Есть ли статистически значимый эффект (конверсия)?",
                    f"{'Да' if cv.significant else 'Нет'} (p = {fmt_p(c.p_value)}, α = {alpha})",
                ],
                [
                    "Какова величина эффекта?",
                    f"{pp(c.diff)}, относительно {c.rel_lift * 100:+.2f}%",
                ],
                ["Положительный или отрицательный?", DIRECTION_RU[cv.direction]],
                [
                    "Включает ли доверительный интервал ноль?",
                    "Да" if cv.ci_includes_zero else "Нет",
                ],
                ["Достаточно ли данных?", enough],
                [
                    "Практически значим ли эффект?",
                    f"{VERDICT_RU[cv.practical]} (порог {pp(meta['practical_threshold_abs'])})",
                ],
                [
                    "Подтверждается ли выручка?",
                    f"{VERDICT_RU[av.practical]}"
                    + ("" if r.arpu_methods_agree else "; методы расходятся"),
                ],
            ],
        )
    )
    add(cv.summary + "\n")
    add(f"**Итог: `{r.recommendation}`.** {r.recommendation_text}\n")

    # ---- 9. Ограничения
    add("## 9. Ограничения анализа\n")
    for item in (
        "Данные синтетические: нет ботов, дублирующихся событий и потерь логов.",
        "Анализ отвечает на вопрос об **среднем** эффекте. Различия по сегментам "
        "(устройство, канал, новизна) проверяются отдельно и требуют поправки на "
        "множественные сравнения (поправка Холма реализована в `hypothesis.py`).",
        "Период теста 28 дней: долгосрочный эффект и эффект новизны не оцениваются.",
        "Практические пороги заданы в конфиге и должны определяться бизнесом.",
        "Баланс групп проверен только по наблюдаемым признакам (см. EDA).",
        "Доверительный интервал показывает неопределённость оценки при условии, что "
        "модель теста верна, а не вероятность того, что истинный эффект лежит внутри.",
    ):
        add(f"- {item}")
    add("")

    # ---- Приложение
    if Path(methods_path).exists():
        add("---\n")
        add("## Приложение: методы и формулы\n")
        body = Path(methods_path).read_text(encoding="utf-8")
        add(body.replace("# Статистические методы ExperimentLab", "", 1).strip() + "\n")
    return "\n".join(L)
