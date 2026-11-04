"""Markdown-отчёт по AdvancedResult. Только форматирование, расчётов здесь нет."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from experimentlab.analysis.advanced_engine import AdvancedResult
from experimentlab.analysis.multiple_testing import family_wise_error_probability
from experimentlab.analysis.report import fmt_p, md_table, pct, pp
from experimentlab.config import PROJECT_ROOT

METHODS_PATH = PROJECT_ROOT / "docs" / "advanced_methods.md"
ICON = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "na": "N/A"}
SEV = {"critical": "КРИТИЧНО", "warning": "ПРЕДУПРЕЖДЕНИЕ", "info": "инфо", "ok": "OK"}


def _n(v, digits: int = 4) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{digits}f}"


def render_advanced_report(r: AdvancedResult, methods_path: Path = METHODS_PATH) -> str:
    L: list[str] = []
    add = L.append
    base, dec = r.base, r.decision
    meta, c = base.meta, base.conversion
    alpha = meta["alpha"]
    lvl = f"{1 - alpha:.0%}"

    add("# ExperimentLab: расширенный анализ эксперимента\n")
    add(
        f"_Создан автоматически {datetime.now():%Y-%m-%d %H:%M}. Источник данных: {meta['source']}._\n"
    )
    add(
        f"Пользователей: {meta['n_users']:,} (control {meta['n_control']:,}, "
        f"treatment {meta['n_treatment']:,}). Период: {meta['period_start']} - {meta['period_end']}.\n"
    )

    # 1
    add("## 1. Итоговое решение\n")
    add(f"**`{dec.code}`.** {dec.text}\n")
    add(
        "Эксперимент считается успешным только при выполнении **всех** заранее заданных критериев. "
        "Значимость одной метрики этого не определяет.\n"
    )
    add(
        md_table(
            ["Критерий", "Статус", "Детали"],
            [[k.name, ICON[k.status], k.detail] for k in dec.criteria],
        )
    )
    if dec.warnings:
        add("**Предупреждения:**\n")
        for w in dec.warnings:
            add(f"- {w}")
        add("")

    # 2
    s = r.srm
    add("## 2. Проверка SRM\n")
    add(
        md_table(
            ["Группа", "Ожидаемое число", "Фактическое число"],
            [
                ["control", f"{s.expected_control:,.0f}", f"{s.n_control:,}"],
                ["treatment", f"{s.expected_treatment:,.0f}", f"{s.n_treatment:,}"],
            ],
        )
    )
    add(
        f"Chi-square goodness-of-fit: χ² = {s.chi2:.3f}, df = 1, p = {fmt_p(s.p_value)} "
        f"(порог {s.alpha}). Результат: **{'пройдена' if s.passed else 'НЕ ПРОЙДЕНА'}**.\n"
    )
    if s.warning:
        add("> " + s.warning + "\n")
    seg = r.srm_segments
    if len(seg):
        flagged = seg[seg["flagged"]]
        add(
            f"По сегментам и дням: {len(seg)} проверок, порог с поправкой Bonferroni "
            f"{seg['alpha_adj'].iloc[0]:.2e}. Отмечено: **{len(flagged)}**.\n"
        )
        if len(flagged):
            add(
                md_table(
                    ["Признак", "Уровень", "control", "treatment", "p"],
                    [
                        [
                            x.dimension,
                            x.level,
                            f"{x.n_control:,}",
                            f"{x.n_treatment:,}",
                            fmt_p(x.p_value),
                        ]
                        for x in flagged.itertuples()
                    ],
                )
            )

    # 3
    b = r.bootstrap
    add("## 3. Bootstrap\n")
    add(
        f"Конверсия: {b['conversion'].n_boot:,} повторений; ARPU: {b['arpu'].n_boot:,}; "
        f"seed {b['arpu'].seed}. Результаты воспроизводимы.\n"
    )
    add(
        md_table(
            ["Метрика", "Оценка", f"Bootstrap {lvl} ДИ", f"Аналитический {lvl} ДИ"],
            [
                [
                    "Разница конверсий",
                    pp(b["conversion"].diff),
                    f"[{pp(b['conversion'].ci_low)}; {pp(b['conversion'].ci_high)}]",
                    f"[{pp(c.ci_low)}; {pp(c.ci_high)}]",
                ],
                [
                    "Relative lift конверсии",
                    f"{b['conversion_relative_lift'].diff * 100:+.2f}%",
                    f"[{b['conversion_relative_lift'].ci_low * 100:+.2f}%; "
                    f"{b['conversion_relative_lift'].ci_high * 100:+.2f}%]",
                    f"[{c.rel_lift_ci_low * 100:+.2f}%; {c.rel_lift_ci_high * 100:+.2f}%]",
                ],
                [
                    "Разница ARPU",
                    _n(b["arpu"].diff),
                    f"[{_n(b['arpu'].ci_low)}; {_n(b['arpu'].ci_high)}]",
                    f"[{_n(base.arpu_welch.ci_low)}; {_n(base.arpu_welch.ci_high)}]",
                ],
            ],
        )
    )

    # 4
    sa, t = r.segments, r.segments.table
    add("## 4. Сегментный анализ (исследовательский)\n")
    add(
        "Сегменты заданы в конфиге **заранее**. Результаты этого раздела формируют гипотезы для "
        "следующих тестов и **не могут обосновать запуск**.\n"
    )
    if len(sa.heterogeneity):
        add(
            "**Омнибус-тест: различается ли эффект между уровнями признака** "
            "(LR-тест взаимодействия, поправка Холма на число признаков):\n"
        )
        add(
            md_table(
                ["Признак", "Уровней", "LR", "df", "p", "p (Холм)", "Вывод"],
                [
                    [
                        h.dimension,
                        h.n_levels,
                        f"{h.lr_stat:.2f}",
                        h.df,
                        fmt_p(h.p_value),
                        fmt_p(h.p_holm),
                        "различия есть" if h.significant_holm else "различия не доказаны",
                    ]
                    for h in sa.heterogeneity.itertuples()
                ],
            )
        )
    if len(t):
        add(
            f"Тестов в семействе: **{sa.n_tests}**. Ожидаемое число ложных находок без поправки при "
            f"полном отсутствии эффекта: {sa.expected_false_positives:.1f}. "
            f"«Скорр. ДИ» построен с уровнем α/m (Bonferroni).\n"
        )
        rows = []
        for x in t.itertuples():
            flags = (
                ", ".join(
                    k
                    for k, col in (
                        ("сырой", x.sig_raw),
                        ("Bonf", x.sig_bonferroni),
                        ("Holm", x.sig_holm),
                        ("BH", x.sig_bh),
                    )
                    if col
                )
                or "-"
            )
            rows.append(
                [
                    x.dimension,
                    x.level,
                    f"{x.n_control:,}/{x.n_treatment:,}",
                    pct(x.p_control),
                    pct(x.p_treatment),
                    pp(x.diff),
                    f"[{pp(x.ci_low_bonf)}; {pp(x.ci_high_bonf)}]",
                    fmt_p(x.p_value),
                    fmt_p(x.p_holm),
                    fmt_p(x.q_bh),
                    flags,
                ]
            )
        add(
            md_table(
                [
                    "Признак",
                    "Уровень",
                    "n (C/T)",
                    "Конв. C",
                    "Конв. T",
                    "Разница",
                    "Скорр. ДИ",
                    "p сырой",
                    "p Холм",
                    "q BH",
                    "Значим по",
                ],
                rows,
            )
        )
    if sa.skipped:
        add(
            "Не тестировались (слишком мало данных): "
            + "; ".join(f"{x['dimension']}: {x['level']} ({x['reason']})" for x in sa.skipped)
            + "\n"
        )

    # 5
    add("## 5. Множественные сравнения\n")
    if len(t):
        m = sa.n_tests
        add(
            md_table(
                ["Подход", "Значимых сегментов", "Что контролируется"],
                [
                    [
                        "Без поправки (p < α)",
                        int(t["sig_raw"].sum()),
                        "ничего: риск ложных находок растёт с числом тестов",
                    ],
                    [
                        "Bonferroni",
                        int(t["sig_bonferroni"].sum()),
                        "вероятность хотя бы одной ложной находки (FWER)",
                    ],
                    ["Holm", int(t["sig_holm"].sum()), "FWER, не слабее Bonferroni"],
                    [
                        f"Benjamini-Hochberg (q = {sa.fdr_q})",
                        int(t["sig_bh"].sum()),
                        "ожидаемая доля ложных находок среди находок (FDR)",
                    ],
                ],
            )
        )
        add(
            f"При {m} независимых тестах и полном отсутствии эффекта вероятность хотя бы одной ложной "
            f"«находки» без поправки составляет {family_wise_error_probability(m, alpha):.0%}. "
            "Поэтому сырые p-values сегментов нельзя использовать для решения.\n"
        )
    else:
        add("Сегментных тестов нет.\n")

    # 6
    add("## 6. Guardrail-метрики\n")
    add(
        "Проверяется не «есть ли изменение», а «исключено ли ухудшение сверх допуска» "
        "(односторонний тест non-inferiority).\n"
    )
    add(
        md_table(
            [
                "Метрика",
                "Обязательна",
                "Control",
                "Treatment",
                "Разница",
                "Граница ДИ",
                "Предел",
                "Статус",
            ],
            [
                [
                    g.label,
                    "да" if g.required else "нет",
                    _n(g.mean_control),
                    _n(g.mean_treatment),
                    _n(g.diff),
                    _n(g.bound),
                    _n(g.limit),
                    g.status,
                ]
                for g in r.guardrails
            ],
        )
    )
    for g in r.guardrails:
        add(f"- **{g.label}**: {g.detail}.")
    add(
        "\nЧтобы включить возвраты и отказы от оформления, добавьте в данные столбцы `refunded` и "
        "`checkout_abandoned` и задайте `required: true` в конфиге.\n"
    )

    # 7
    sv = r.sensitivity
    add("## 7. Анализ чувствительности\n")
    add(
        md_table(
            ["Метрика", "Метод", "В вердикте", "Оценка", f"{lvl} ДИ", "p", "Значимо", "Примечание"],
            [
                [
                    e.metric,
                    e.method,
                    "да" if e.in_family else "нет",
                    _n(e.estimate, 5),
                    "-" if e.ci_low is None else f"[{e.ci_low:+.5f}; {e.ci_high:+.5f}]",
                    "-" if e.p_value is None else fmt_p(e.p_value),
                    "да" if e.significant else "нет",
                    e.note,
                ]
                for e in sv.estimates
            ],
        )
    )
    add(
        f"- **Конверсия:** {'УСТОЙЧИВО' if sv.conversion_robust else 'НЕУСТОЙЧИВО'}: {sv.conversion_reason}."
    )
    add(f"- **ARPU:** {'УСТОЙЧИВО' if sv.arpu_robust else 'НЕУСТОЙЧИВО'}: {sv.arpu_reason}.")
    add(
        "- Методы с пометкой «нет» отвечают на другой вопрос (сдвиг распределения, лог-шкала) и в вердикт "
        "не входят; их расхождение со средним не противоречие."
    )
    add(
        "- Регрессия использует только признаки, известные до эксперимента, поэтому оценка остаётся "
        "несмещённой, а интервал уже.\n"
    )
    add("**Влияние крупных заказов:**\n")
    add(
        md_table(
            [
                "Группа",
                "Доля выручки у топ-1% пользователей",
                "Максимальный чек",
                "Доля выручки от одного чека",
            ],
            [
                [
                    g,
                    pct(v["top1pct_users_share"], 1),
                    f"{v['max_order']:,.2f}",
                    pct(v["max_order_share"], 2),
                ]
                for g, v in sv.influence.items()
            ],
        )
    )
    margin = c.ci_low - meta["practical_threshold_abs"]
    add(
        f"**Запас до порога практической значимости:** нижняя граница ДИ {pp(c.ci_low)}, порог "
        f"{pp(meta['practical_threshold_abs'])}, запас {pp(margin)}. Если порог бизнеса выше нижней "
        "границы, классификация «практически значим» перестанет выполняться.\n"
    )

    # 8
    q = r.quality
    add("## 8. Качество эксперимента\n")
    add(
        md_table(
            ["Уровень", "Код", "Сообщение"],
            [[SEV[f.severity], f.code, f.message] for f in q.findings],
        )
    )
    add("Таблица баланса (первые 6 строк по |SMD|):\n")
    add(
        md_table(
            ["Признак", "|SMD|", "Сбалансирован"],
            [
                [x.variable, f"{x.abs_smd:.4f}", "да" if x.balanced else "нет"]
                for x in q.balance.head(6).itertuples()
            ],
        )
    )

    # 9
    n_raw = int(t["sig_raw"].sum()) if len(t) else 0
    n_holm = int(t["sig_holm"].sum()) if len(t) else 0
    add("## 9. Как читать результаты: правильно и неправильно\n")
    add(
        md_table(
            ["Ситуация", "Правильно", "Неправильно"],
            [
                [
                    "Конверсия значима, guardrail не проверен или не пройден",
                    "Успех первичной метрики не равен успеху эксперимента; смотреть статус всех критериев в разделе 1",
                    "«p < 0,05, внедряем»",
                ],
                [
                    f"В сегментах значимо без поправки: {n_raw}, по Холму: {n_holm}",
                    "Опираться на скорректированные результаты; остальное это гипотезы для нового теста",
                    "Выбрать сегмент с лучшим сырым p-value и запускать только для него",
                ],
                [
                    "Guardrail INCONCLUSIVE",
                    "Данные не исключают вред: запуск только с мониторингом",
                    "«Ухудшения не видно, значит, его нет»",
                ],
                [
                    "Bootstrap и тест дали разные выводы",
                    "Разобраться (хвост, выбросы), решение не принимать автоматически",
                    "Показать тот метод, который дал красивый результат",
                ],
                [
                    "Предупреждение MDE выше порога",
                    "Признать, что тест может не заметить минимально значимый эффект",
                    "Утверждать, что эффекта нет, если p > α",
                ],
            ],
        )
    )

    # 10
    add("## 10. Ограничения\n")
    for item in (
        "Данные синтетические: эффект однороден, нет эффекта новизны, ботов и потерь данных.",
        "Возвраты и отказы от оформления заказа в данных отсутствуют, поэтому эти guardrail не оценены.",
        "AOV считается только по покупателям, которые отбираются после назначения группы; это "
        "защитная метрика, а не независимое сравнение.",
        "Проверка стабильности эффекта маломощна: отсутствие различий не доказывает стабильность.",
        "Пороги (допуски guardrail, практический порог, пороги SMD) эвристические и должны задаваться бизнесом.",
        "Сегментный анализ исследовательский и никогда не обосновывает запуск.",
    ):
        add(f"- {item}")
    add("")

    if Path(methods_path).exists():
        add("---\n")
        add("## Приложение: методы и примеры интерпретации\n")
        add(
            Path(methods_path)
            .read_text(encoding="utf-8")
            .replace("# Расширенные методы ExperimentLab", "", 1)
            .strip()
            + "\n"
        )
    return "\n".join(L)
