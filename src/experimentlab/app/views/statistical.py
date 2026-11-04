"""Statistical Results: тесты, p-value, интервалы, guardrail, устойчивость, мощность."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis.interpretation import p_value_statement
from experimentlab.analysis.report import fmt_p, pct, pp
from experimentlab.app import charts, components
from experimentlab.app.theme import STATUS_TONE


def _primary_tab(r) -> None:
    base, meta = r.base, r.base.meta
    c, alpha = base.conversion, meta["alpha"]
    lvl = f"{1 - alpha:.0%}"
    st.markdown("##### План анализа (зафиксирован до просмотра результатов)")
    components.show_table(
        pd.DataFrame(
            [
                {"Метрика": h.metric, "Роль": h.role, "H0": h.h0, "H1": h.h1, "Тест": h.test}
                for h in base.hypotheses
            ]
        )
    )
    st.caption(
        f"α = {alpha} (ошибка I рода); целевая мощность {meta['power']:.0%} "
        f"(ошибка II рода β = {meta['type_2_error']:.2f}). Тест двусторонний: "
        "дизайн может и ухудшить метрику."
    )
    cols = st.columns(4, gap="medium")
    sig = c.p_value < alpha
    with cols[0]:
        components.kpi_card("z-статистика", f"{c.z:.3f}")
    with cols[1]:
        components.kpi_card(
            "p-value", fmt_p(c.p_value), tone="good" if sig and c.diff > 0 else "neutral"
        )
    with cols[2]:
        components.kpi_card(
            "Разница (T − C)", pp(c.diff), caption=f"{lvl} ДИ [{pp(c.ci_low)}; {pp(c.ci_high)}]"
        )
    with cols[3]:
        components.kpi_card(
            "Относительный прирост",
            f"{c.rel_lift * 100:+.2f}%",
            caption=f"{lvl} ДИ [{c.rel_lift_ci_low * 100:+.2f}%; {c.rel_lift_ci_high * 100:+.2f}%]",
        )
    st.markdown("**Смысл p-value.** " + p_value_statement(c.p_value, alpha))
    st.markdown("**Вывод.** " + base.conversion_verdict.summary)
    st.caption(
        "Предпосылки: "
        + (
            "; ".join(c.assumption_issues)
            if c.assumption_issues
            else "независимость наблюдений (одна строка на пользователя), не менее 10 успехов и "
            "10 неудач в каждой группе."
        )
    )
    with st.expander("Формулы и подстановка чисел"):
        st.latex(
            r"\hat p=\frac{x_C+x_T}{n_C+n_T},\quad SE_0=\sqrt{\hat p(1-\hat p)\left(\tfrac{1}{n_C}+\tfrac{1}{n_T}\right)},\quad z=\frac{\hat p_T-\hat p_C}{SE_0}"
        )
        st.latex(
            r"\hat\Delta\pm z_{1-\alpha/2}\,SE_1,\qquad SE_1=\sqrt{\tfrac{\hat p_C(1-\hat p_C)}{n_C}+\tfrac{\hat p_T(1-\hat p_T)}{n_T}}"
        )
        st.code(
            f"p_C = {c.x_control:,} / {c.n_control:,} = {c.p_control:.5f}\n"
            f"p_T = {c.x_treatment:,} / {c.n_treatment:,} = {c.p_treatment:.5f}\n"
            f"Δ = {c.diff:+.5f}\nSE0 = {c.se_pooled:.6f}\nz = {c.z:.4f}\n"
            f"p-value = {fmt_p(c.p_value)}\nSE1 = {c.se_unpooled:.6f}\n"
            f"{lvl} ДИ = [{c.ci_low:+.5f}; {c.ci_high:+.5f}]",
            language="text",
        )
    b = r.bootstrap
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Метод": "Аналитический (z-тест)",
                    f"{lvl} ДИ разницы": f"[{pp(c.ci_low)}; {pp(c.ci_high)}]",
                },
                {
                    "Метод": f"Bootstrap ({b['conversion'].n_boot:,} повторов, seed {b['conversion'].seed})",
                    f"{lvl} ДИ разницы": f"[{pp(b['conversion'].ci_low)}; {pp(b['conversion'].ci_high)}]",
                },
            ]
        )
    )


def _revenue_tab(r) -> None:
    base, alpha = r.base, r.base.meta["alpha"]
    lvl = f"{1 - alpha:.0%}"
    w, bt, mw, ap = base.arpu_welch, base.arpu_bootstrap, base.arpu_mwu, base.arppu_welch
    sm = base.revenue_summary
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Показатель": "Покупателей",
                    "Control": f"{sm['control']['payers']:,}",
                    "Treatment": f"{sm['treatment']['payers']:,}",
                },
                {
                    "Показатель": "Общая выручка",
                    "Control": f"{sm['control']['total']:,.2f}",
                    "Treatment": f"{sm['treatment']['total']:,.2f}",
                },
                {
                    "Показатель": "ARPU (на всех пользователей)",
                    "Control": f"{sm['control']['arpu']:.4f}",
                    "Treatment": f"{sm['treatment']['arpu']:.4f}",
                },
                {
                    "Показатель": "ARPPU (на покупателя)",
                    "Control": f"{sm['control']['arppu']:.4f}",
                    "Treatment": f"{sm['treatment']['arppu']:.4f}",
                },
            ]
        )
    )
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Тест": "Уэлч (первичный для ARPU)",
                    "Оценка": f"{w.diff:+.4f}",
                    f"{lvl} ДИ": f"[{w.ci_low:+.4f}; {w.ci_high:+.4f}]",
                    "p-value": fmt_p(w.p_value),
                },
                {
                    "Тест": "Bootstrap по среднему (проверка)",
                    "Оценка": f"{bt.diff:+.4f}",
                    f"{lvl} ДИ": f"[{bt.ci_low:+.4f}; {bt.ci_high:+.4f}]",
                    "p-value": "—",
                },
                {
                    "Тест": "Манн-Уитни (диагностика, другой вопрос)",
                    "Оценка": f"P(T>C)={mw.prob_superiority:.4f}",
                    f"{lvl} ДИ": "—",
                    "p-value": fmt_p(mw.p_value),
                },
                {
                    "Тест": "Уэлч для ARPPU (описательно)",
                    "Оценка": f"{ap.diff:+.4f}",
                    f"{lvl} ДИ": f"[{ap.ci_low:+.4f}; {ap.ci_high:+.4f}]",
                    "p-value": fmt_p(ap.p_value),
                },
            ]
        )
    )
    st.markdown(
        f"**Согласованность Уэлча и bootstrap:** {'согласуются' if base.arpu_methods_agree else 'РАСХОДЯТСЯ, вывод по выручке не делается'}."
    )
    st.markdown("**Смысл p-value.** " + p_value_statement(w.p_value, alpha))
    st.markdown("**Вывод по ARPU.** " + base.arpu_verdict.summary)
    st.caption(
        "Манн-Уитни проверяет сдвиг распределения, а не среднее, поэтому в решении не участвует. "
        "ARPPU считается по покупателям, отобранным после назначения группы, и не является независимым тестом."
    )


def _guardrail_tab(r) -> None:
    st.caption(
        "Проверяется не «есть ли изменение», а «исключено ли ухудшение сверх допуска» "
        "(односторонний тест non-inferiority)."
    )
    rows = []
    for g in r.guardrails:

        def f(v, g=g):
            if v != v:
                return "—"
            return pct(v, 2) if g.kind == "proportion" else f"{v:,.3f}"

        rows.append(
            {
                "Метрика": g.label,
                "Обязательна": "да" if g.required else "нет",
                "Control": f(g.mean_control),
                "Treatment": f(g.mean_treatment),
                "Граница ДИ": f(g.bound),
                "Допустимый предел": f(g.limit),
                "Статус": g.status,
            }
        )
    components.show_table(pd.DataFrame(rows))
    norm = dd.guardrail_normalized(r.guardrails)
    if norm.empty:
        components.empty_state("Нет измеренных guardrail-метрик.")
    else:
        components.chart_header(
            "Guardrail в единицах допуска",
            "Исключено ли ухудшение защитных метрик сверх допустимого предела?",
            "точка = эффект, ус = нижняя граница интервала; левее красной линии = вред сверх допуска",
        )
        components.show_chart(charts.guardrail_chart(norm), key="st_guard")
    for g in r.guardrails:
        st.markdown(
            components.badge(g.status, STATUS_TONE[g.status]) + f" **{g.label}**: {g.detail}",
            unsafe_allow_html=True,
        )
    st.caption(
        "Незначимое изменение не доказывает отсутствие вреда. Статус INCONCLUSIVE означает, "
        "что данные не исключают ухудшения."
    )


def _robust_tab(r) -> None:
    sv, meta = r.sensitivity, r.base.meta
    lvl = f"{1 - meta['alpha']:.0%}"
    left, right = st.columns(2, gap="large")
    with left:
        components.chart_header(
            "Конверсия разными методами",
            "Зависит ли вывод о конверсии от выбора метода?",
            f"точка = оценка, усы = {lvl} интервал; методы заданы заранее",
        )
        components.show_chart(
            charts.sensitivity_forest(
                sv.estimates,
                "conversion",
                100.0,
                "Разница конверсий, п.п.",
                meta["practical_threshold_abs"] * 100,
            ),
            key="st_rob_conv",
        )
        st.markdown(
            f"**{'Устойчиво' if sv.conversion_robust else 'Неустойчиво'}:** {sv.conversion_reason}."
        )
    with right:
        components.chart_header(
            "ARPU разными методами и влияние выбросов",
            "Держится ли вывод о выручке на нескольких крупных заказах?",
            "винзоризация ограничивает чеки перцентилем",
        )
        components.show_chart(
            charts.sensitivity_forest(sv.estimates, "arpu", 1.0, "Разница ARPU", None),
            key="st_rob_arpu",
        )
        st.markdown(f"**{'Устойчиво' if sv.arpu_robust else 'Неустойчиво'}:** {sv.arpu_reason}.")
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Метрика": e.metric,
                    "Метод": e.method,
                    "В вердикте": "да" if e.in_family else "нет",
                    "Оценка": f"{e.estimate:+.5f}",
                    f"{lvl} ДИ": "—"
                    if e.ci_low is None
                    else f"[{e.ci_low:+.5f}; {e.ci_high:+.5f}]",
                    "p-value": "—" if e.p_value is None else fmt_p(e.p_value),
                    "Значимо": "да" if e.significant else "нет",
                    "Примечание": e.note,
                }
                for e in sv.estimates
            ]
        )
    )
    st.caption(
        "Методы с пометкой «нет» отвечают на другой вопрос (сдвиг распределения, лог-шкала); "
        "их расхождение со средним не противоречие."
    )


def _power_tab(ctx, r) -> None:
    base, meta, pw = r.base, r.base.meta, r.base.power
    share = float(ctx.cfg["treatment_share"])
    ratio = share / (1 - share)
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Цель": p["label"],
                    "Эффект": pp(p["delta_abs"]),
                    "Нужно на группу": f"{p['n_control']:,}",
                    "Нужно всего": f"{p['n_total']:,}",
                    "Дней": p["days_raw"],
                    "Дней (целые недели)": p["days_full_weeks"],
                }
                for p in pw["plan"]
            ]
        )
    )
    cols = st.columns(4, gap="medium")
    with cols[0]:
        components.kpi_card("Control в тесте", f"{pw['n_control_actual']:,}")
    with cols[1]:
        components.kpi_card("Мощность при плановом эффекте", f"{pw['power_at_planned_mde']:.1%}")
    with cols[2]:
        components.kpi_card(
            "MDE (конверсия)",
            pp(pw["mde_actual_abs"]).replace("+", ""),
            caption=f"{pw['mde_actual_rel'] * 100:.1f}% от конверсии Control",
        )
    with cols[3]:
        components.kpi_card(
            "MDE (ARPU)", f"{pw['mde_arpu_rel'] * 100:.1f}%", caption="от ARPU Control"
        )
    st.markdown(base.sufficiency["text"])
    st.caption(
        "Мощность по наблюдаемому эффекту не считается: она однозначно выражается через p-value."
    )

    st.markdown("##### Калькулятор «что если»")
    c1, c2, c3, c4 = st.columns(4)
    baseline = (
        c1.number_input(
            "Базовая конверсия, %",
            0.5,
            60.0,
            round(pw["baseline_assumed"] * 100, 2),
            step=0.5,
            key="pw_base",
        )
        / 100
    )
    effect = (
        c2.number_input(
            "Ожидаемый рост, п.п.",
            0.1,
            10.0,
            round(meta["planned_mde_abs"] * 100, 2),
            step=0.1,
            key="pw_eff",
        )
        / 100
    )
    alpha_in = c3.selectbox("Уровень значимости α", [0.01, 0.05, 0.10], index=1, key="pw_alpha")
    power_in = c4.selectbox("Мощность", [0.7, 0.8, 0.9, 0.95], index=1, key="pw_power")
    daily = st.number_input(
        "Пользователей в день",
        100,
        10_000_000,
        max(100, int(round(pw["daily_users"]))),
        step=100,
        key="pw_daily",
    )
    try:
        sc = dd.power_scenario(baseline, effect, alpha_in, power_in, float(daily), ratio)
    except ValueError as exc:
        components.empty_state(f"Сценарий невозможен: {exc}")
        return
    k = st.columns(4, gap="medium")
    with k[0]:
        components.kpi_card("На группу (control)", f"{sc.n_control:,}")
    with k[1]:
        components.kpi_card("Всего", f"{sc.n_total:,}")
    with k[2]:
        components.kpi_card("Дней", str(sc.days_raw))
    with k[3]:
        components.kpi_card("Дней (целые недели)", str(sc.days_full_weeks))
    n_max = max(2 * sc.n_control, pw["n_control_actual"] * 1.2)
    deltas = sorted({round(x, 4) for x in (0.005, 0.01, 0.015, 0.02, effect)})
    left, right = st.columns(2, gap="large")
    with left:
        components.chart_header(
            "Мощность и размер выборки",
            "Сколько пользователей нужно, чтобы заметить эффект заданного размера?",
            "кривая выше пунктира цели = эффект будет обнаружен",
        )
        components.show_chart(
            charts.power_curve_chart(
                dd.power_curve(baseline, deltas, n_max, alpha_in, ratio),
                pw["n_control_actual"],
                power_in,
            ),
            key="st_power",
        )
    with right:
        components.chart_header(
            "Минимальный обнаружимый эффект",
            "Какой минимальный эффект способен заметить тест нашего объёма?",
            "линия ниже практического порога = тест достаточно чувствителен",
        )
        components.show_chart(
            charts.mde_curve_chart(
                dd.mde_curve(baseline, n_max, alpha_in, power_in, ratio),
                pw["n_control_actual"],
                meta["practical_threshold_abs"],
                meta["planned_mde_abs"],
            ),
            key="st_mde",
        )


def render(ctx) -> None:
    components.page_header(
        ctx,
        "Statistical Results",
        "Проверка гипотез, интервалы, защитные метрики, устойчивость и мощность.",
    )
    r = components.require(ctx)
    t1, t2, t3, t4, t5 = st.tabs(
        ["Конверсия (первичная)", "Выручка", "Guardrails", "Устойчивость", "Power Analysis"]
    )
    with t1:
        _primary_tab(r)
    with t2:
        _revenue_tab(r)
    with t3:
        _guardrail_tab(r)
    with t4:
        _robust_tab(r)
    with t5:
        _power_tab(ctx, r)
