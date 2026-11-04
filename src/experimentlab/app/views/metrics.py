"""Metrics Analysis: поведение выбранной метрики в группах, во времени и по распределению."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis.report import fmt_p, pct, pp
from experimentlab.app import charts, components

OPTIONS = {
    "conversion": "Конверсия (первичная по плану)",
    "arpu": "ARPU: выручка на пользователя (вторичная)",
    "aov": "AOV: средний чек покупателей (защитная)",
}


def _val(unit: str, v: float) -> str:
    return pct(v, 3) if unit == "share" else f"{v:,.3f}"


def _diff(unit: str, v: float) -> str:
    return pp(v) if unit == "share" else f"{v:+,.3f}"


def render(ctx) -> None:
    components.page_header(
        ctx, "Metrics Analysis", "Сравнение Control и Treatment по метрикам эксперимента."
    )
    r = components.require(ctx)
    meta = r.base.meta
    alpha = meta["alpha"]

    metric = st.radio(
        "Метрика анализа",
        list(OPTIONS),
        format_func=OPTIONS.__getitem__,
        horizontal=True,
        key="metric_choice",
    )
    if metric != "conversion":
        components.alert(
            "info",
            "Это просмотр дополнительной метрики. Решение по эксперименту "
            "принимается по конверсии (первичная метрика, заданная заранее). "
            "Выбор другой метрики решение не меняет.",
        )
    try:
        cmp_ = dd.compare_metric(ctx.df, metric, alpha)
        group_ci = dd.group_metric_ci(ctx.df, metric, alpha)
        daily = dd.daily_metrics(ctx.df, metric)
    except ValueError as exc:
        components.empty_state(f"Для выбранной метрики недостаточно данных: {exc}")
        return

    lvl = f"{1 - alpha:.0%}"
    st.caption(f"{cmp_.test}. {cmp_.note}")
    cols = st.columns(5, gap="medium")
    tone = (
        "good"
        if cmp_.p_value < alpha and cmp_.diff > 0
        else "bad"
        if cmp_.p_value < alpha
        else "neutral"
    )
    with cols[0]:
        components.kpi_card(
            "Control", _val(cmp_.unit, cmp_.control), caption=f"n = {cmp_.n_control:,}"
        )
    with cols[1]:
        components.kpi_card(
            "Treatment", _val(cmp_.unit, cmp_.treatment), caption=f"n = {cmp_.n_treatment:,}"
        )
    with cols[2]:
        components.kpi_card(
            "Разница (T − C)",
            _diff(cmp_.unit, cmp_.diff),
            tone=tone,
            caption=f"{lvl} ДИ [{_diff(cmp_.unit, cmp_.ci_low)}; {_diff(cmp_.unit, cmp_.ci_high)}]",
        )
    with cols[3]:
        components.kpi_card("Относительно Control", f"{cmp_.rel_lift * 100:+.1f}%", tone=tone)
    with cols[4]:
        components.kpi_card(
            "p-value",
            fmt_p(cmp_.p_value),
            tone=tone,
            caption=f"α = {alpha}; это не вероятность того, что H0 верна",
        )

    tab1, tab2, tab3 = st.tabs(["Сравнение групп", "Динамика во времени", "Распределение выручки"])
    with tab1:
        components.show_table(
            pd.DataFrame(
                [
                    {
                        "Метрика": cmp_.label,
                        "Control": _val(cmp_.unit, cmp_.control),
                        "Treatment": _val(cmp_.unit, cmp_.treatment),
                        "Разница (T − C)": _diff(cmp_.unit, cmp_.diff),
                        "Относительно": f"{cmp_.rel_lift * 100:+.2f}%",
                        f"{lvl} ДИ": f"[{_diff(cmp_.unit, cmp_.ci_low)}; {_diff(cmp_.unit, cmp_.ci_high)}]",
                        "p-value": fmt_p(cmp_.p_value),
                        "Тест": cmp_.test,
                        "n Control": f"{cmp_.n_control:,}",
                        "n Treatment": f"{cmp_.n_treatment:,}",
                    }
                ]
            )
        )
        components.chart_header(
            f"{cmp_.label} по группам",
            "Насколько различаются группы с учётом неточности оценки?",
            "точка = значение, усы = 95% интервал; ось не от нуля",
        )
        components.show_chart(charts.group_ci_chart(group_ci, cmp_.unit, cmp_.label), key="m_group")
    with tab2:
        mode = st.radio(
            "Показывать", ["Накопленное значение", "По дням"], horizontal=True, key="ts_mode"
        )
        components.chart_header(
            f"{cmp_.label} во времени",
            "Стабилизируется ли метрика, или разница групп это шум первых дней?",
            "накопленная линия сглаживает шум; по дням видны колебания",
        )
        components.show_chart(
            charts.metric_time_chart(
                daily, "cum" if mode.startswith("Накопленное") else "daily", cmp_.unit, cmp_.label
            ),
            key="m_time",
        )
        if metric == "conversion":
            cd = dd.cumulative_conversion_difference(ctx.df, alpha)
            if cd.empty:
                components.empty_state("Недостаточно данных для накопленной разницы.")
            else:
                components.chart_header(
                    "Накопленная разница конверсий",
                    "Как менялась оценка эффекта и её неопределённость по мере набора данных?",
                    "интервал сужается с ростом выборки; ранние дни нельзя использовать для решения",
                )
                components.show_chart(
                    charts.cumulative_diff_chart(cd, meta["practical_threshold_abs"] * 100),
                    key="m_cumdiff",
                )
                st.caption(
                    "Ограничение: p-value в разные дни нельзя сравнивать как независимые проверки. "
                    "Остановка по первому «значимому» дню (peeking) повышает долю ложных находок."
                )
    with tab3:
        hist = dd.revenue_histogram(ctx.df)
        if hist.empty:
            components.empty_state("Нет покупателей для построения распределения.")
        else:
            components.chart_header(
                "Распределение чека покупателей",
                "Насколько скошена выручка и совпадают ли формы распределений в группах?",
                "логарифмическая шкала; правый хвост = редкие крупные заказы",
            )
            components.show_chart(charts.revenue_hist_chart(hist), key="m_hist")
            infl = r.sensitivity.influence
            components.show_table(
                pd.DataFrame(
                    [
                        {
                            "Группа": g,
                            "Доля выручки у топ-1% пользователей": pct(v["top1pct_users_share"], 1),
                            "Максимальный чек": f"{v['max_order']:,.2f}",
                            "Доля выручки от одного чека": pct(v["max_order_share"], 2),
                        }
                        for g, v in infl.items()
                    ]
                )
            )
            st.caption(
                "Крупные заказы не удаляются: это реальные покупки. Устойчивость к ним "
                "проверяется на странице Statistical Results (винзоризация, bootstrap)."
            )
