"""Business Report: итоговый вывод простым языком, оценка эффекта, выгрузка отчётов."""

from __future__ import annotations

import json

import streamlit as st

from experimentlab.analysis.advanced_report import render_advanced_report
from experimentlab.analysis.business_summary import build_business_summary
from experimentlab.app import components


def render(ctx) -> None:
    components.page_header(
        ctx,
        "Business Report",
        "Итоговый вывод для бизнеса: решение, ожидаемый эффект, риски и следующие шаги.",
    )
    components.require(ctx)
    off = ctx.official
    if ctx.is_filtered:
        components.alert(
            "info", "Бизнес-отчёт всегда строится по всей аудитории, а не по выбранной подвыборке."
        )
    daily = off.base.power["daily_users"]
    monthly = st.number_input(
        "Охват в месяц, пользователей",
        min_value=1000,
        value=max(1000, int(round(daily * 30, -2))),
        step=1000,
        key="biz_monthly",
    )
    bs = build_business_summary(off, float(monthly))
    components.decision_banner(bs.code, off.decision.text)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Что мы выяснили")
        for x in bs.findings:
            st.markdown(f"- {x}")
    with c2:
        st.markdown("##### Следующие шаги")
        for i, x in enumerate(bs.next_steps, 1):
            st.markdown(f"{i}. {x}")

    lvl = f"{1 - off.base.meta['alpha']:.0%}"
    im = bs.impact
    st.markdown("##### Ожидаемый эффект при запуске на всю аудиторию")
    k = st.columns(2, gap="medium")
    with k[0]:
        components.kpi_card(
            "Дополнительные заказы в месяц",
            f"{im.extra_orders:+,.0f}",
            caption=f"{lvl} ДИ [{im.extra_orders_low:+,.0f}; {im.extra_orders_high:+,.0f}]",
            tone="good" if im.extra_orders_low > 0 else "neutral",
        )
    with k[1]:
        components.kpi_card(
            "Изменение выручки в месяц",
            f"{im.extra_revenue:+,.0f}",
            caption=f"{lvl} ДИ [{im.extra_revenue_low:+,.0f}; {im.extra_revenue_high:+,.0f}]",
            tone="good" if im.extra_revenue_low > 0 else "neutral",
        )
    st.caption(im.assumption)

    st.markdown("##### Риски и ограничения")
    for x in bs.risks:
        st.markdown(f"- {x}")

    st.markdown("##### Скачать")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Бизнес-вывод (.md)", bs.markdown, f"business_{ctx.experiment_id}.md", "text/markdown"
    )
    d2.download_button(
        "Полный статистический отчёт (.md)",
        render_advanced_report(off),
        f"advanced_report_{ctx.experiment_id}.md",
        "text/markdown",
    )
    d3.download_button(
        "Результаты (.json)",
        json.dumps(off.to_dict(), ensure_ascii=False, indent=2),
        f"results_{ctx.experiment_id}.json",
        "application/json",
    )
    with st.expander("Показать текст бизнес-вывода"):
        st.markdown(bs.markdown)
