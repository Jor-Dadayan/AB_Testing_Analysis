"""Experiment Quality: SRM, баланс, достаточность выборки, длительность, стабильность."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis.report import fmt_p, pp
from experimentlab.app import charts, components
from experimentlab.app.theme import DIM_LABELS


def render(ctx) -> None:
    components.page_header(
        ctx,
        "Experiment Quality",
        "Можно ли доверять результатам: корректность разбиения, баланс групп, объём и длительность.",
    )
    r = components.require(ctx)
    s, q, base, adv = r.srm, r.quality, r.base, ctx.cfg["advanced"]
    pw, meta = base.power, base.meta
    n_crit = sum(f.severity == "critical" for f in q.findings)
    n_warn = sum(f.severity == "warning" for f in q.findings)

    cols = st.columns(4, gap="medium")
    with cols[0]:
        components.kpi_card(
            "SRM",
            "Пройдена" if s.passed else "Нарушено",
            tone="good" if s.passed else "bad",
            caption=f"p = {fmt_p(s.p_value)} (порог {s.alpha})",
        )
    with cols[1]:
        components.kpi_card("Критические проблемы", str(n_crit), tone="bad" if n_crit else "good")
    with cols[2]:
        components.kpi_card("Предупреждения", str(n_warn), tone="warn" if n_warn else "good")
    with cols[3]:
        ok = base.sufficiency["data_sufficient"]
        components.kpi_card(
            "Объём для планового MDE",
            "Достаточно" if ok else "Недостаточно",
            tone="good" if ok else "warn",
            caption=f"{pw['n_control_actual']:,} из {pw['plan'][0]['n_control']:,} на группу",
        )

    st.markdown("#### 1. Корректность разбиения (SRM)")
    if s.warning:
        components.alert("error", s.warning)
    left, right = st.columns(2, gap="large")
    with left:
        components.chart_header(
            "Ожидаемое и фактическое число пользователей",
            "Соответствует ли размер групп плану?",
            f"хи-квадрат: χ² = {s.chi2:.2f}, df = 1",
        )
        components.show_chart(charts.srm_counts_chart(s), key="q_srm")
    with right:
        ts = dd.srm_time_series(r.srm_segments)
        components.chart_header(
            "Доля Treatment по дням",
            "Набирались ли группы в плановой пропорции в каждый день?",
            "красные точки = нарушение с поправкой Bonferroni",
        )
        if ts.empty:
            components.empty_state("Недостаточно данных для проверки по дням.")
        else:
            components.show_chart(
                charts.srm_daily_chart(ts, s.expected_treatment_share), key="q_srm_day"
            )
    seg = r.srm_segments
    if len(seg):
        flagged = seg[seg["flagged"]]
        st.caption(
            f"Проверок по сегментам и дням: {len(seg)}; порог с поправкой {seg['alpha_adj'].iloc[0]:.2e}; "
            f"отмечено: {len(flagged)}."
        )
        if len(flagged):
            components.show_table(
                pd.DataFrame(
                    [
                        {
                            "Признак": DIM_LABELS.get(x.dimension, x.dimension),
                            "Уровень": x.level,
                            "Control": x.n_control,
                            "Treatment": x.n_treatment,
                            "p-value": fmt_p(x.p_value),
                        }
                        for x in flagged.itertuples()
                    ]
                )
            )

    st.markdown("#### 2. Баланс групп по признакам до эксперимента")
    components.chart_header(
        "Стандартизованная разница (SMD)",
        "Похожи ли группы по возрасту, странам, устройствам и каналам?",
        "ниже 0.10 = баланс; не зависит от размера выборки",
    )
    components.show_chart(
        charts.balance_chart(
            q.balance, float(adv["balance_smd_warn"]), float(adv["balance_smd_critical"])
        ),
        key="q_balance",
    )
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Признак": x.variable,
                    "Тип": x.kind,
                    "|SMD|": f"{x.abs_smd:.4f}",
                    "p-value (справочно)": fmt_p(x.p_value),
                    "Сбалансирован": "да" if x.balanced else "нет",
                }
                for x in q.balance.itertuples()
            ]
        ),
        height=320,
    )

    st.markdown("#### 3. Достаточность выборки и длительность")
    st.markdown(base.sufficiency["text"])
    k = st.columns(3, gap="medium")
    with k[0]:
        components.kpi_card("Период", f"{meta['period_start']} — {meta['period_end']}")
    with k[1]:
        components.kpi_card(
            "Расчётная длительность",
            f"{pw['plan'][0]['days_full_weeks']} дн.",
            caption="для планового MDE, до целых недель",
        )
    with k[2]:
        components.kpi_card(
            "MDE на фактическом объёме",
            pp(pw["mde_actual_abs"]).replace("+", ""),
            caption=f"порог {pp(meta['practical_threshold_abs']).replace('+', '')}",
        )

    st.markdown("#### 4. Стабильность эффекта во времени")
    if q.stability:
        components.chart_header(
            "Эффект в первой и второй половине",
            "Не ослабевает ли эффект со временем (эффект новизны)?",
            f"разница эффектов: z = {q.stability['z']:.2f}, p = {fmt_p(q.stability['p_value'])}",
        )
        components.show_chart(charts.stability_chart(q.stability), key="q_stab")
        st.caption("Проверка маломощна: отсутствие различий не доказывает стабильность.")
    else:
        components.empty_state("Период слишком короткий для проверки стабильности.")

    st.markdown("#### 5. Все проверки")
    components.findings_list(q.findings)
