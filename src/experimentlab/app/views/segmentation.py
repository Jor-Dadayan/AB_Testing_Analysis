"""Segmentation: где эффект может отличаться (исследовательский раздел)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentlab.analysis.report import fmt_p, pct, pp
from experimentlab.app import charts, components
from experimentlab.app.theme import DIM_LABELS


def render(ctx) -> None:
    components.page_header(
        ctx,
        "Segmentation",
        "Различается ли эффект по устройствам, странам, каналам, возрасту и типу пользователей.",
    )
    r = components.require(ctx)
    sa, t = r.segments, r.segments.table
    components.alert(
        "info",
        "Сегменты заданы в конфиге заранее. Раздел исследовательский: он формирует "
        "гипотезы для следующих тестов и не может обосновать запуск «только для сегмента X». "
        "Хороший результат в одном сегменте решение не улучшает.",
    )
    if sa.skipped:
        with st.expander(f"Не тестировались (мало данных): {len(sa.skipped)}"):
            components.show_table(
                pd.DataFrame(
                    [
                        {
                            "Признак": DIM_LABELS.get(x["dimension"], x["dimension"]),
                            "Сегмент": x["level"],
                            "Control": x["n_control"],
                            "Treatment": x["n_treatment"],
                            "Причина": x["reason"],
                        }
                        for x in sa.skipped
                    ]
                )
            )
    if t.empty:
        components.empty_state("Нет сегментов, в которых достаточно пользователей для теста.")
        return

    cols = st.columns(5, gap="medium")
    with cols[0]:
        components.kpi_card(
            "Тестов в семействе",
            str(sa.n_tests),
            caption=f"ожидаемо ложных находок: {sa.expected_false_positives:.1f}",
        )
    with cols[1]:
        components.kpi_card("Значимо без поправки", str(int(t["sig_raw"].sum())), tone="warn")
    with cols[2]:
        components.kpi_card("Bonferroni", str(int(t["sig_bonferroni"].sum())))
    with cols[3]:
        components.kpi_card(
            "Holm", str(int(t["sig_holm"].sum())), caption="используется для решений"
        )
    with cols[4]:
        components.kpi_card(
            "Benjamini-Hochberg", str(int(t["sig_bh"].sum())), caption=f"FDR q = {sa.fdr_q}"
        )

    if len(sa.heterogeneity):
        st.markdown("##### Омнибус-тест: различается ли эффект между уровнями признака")
        st.caption("LR-тест взаимодействия «группа × сегмент», поправка Холма на число признаков.")
        components.show_table(
            pd.DataFrame(
                [
                    {
                        "Признак": DIM_LABELS.get(h.dimension, h.dimension),
                        "Уровней": h.n_levels,
                        "LR": f"{h.lr_stat:.2f}",
                        "df": h.df,
                        "p-value": fmt_p(h.p_value),
                        "p (Холм)": fmt_p(h.p_holm),
                        "Вывод": "различия есть" if h.significant_holm else "различия не доказаны",
                    }
                    for h in sa.heterogeneity.itertuples()
                ]
            )
        )

    dims = list(dict.fromkeys(t["dimension"]))
    choice = st.selectbox(
        "Показать признак",
        ["all"] + dims,
        key="seg_dim",
        format_func=lambda d: "Все признаки" if d == "all" else DIM_LABELS.get(d, d),
    )
    view = t if choice == "all" else t[t["dimension"] == choice]
    left, right = st.columns(2, gap="large")
    with left:
        components.chart_header(
            "Эффект по сегментам",
            "Где разница конверсий отличается от нуля после поправки на множественность?",
            "усы = интервал с поправкой Bonferroni; цвет = вывод по Холму",
        )
        components.show_chart(charts.segment_forest(view), key="seg_forest")
    with right:
        components.chart_header(
            "Сырые и скорректированные p-values",
            "Сколько «находок» исчезает после поправок?",
            "точки справа от красной линии α не значимы; сдвиг вправо = эффект поправки",
        )
        components.show_chart(charts.pvalue_chart(view, sa.alpha), key="seg_p")

    st.markdown("##### Таблица сегментов")
    components.show_table(
        pd.DataFrame(
            [
                {
                    "Признак": DIM_LABELS.get(x.dimension, x.dimension),
                    "Сегмент": x.level,
                    "n (C / T)": f"{x.n_control:,} / {x.n_treatment:,}",
                    "Конв. Control": pct(x.p_control),
                    "Конв. Treatment": pct(x.p_treatment),
                    "Разница": pp(x.diff),
                    "Скорр. ДИ": f"[{pp(x.ci_low_bonf)}; {pp(x.ci_high_bonf)}]",
                    "p сырой": fmt_p(x.p_value),
                    "p Холм": fmt_p(x.p_holm),
                    "q BH": fmt_p(x.q_bh),
                    "Значим по": ", ".join(
                        k
                        for k, v in (
                            ("сырой", x.sig_raw),
                            ("Bonf", x.sig_bonferroni),
                            ("Holm", x.sig_holm),
                            ("BH", x.sig_bh),
                        )
                        if v
                    )
                    or "—",
                }
                for x in view.itertuples()
            ]
        ),
        height=420,
    )
    st.caption(
        "Ограничения: сегменты пересекаются (устройство и канал связаны), поэтому тесты не независимы; "
        "малые сегменты дают широкие интервалы; отсутствие значимости не означает отсутствие различий."
    )
