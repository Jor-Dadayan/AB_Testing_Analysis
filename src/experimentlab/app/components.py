"""Переиспользуемые элементы интерфейса. Статистики здесь нет."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from experimentlab.analysis.business_summary import DECISION_LABELS
from experimentlab.app.theme import STATUS_TONE, TONES


def esc(x) -> str:
    return html.escape(str(x))


def fmt_int(v) -> str:
    return f"{int(v):,}"


def badge(text: str, tone: str = "neutral") -> str:
    fg, bg, border = TONES[tone]
    return (
        f'<span class="el-badge" style="color:{fg};background:{bg};border-color:{border}">'
        f"{esc(text)}</span>"
    )


def kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    tone: str = "neutral",
    caption: str | None = None,
) -> None:
    fg = TONES[tone][0]
    parts = [
        f'<div class="el-kpi"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div>'
    ]
    if delta:
        parts.append(f'<div class="delta" style="color:{fg}">{esc(delta)}</div>')
    if caption:
        parts.append(f'<div class="caption">{esc(caption)}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def decision_banner(code: str, text: str) -> None:
    label, tone = DECISION_LABELS[code]
    fg, bg, border = TONES[tone]
    st.markdown(
        f'<div class="el-banner" style="background:{bg};border-color:{border}">'
        f'<div class="tag" style="color:{fg}">ИТОГОВОЕ РЕШЕНИЕ</div>'
        f'<div class="title" style="color:{fg}">{esc(label)}<span class="code">{esc(code)}</span></div>'
        f'<div class="text">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def alert(kind: str, text: str) -> None:
    {"info": st.info, "warning": st.warning, "error": st.error, "success": st.success}[kind](text)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="el-empty">{esc(message)}</div>', unsafe_allow_html=True)


def chart_header(title: str, question: str, how_to_read: str | None = None) -> None:
    st.markdown(f"##### {title}")
    st.caption(f"Вопрос: {question}" + (f"  ·  Как читать: {how_to_read}" if how_to_read else ""))


def show_chart(fig, key: str | None = None) -> None:
    st.plotly_chart(
        fig,
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    """Таблица на всю ширину (совместимо с разными версиями Streamlit)."""
    try:
        st.dataframe(df, hide_index=True, height=height, width="stretch")
    except Exception:  # старые версии Streamlit не знают width="stretch"
        st.dataframe(df, hide_index=True, height=height)


def criteria_list(criteria) -> None:
    label = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "na": "N/A"}
    rows = [
        f'<div class="el-crit"><div>{badge(label[k.status], STATUS_TONE[k.status])}</div>'
        f'<div><div class="name">{esc(k.name)}</div><div class="detail">{esc(k.detail)}</div></div></div>'
        for k in criteria
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def findings_list(findings) -> None:
    label = {"critical": "КРИТИЧНО", "warning": "ВНИМАНИЕ", "info": "ИНФО", "ok": "OK"}
    rows = [
        f'<div class="el-crit"><div>{badge(label[f.severity], STATUS_TONE[f.severity])}</div>'
        f'<div><div class="name">{esc(f.message)}</div>'
        f'<div class="detail">код проверки: {esc(f.code)}</div></div></div>'
        for f in findings
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def page_header(ctx, title: str, subtitle: str) -> None:
    """Заголовок страницы, строка контекста и сквозные предупреждения."""
    st.markdown(f"## {title}")
    st.caption(subtitle)
    chips = [ctx.experiment_label]
    if ctx.result is not None:
        m = ctx.result.base.meta
        chips += [f"{m['period_start']} — {m['period_end']}", f"{m['n_users']:,} пользователей"]
    if ctx.is_filtered:
        chips.append("подвыборка: " + ctx.filter_summary)
    st.markdown(
        "".join(badge(c, "info" if i == 0 else "neutral") for i, c in enumerate(chips)),
        unsafe_allow_html=True,
    )
    if ctx.experiment_id == "primary" and not ctx.db.ok and ctx.result is not None:
        alert("warning", f"{ctx.db.message}. Показаны данные из CSV-файла.")
    if ctx.is_filtered:
        alert(
            "info",
            "Фильтры сегментов активны: это исследовательский просмотр подвыборки. "
            "Решение о запуске принимается только по всей аудитории.",
        )


def require(ctx):
    """Возвращает результат анализа или показывает понятную ошибку и останавливает страницу."""
    if ctx.result is None:
        st.error(ctx.error or "Нет данных для отображения.")
        if ctx.error_kind == "data":
            st.markdown("**Как исправить**")
            st.code(
                "python scripts/generate_data.py\n"
                "docker compose up -d --wait\n"
                "python scripts/load_to_db.py        # необязательно: без базы берётся CSV\n"
                "python scripts/build_experiments.py # демо-эксперименты",
                language="bash",
            )
        else:
            st.caption("Ослабьте фильтры сегментов или выберите другой эксперимент.")
        st.stop()
    return ctx.result
