"""Построители графиков Plotly. Принимают готовые данные из analysis, ничего не считают."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experimentlab.app.theme import (
    COLORS,
    DIM_LABELS,
    GROUP_COLORS,
    GROUP_LABELS,
    rgba,
    style_figure,
)


def _fmt(unit: str, v: float) -> str:
    return f"{v:.2%}" if unit == "share" else f"{v:,.2f}"


def _axis_fmt(unit: str) -> str:
    return ".1%" if unit == "share" else ",.2f"


# ---------------------------------------------------------------- метрики
def group_ci_chart(frame: pd.DataFrame, unit: str, label: str) -> go.Figure:
    """Значение метрики в группах с доверительным интервалом."""
    fig = go.Figure()
    for r in frame.itertuples():
        g = str(r.group)
        color, name = GROUP_COLORS.get(g, COLORS["neutral"]), GROUP_LABELS.get(g, g)
        fig.add_trace(
            go.Scatter(
                x=[name],
                y=[r.value],
                mode="markers+text",
                text=[_fmt(unit, r.value)],
                textposition="middle right",
                textfont=dict(size=13, color=color),
                marker=dict(size=15, color=color),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[r.ci_high - r.value],
                    arrayminus=[r.value - r.ci_low],
                    color=color,
                    thickness=2.5,
                    width=9,
                ),
                showlegend=False,
                hovertemplate=f"{name}<br>{label}: %{{y:{_axis_fmt(unit)}}}<br>"
                f"n = {int(r.n):,}<extra></extra>",
            )
        )
    fig.update_yaxes(title=label, tickformat=_axis_fmt(unit))
    fig.update_xaxes(range=[-0.6, 1.6])
    return style_figure(fig, height=340)


def effect_threshold_chart(diff: float, lo: float, hi: float, threshold: float) -> go.Figure:
    """Разница конверсий с интервалом относительно нуля и практического порога (п.п.)."""
    d, lo, hi, thr = diff * 100, lo * 100, hi * 100, threshold * 100
    span = max(hi, thr) - min(lo, 0.0)
    x_min, x_max = min(lo, 0.0) - 0.25 * span, max(hi, thr) + 0.25 * span
    fig = go.Figure()
    fig.add_vrect(x0=thr, x1=x_max, fillcolor=rgba(COLORS["good"], 0.10), line_width=0)
    fig.add_trace(
        go.Scatter(
            x=[d],
            y=["Разница конверсий"],
            mode="markers",
            showlegend=False,
            marker=dict(size=15, color=COLORS["treatment"]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[hi - d],
                arrayminus=[d - lo],
                color=COLORS["treatment"],
                thickness=3,
                width=10,
            ),
            hovertemplate=f"оценка {d:+.2f} п.п.<br>ДИ [{lo:+.2f}; {hi:+.2f}]<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
    fig.add_vline(
        x=thr,
        line_dash="dash",
        line_color=COLORS["good"],
        annotation_text=f"порог {thr:.2f} п.п.",
        annotation_position="top",
    )
    fig.update_xaxes(range=[x_min, x_max], title="Разница Treatment − Control, п.п.")
    fig.update_yaxes(showticklabels=False)
    return style_figure(fig, height=340)


def metric_time_chart(daily: pd.DataFrame, mode: str, unit: str, label: str) -> go.Figure:
    """Метрика по дням или накопленная, по группам."""
    col = "cumulative" if mode == "cum" else "daily"
    fig = go.Figure()
    for g in ("control", "treatment"):
        d = daily[daily["group_name"] == g]
        fig.add_trace(
            go.Scatter(
                x=d["signup_date"],
                y=d[col],
                name=GROUP_LABELS[g],
                mode="lines+markers" if mode != "cum" else "lines",
                line=dict(color=GROUP_COLORS[g], width=2.5),
                marker=dict(size=5),
                connectgaps=True,
                hovertemplate=f"{GROUP_LABELS[g]}<br>%{{x|%d.%m.%Y}}<br>{label}: %{{y:{_axis_fmt(unit)}}}<extra></extra>",
            )
        )
    fig.update_yaxes(title=label, tickformat=_axis_fmt(unit))
    return style_figure(fig, height=360)


def cumulative_diff_chart(cd: pd.DataFrame, threshold_pp: float) -> go.Figure:
    """Накопленная разница конверсий с интервалом: сходится ли оценка со временем."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cd["signup_date"],
            y=cd["ci_high"] * 100,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cd["signup_date"],
            y=cd["ci_low"] * 100,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(COLORS["treatment"], 0.18),
            name="95% ДИ",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cd["signup_date"],
            y=cd["diff"] * 100,
            mode="lines",
            name="Разница T − C",
            line=dict(color=COLORS["treatment"], width=2.5),
            hovertemplate="%{x|%d.%m.%Y}<br>разница %{y:+.2f} п.п.<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1)
    fig.add_hline(
        y=threshold_pp,
        line_dash="dash",
        line_color=COLORS["good"],
        annotation_text="практический порог",
        annotation_position="top left",
    )
    fig.update_yaxes(title="Накопленная разница, п.п.")
    return style_figure(fig, height=360)


def revenue_hist_chart(hist: pd.DataFrame) -> go.Figure:
    """Распределение чека покупателей на логарифмической шкале."""
    fig = go.Figure()
    for g in ("control", "treatment"):
        d = hist[hist["group"] == g]
        fig.add_trace(
            go.Scatter(
                x=d["mid"],
                y=d["share"],
                name=GROUP_LABELS[g],
                mode="lines",
                fill="tozeroy",
                line=dict(color=GROUP_COLORS[g], width=2),
                fillcolor=rgba(GROUP_COLORS[g], 0.15),
                hovertemplate=f"{GROUP_LABELS[g]}<br>чек ≈ %{{x:,.0f}}<br>доля покупателей %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_xaxes(type="log", title="Выручка с покупателя (лог. шкала)")
    fig.update_yaxes(title="Доля покупателей в интервале", tickformat=".0%")
    return style_figure(fig, height=360)


# ---------------------------------------------------------------- интервалы
def interval_forest(
    rows: list[dict],
    x_title: str,
    thr: float | None = None,
    zero_line: bool = True,
    height: int | None = None,
) -> go.Figure:
    """Оценки с интервалами в строках. Ключи строки: label, est, lo, hi, color, legend."""
    fig = go.Figure()
    seen: set[str] = set()
    for r in rows:
        leg = r.get("legend", "")
        fig.add_trace(
            go.Scatter(
                x=[r["est"]],
                y=[r["label"]],
                mode="markers",
                marker=dict(size=10, color=r["color"]),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r["hi"] - r["est"]],
                    arrayminus=[r["est"] - r["lo"]],
                    color=r["color"],
                    thickness=2,
                    width=5,
                ),
                name=leg,
                legendgroup=leg,
                showlegend=bool(leg) and leg not in seen,
                hovertemplate=f"{r['label']}<br>оценка {r['est']:+.3f}<br>"
                f"ДИ [{r['lo']:+.3f}; {r['hi']:+.3f}]<extra></extra>",
            )
        )
        seen.add(leg)
    if zero_line:
        fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
    if thr is not None:
        fig.add_vline(
            x=thr,
            line_dash="dash",
            line_color=COLORS["good"],
            annotation_text="практический порог",
            annotation_position="top",
        )
    fig.update_yaxes(
        autorange="reversed", categoryorder="array", categoryarray=[r["label"] for r in rows]
    )
    fig.update_xaxes(title=x_title)
    return style_figure(fig, height=height or max(260, 34 * len(rows) + 110))


def sensitivity_forest(
    estimates, metric: str, scale: float, x_title: str, thr: float | None = None
) -> go.Figure:
    """Оценки одного вопроса разными методами (только методы, входящие в вердикт)."""
    rows = [
        {
            "label": e.method,
            "est": e.estimate * scale,
            "lo": e.ci_low * scale,
            "hi": e.ci_high * scale,
            "color": COLORS["treatment"],
            "legend": "",
        }
        for e in estimates
        if e.metric == metric and e.in_family and e.ci_low is not None
    ]
    return interval_forest(rows, x_title, thr)


def segment_forest(table: pd.DataFrame) -> go.Figure:
    """Эффект по сегментам с интервалом, скорректированным по Bonferroni."""
    rows = []
    for r in table.itertuples():
        if r.sig_holm and r.diff > 0:
            color, leg = COLORS["good"], "Значимый рост (после поправки Холма)"
        elif r.sig_holm:
            color, leg = COLORS["bad"], "Значимое снижение (после поправки Холма)"
        else:
            color, leg = COLORS["neutral"], "Не подтверждено после поправки"
        rows.append(
            {
                "label": f"{DIM_LABELS.get(r.dimension, r.dimension)}: {r.level}",
                "est": r.diff * 100,
                "lo": r.ci_low_bonf * 100,
                "hi": r.ci_high_bonf * 100,
                "color": color,
                "legend": leg,
            }
        )
    return interval_forest(rows, "Разница конверсий Treatment − Control, п.п. (скорр. ДИ)")


def pvalue_chart(table: pd.DataFrame, alpha: float) -> go.Figure:
    """Сырые и скорректированные p-values сегментов."""
    t = table.sort_values("p_value").reset_index(drop=True)
    labels = [
        f"{DIM_LABELS.get(d, d)}: {lv}" for d, lv in zip(t["dimension"], t["level"], strict=True)
    ]
    fig = go.Figure()
    for col, name, symbol, color in (
        ("p_value", "сырой p", "circle", COLORS["ink"]),
        ("p_holm", "p с поправкой Холма", "diamond", COLORS["treatment"]),
        ("q_bh", "q Бенджамини-Хохберга", "triangle-up", COLORS["warn"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=np.clip(t[col].to_numpy(float), 1e-300, 1.0),
                y=labels,
                mode="markers",
                name=name,
                marker=dict(symbol=symbol, size=9, color=color),
                hovertemplate=f"{name}<br>%{{y}}<br>%{{x:.4f}}<extra></extra>",
            )
        )
    fig.add_vline(
        x=alpha,
        line_dash="dash",
        line_color=COLORS["bad"],
        annotation_text=f"α = {alpha}",
        annotation_position="top",
    )
    fig.update_xaxes(type="log", title="p-value (лог. шкала)")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    return style_figure(fig, height=max(300, 28 * len(labels) + 130))


# ---------------------------------------------------------------- качество
def srm_counts_chart(srm) -> go.Figure:
    """Ожидаемое и фактическое число пользователей по группам."""
    groups = ["control", "treatment"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[GROUP_LABELS[g] for g in groups],
            y=[srm.expected_control, srm.expected_treatment],
            name="Ожидалось",
            marker_color=rgba(COLORS["neutral"], 0.6),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[GROUP_LABELS[g] for g in groups],
            y=[srm.n_control, srm.n_treatment],
            name="Фактически",
            marker_color=[GROUP_COLORS[g] for g in groups],
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Пользователей", tickformat=",d")
    return style_figure(fig, height=340)


def srm_daily_chart(ts: pd.DataFrame, expected_share: float) -> go.Figure:
    """Доля treatment среди пришедших по дням."""
    colors = [COLORS["bad"] if f else COLORS["treatment"] for f in ts["flagged"]]
    fig = go.Figure(
        go.Scatter(
            x=ts["date"],
            y=ts["share_treatment"],
            mode="lines+markers",
            showlegend=False,
            line=dict(color=COLORS["treatment"], width=2),
            marker=dict(size=8, color=colors),
            hovertemplate="%{x|%d.%m.%Y}<br>доля treatment %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=expected_share,
        line_dash="dash",
        line_color=COLORS["muted"],
        annotation_text="план",
        annotation_position="top left",
    )
    fig.update_yaxes(title="Доля Treatment", tickformat=".1%")
    return style_figure(fig, height=340)


def balance_chart(balance: pd.DataFrame, warn: float, critical: float) -> go.Figure:
    """Стандартизованная разница (|SMD|) признаков до эксперимента."""
    b = balance.sort_values("abs_smd", ascending=True)
    colors = [
        COLORS["bad"] if v >= critical else COLORS["warn"] if v >= warn else COLORS["good"]
        for v in b["abs_smd"]
    ]
    fig = go.Figure(
        go.Bar(
            x=b["abs_smd"],
            y=b["variable"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>|SMD| = %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=warn, line_dash="dot", line_color=COLORS["warn"])
    fig.add_vline(x=critical, line_dash="dot", line_color=COLORS["bad"])
    fig.update_xaxes(title="|SMD| (пороги: 0.10 и 0.25)")
    return style_figure(fig, height=max(280, 30 * len(b) + 100))


def stability_chart(info: dict) -> go.Figure:
    """Эффект в первой и второй половине эксперимента."""
    fig = go.Figure(
        go.Bar(
            x=["Первая половина", "Вторая половина"],
            y=[info["diff_early"] * 100, info["diff_late"] * 100],
            marker_color=[COLORS["control"], COLORS["treatment"]],
            text=[f"{info['diff_early'] * 100:+.2f}", f"{info['diff_late'] * 100:+.2f}"],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1)
    fig.update_yaxes(title="Разница конверсий, п.п.")
    return style_figure(fig, height=320)


def guardrail_chart(norm: pd.DataFrame) -> go.Figure:
    """Эффект и нижняя граница интервала в единицах допуска; вред сверх допуска левее -1."""
    fig = go.Figure()
    lo = min(-1.6, float(norm["bound"].min()) - 0.3) if len(norm) else -1.6
    hi = max(1.0, float(norm["effect"].max()) + 0.3) if len(norm) else 1.0
    fig.add_vrect(x0=lo, x1=-1, fillcolor=rgba(COLORS["bad"], 0.08), line_width=0)
    for r in norm.itertuples():
        color = (
            COLORS["good"]
            if r.status == "PASS"
            else COLORS["bad"]
            if r.status == "FAIL"
            else COLORS["warn"]
        )
        fig.add_trace(
            go.Scatter(
                x=[r.effect],
                y=[r.label],
                mode="markers",
                showlegend=False,
                marker=dict(size=12, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[0],
                    arrayminus=[r.effect - r.bound],
                    color=color,
                    thickness=3,
                    width=8,
                ),
                hovertemplate=f"{r.label}<br>эффект {r.effect:+.2f} допусков<br>"
                f"граница ДИ {r.bound:+.2f}<extra></extra>",
            )
        )
    fig.add_vline(
        x=-1,
        line_dash="dash",
        line_color=COLORS["bad"],
        annotation_text="допустимый предел",
        annotation_position="top",
    )
    fig.update_xaxes(
        range=[lo, hi], title="Эффект в единицах допуска (меньше −1 = вред сверх допуска)"
    )
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, height=max(240, 70 * len(norm) + 110))


# ---------------------------------------------------------------- мощность
def power_curve_chart(curve: pd.DataFrame, n_actual: float, target: float) -> go.Figure:
    """Мощность как функция размера выборки для нескольких эффектов."""
    fig = go.Figure()
    palette = [
        COLORS["control"],
        COLORS["treatment"],
        COLORS["warn"],
        COLORS["good"],
        COLORS["bad"],
    ]
    for i, (delta, d) in enumerate(curve.groupby("delta")):
        fig.add_trace(
            go.Scatter(
                x=d["n_control"],
                y=d["power"],
                mode="lines",
                name=f"{delta * 100:+.2f} п.п.",
                line=dict(color=palette[i % len(palette)], width=2.5),
                hovertemplate=f"эффект {delta * 100:+.2f} п.п.<br>n = %{{x:,}}<br>мощность %{{y:.1%}}<extra></extra>",
            )
        )
    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color=COLORS["muted"],
        annotation_text=f"цель {target:.0%}",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=n_actual,
        line_dash="dot",
        line_color=COLORS["ink"],
        annotation_text="фактический объём",
        annotation_position="top",
    )
    fig.update_xaxes(title="Пользователей в контрольной группе", tickformat=",d")
    fig.update_yaxes(title="Мощность", tickformat=".0%", range=[0, 1.02])
    return style_figure(fig, height=380)


def mde_curve_chart(
    curve: pd.DataFrame, n_actual: float, practical: float, planned: float
) -> go.Figure:
    """Минимальный обнаружимый эффект как функция размера выборки."""
    fig = go.Figure(
        go.Scatter(
            x=curve["n_control"],
            y=curve["mde"] * 100,
            mode="lines",
            showlegend=False,
            line=dict(color=COLORS["treatment"], width=2.5),
            hovertemplate="n = %{x:,}<br>MDE %{y:.2f} п.п.<extra></extra>",
        )
    )
    fig.add_hline(
        y=practical * 100,
        line_dash="dash",
        line_color=COLORS["good"],
        annotation_text="практический порог",
        annotation_position="top right",
    )
    fig.add_hline(
        y=planned * 100,
        line_dash="dot",
        line_color=COLORS["muted"],
        annotation_text="плановый MDE",
        annotation_position="top right",
    )
    fig.add_vline(
        x=n_actual,
        line_dash="dot",
        line_color=COLORS["ink"],
        annotation_text="фактический объём",
        annotation_position="top",
    )
    fig.update_xaxes(title="Пользователей в контрольной группе", tickformat=",d")
    fig.update_yaxes(title="MDE, п.п.")
    return style_figure(fig, height=360)
