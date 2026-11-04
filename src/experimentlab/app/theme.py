"""Единая цветовая система, стиль графиков и CSS."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "control": "#64748B",
    "treatment": "#4F46E5",
    "good": "#16A34A",
    "warn": "#D97706",
    "bad": "#DC2626",
    "neutral": "#94A3B8",
    "ink": "#0F172A",
    "muted": "#64748B",
    "grid": "#E2E8F0",
}
GROUP_COLORS = {"control": COLORS["control"], "treatment": COLORS["treatment"]}
GROUP_LABELS = {"control": "Control (A)", "treatment": "Treatment (B)"}
DIM_LABELS = {
    "device_type": "Устройство",
    "country": "Страна",
    "traffic_source": "Канал",
    "age_group": "Возраст",
    "is_new_user": "Тип пользователя",
    "signup_date": "Дата регистрации",
}
# тон -> (текст, фон, рамка)
TONES = {
    "good": ("#166534", "#DCFCE7", "#86EFAC"),
    "warn": ("#92400E", "#FEF3C7", "#FCD34D"),
    "bad": ("#991B1B", "#FEE2E2", "#FCA5A5"),
    "neutral": ("#475569", "#F1F5F9", "#CBD5E1"),
    "info": ("#3730A3", "#E0E7FF", "#A5B4FC"),
}
STATUS_TONE = {
    "pass": "good",
    "fail": "bad",
    "warn": "warn",
    "na": "neutral",
    "PASS": "good",
    "FAIL": "bad",
    "INCONCLUSIVE": "warn",
    "NOT_EVALUATED": "neutral",
    "critical": "bad",
    "warning": "warn",
    "info": "info",
    "ok": "good",
}
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_figure(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    """Единый вид всех графиков."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color=COLORS["ink"]),
        margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


CSS = """
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px;}
h1, h2, h3, h4, h5 {letter-spacing: -0.01em;}
.el-kpi {background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px;
         min-height:122px; box-shadow:0 1px 2px rgba(15,23,42,.04); margin-bottom:.5rem;}
.el-kpi .label {font-size:.74rem; text-transform:uppercase; letter-spacing:.05em;
                color:#64748B; font-weight:600;}
.el-kpi .value {font-size:1.8rem; font-weight:700; color:#0F172A; line-height:1.2; margin-top:4px;}
.el-kpi .delta {font-size:.85rem; font-weight:600; margin-top:2px;}
.el-kpi .caption {font-size:.78rem; color:#64748B; margin-top:4px;}
.el-badge {display:inline-block; padding:2px 10px; margin:0 6px 4px 0; border-radius:999px;
           font-size:.75rem; font-weight:600; border:1px solid transparent;}
.el-banner {border-radius:12px; padding:16px 20px; border:1px solid; margin-bottom:1rem;}
.el-banner .tag {font-size:.72rem; font-weight:700; letter-spacing:.08em;}
.el-banner .title {font-size:1.5rem; font-weight:700; margin:2px 0 6px 0;}
.el-banner .code {font-size:.75rem; font-weight:600; opacity:.7; margin-left:8px;}
.el-banner .text {font-size:.95rem; color:#0F172A;}
.el-crit {display:flex; gap:12px; align-items:flex-start; padding:10px 4px;
          border-bottom:1px solid #E2E8F0;}
.el-crit .name {font-weight:600; font-size:.92rem;}
.el-crit .detail {font-size:.8rem; color:#64748B; margin-top:2px;}
.el-empty {border:1px dashed #CBD5E1; border-radius:12px; padding:28px; text-align:center;
           color:#64748B; background:#F8FAFC;}
@media (max-width: 640px) {
  .el-kpi .value {font-size:1.4rem;}
  .el-banner .title {font-size:1.2rem;}
}
"""


def inject_css() -> None:
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
