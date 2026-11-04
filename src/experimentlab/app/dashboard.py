"""Точка входа: streamlit run src/experimentlab/app/dashboard.py"""

from __future__ import annotations

import streamlit as st

from experimentlab.app import services, theme
from experimentlab.app.views import business, metrics, overview, quality, segmentation, statistical

st.set_page_config(
    page_title="ExperimentLab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded"
)
theme.inject_css()


def _reset_filters() -> None:
    for dim, (_, options) in services.FILTER_SPECS.items():
        st.session_state[f"flt_{dim}"] = list(options)


def render_filters(cfg: dict) -> tuple[str, services.FilterKey]:
    sb = st.sidebar
    sb.markdown("### ExperimentLab")
    sb.caption("A/B-тест новой страницы Checkout")
    labels = {e["id"]: e["label"] for e in services.list_experiments(cfg)}
    exp_id = sb.selectbox("Эксперимент", list(labels), format_func=labels.__getitem__, key="exp_id")
    sb.markdown("**Фильтры сегментов**")
    sb.caption("Исследовательский режим: решение о запуске принимается только по всей аудитории.")
    key: list[tuple[str, tuple[str, ...]]] = []
    for dim, (label, options) in services.FILTER_SPECS.items():
        names = services.VALUE_LABELS.get(dim, {})
        chosen = sb.multiselect(
            label,
            options,
            default=options,
            key=f"flt_{dim}",
            format_func=lambda v, n=names: n.get(v, v),
        )
        if set(chosen) != set(options):
            key.append((dim, tuple(v for v in options if v in chosen)))
    sb.button("Сбросить фильтры", on_click=_reset_filters)
    return exp_id, tuple(key)


def render_status(ctx: services.Context) -> None:
    sb = st.sidebar
    sb.divider()
    if ctx.db.ok:
        sb.success(f"PostgreSQL: {ctx.db.n_users:,} пользователей")
    else:
        sb.warning(ctx.db.message)
    sb.caption(f"Источник: {ctx.source}")
    sb.button("Обновить данные", on_click=st.cache_data.clear)


cfg = services.get_config()
exp_id, filter_key = render_filters(cfg)
ctx = services.build_context(exp_id, filter_key)
render_status(ctx)

pages = [
    st.Page(
        lambda: overview.render(ctx),
        title="Overview",
        url_path="overview",
        icon=":material/dashboard:",
        default=True,
    ),
    st.Page(
        lambda: metrics.render(ctx),
        title="Metrics Analysis",
        url_path="metrics",
        icon=":material/monitoring:",
    ),
    st.Page(
        lambda: statistical.render(ctx),
        title="Statistical Results",
        url_path="statistics",
        icon=":material/functions:",
    ),
    st.Page(
        lambda: segmentation.render(ctx),
        title="Segmentation",
        url_path="segmentation",
        icon=":material/groups:",
    ),
    st.Page(
        lambda: quality.render(ctx),
        title="Experiment Quality",
        url_path="quality",
        icon=":material/verified:",
    ),
    st.Page(
        lambda: business.render(ctx),
        title="Business Report",
        url_path="business",
        icon=":material/summarize:",
    ),
]
st.navigation(pages).run()
