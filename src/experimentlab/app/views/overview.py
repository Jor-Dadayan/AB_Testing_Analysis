"""Overview: можно ли запускать и насколько результату можно доверять."""

from __future__ import annotations

import streamlit as st

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis.business_summary import DECISION_LABELS
from experimentlab.analysis.report import fmt_p, pct, pp
from experimentlab.app import charts, components


def _tone(verdict) -> str:
    if verdict.significant and verdict.direction == "positive":
        return "good"
    if verdict.significant and verdict.direction == "negative":
        return "bad"
    return "neutral"


def render(ctx) -> None:
    components.page_header(
        ctx,
        "Overview",
        "Итог эксперимента: можно ли запускать новую версию Checkout "
        "и насколько результату можно доверять.",
    )
    r = components.require(ctx)
    base, dec, meta, pw = r.base, r.decision, r.base.meta, r.base.power
    c, cv, av = base.conversion, base.conversion_verdict, base.arpu_verdict
    lvl = f"{1 - meta['alpha']:.0%}"

    if ctx.is_filtered:
        off = DECISION_LABELS[ctx.official.decision.code][0] if ctx.official else "—"
        components.alert(
            "info",
            f"Официальное решение по всей аудитории: {off}. "
            "Ниже показаны метрики выбранной подвыборки.",
        )
    else:
        components.decision_banner(dec.code, dec.text)

    ev = [g for g in r.guardrails if g.status != "NOT_EVALUATED"]
    n_pass = sum(g.status == "PASS" for g in ev)
    open_req = any(
        g.required and g.status in ("INCONCLUSIVE", "NOT_EVALUATED") for g in r.guardrails
    )
    g_tone = (
        "bad"
        if any(g.status == "FAIL" for g in ev)
        else "warn"
        if open_req or any(g.status == "INCONCLUSIVE" for g in ev)
        else "good"
    )
    n_crit = sum(f.severity == "critical" for f in r.quality.findings)
    n_warn = sum(f.severity == "warning" for f in r.quality.findings)

    row1 = st.columns(4, gap="medium")
    with row1[0]:
        components.kpi_card(
            "Конверсия Treatment",
            pct(c.p_treatment),
            f"{pp(c.diff)} к Control ({pct(c.p_control)})",
            _tone(cv),
            f"{lvl} ДИ [{pp(c.ci_low)}; {pp(c.ci_high)}] · p = {fmt_p(c.p_value)}",
        )
    with row1[1]:
        w = base.arpu_welch
        components.kpi_card(
            "ARPU Treatment",
            f"{w.mean_treatment:,.3f}",
            f"{w.diff:+.3f} к Control ({w.mean_control:,.3f})",
            _tone(av),
            f"{lvl} ДИ [{w.ci_low:+.3f}; {w.ci_high:+.3f}] · p = {fmt_p(w.p_value)}",
        )
    with row1[2]:
        components.kpi_card(
            "Разбиение групп (SRM)",
            "Пройдена" if r.srm.passed else "Нарушено",
            tone="good" if r.srm.passed else "bad",
            caption=f"χ² = {r.srm.chi2:.2f} · p = {fmt_p(r.srm.p_value)}",
        )
    with row1[3]:
        components.kpi_card(
            "Guardrail-метрики",
            f"{n_pass} из {len(ev)}",
            tone=g_tone,
            caption=f"подтверждены; не измерены: {len(r.guardrails) - len(ev)}",
        )
    row2 = st.columns(4, gap="medium")
    with row2[0]:
        components.kpi_card(
            "Пользователей",
            components.fmt_int(meta["n_users"]),
            caption=f"Control {meta['n_control']:,} · Treatment {meta['n_treatment']:,}",
        )
    with row2[1]:
        components.kpi_card(
            "MDE на фактическом объёме",
            pp(pw["mde_actual_abs"]).replace("+", ""),
            caption=f"практический порог {pp(meta['practical_threshold_abs']).replace('+', '')}",
            tone="good" if pw["mde_actual_abs"] <= meta["practical_threshold_abs"] else "warn",
        )
    with row2[2]:
        components.kpi_card(
            "Мощность при плановом эффекте",
            f"{pw['power_at_planned_mde']:.0%}",
            caption=f"плановый эффект {pp(meta['planned_mde_abs'])}",
            tone="good" if pw["power_at_planned_mde"] >= meta["power"] else "warn",
        )
    with row2[3]:
        components.kpi_card(
            "Проверки качества",
            f"{n_crit} крит. · {n_warn} предупр.",
            tone="bad" if n_crit else "warn" if n_warn else "good",
        )

    st.markdown("&nbsp;")
    left, right = st.columns(2, gap="large")
    with left:
        components.chart_header(
            "Конверсия по группам",
            "Различаются ли конверсии и насколько неточна оценка?",
            "точка = конверсия, усы = 95% интервал Уилсона; ось не от нуля",
        )
        components.show_chart(
            charts.group_ci_chart(dd.group_metric_ci(ctx.df, "conversion"), "share", "Конверсия"),
            key="ov_group",
        )
    with right:
        components.chart_header(
            "Эффект и практический порог",
            "Лежит ли весь интервал эффекта выше порога, при котором запуск окупается?",
            "зелёная зона = практически значимый эффект",
        )
        components.show_chart(
            charts.effect_threshold_chart(
                c.diff, c.ci_low, c.ci_high, meta["practical_threshold_abs"]
            ),
            key="ov_effect",
        )

    st.markdown("##### Критерии успеха")
    st.caption(
        "Успех требует выполнения всех заранее заданных критериев, а не одной значимой метрики."
    )
    components.criteria_list(dec.criteria)

    msgs: list[tuple[str, str]] = []
    if r.srm.warning:
        msgs.append(("error", r.srm.warning))
    for f in r.quality.findings:
        if f.severity in ("critical", "warning"):
            msgs.append(("error" if f.severity == "critical" else "warning", f.message))
    seen = {m for _, m in msgs}
    msgs += [("warning", w) for w in dec.warnings if w not in seen]
    with st.expander(f"Предупреждения ({len(msgs)})", expanded=bool(msgs)):
        if not msgs:
            components.alert("success", "Предупреждений нет.")
        for kind, text in msgs:
            components.alert(kind, text)
