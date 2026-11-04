import plotly.graph_objects as go
import pytest

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis.advanced_engine import run_advanced
from experimentlab.analysis.business_summary import DECISION_LABELS, build_business_summary
from experimentlab.app import charts, services
from experimentlab.db import health


@pytest.fixture(scope="module")
def result(adv_df, adv_cfg):
    return run_advanced(adv_df, adv_cfg, source="test")


def test_all_charts_build(adv_df, result):
    base, meta = result.base, result.base.meta
    thr = meta["practical_threshold_abs"]
    figs = [
        charts.group_ci_chart(dd.group_metric_ci(adv_df, "conversion"), "share", "Конверсия"),
        charts.group_ci_chart(dd.group_metric_ci(adv_df, "arpu"), "money", "ARPU"),
        charts.effect_threshold_chart(
            base.conversion.diff, base.conversion.ci_low, base.conversion.ci_high, thr
        ),
        charts.metric_time_chart(dd.daily_metrics(adv_df, "arpu"), "cum", "money", "ARPU"),
        charts.metric_time_chart(
            dd.daily_metrics(adv_df, "conversion"), "daily", "share", "Конверсия"
        ),
        charts.cumulative_diff_chart(dd.cumulative_conversion_difference(adv_df), thr * 100),
        charts.revenue_hist_chart(dd.revenue_histogram(adv_df)),
        charts.sensitivity_forest(
            result.sensitivity.estimates, "conversion", 100.0, "п.п.", thr * 100
        ),
        charts.sensitivity_forest(result.sensitivity.estimates, "arpu", 1.0, "ARPU"),
        charts.segment_forest(result.segments.table),
        charts.pvalue_chart(result.segments.table, result.segments.alpha),
        charts.srm_counts_chart(result.srm),
        charts.srm_daily_chart(dd.srm_time_series(result.srm_segments), 0.5),
        charts.balance_chart(result.quality.balance, 0.1, 0.25),
        charts.stability_chart(result.quality.stability),
        charts.guardrail_chart(dd.guardrail_normalized(result.guardrails)),
        charts.power_curve_chart(dd.power_curve(0.10, [0.005, 0.01], 20_000), 14_751, 0.8),
        charts.mde_curve_chart(dd.mde_curve(0.10, 20_000), 14_751, thr, 0.01),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) > 0 for f in figs)


def test_business_summary_content(result):
    bs = build_business_summary(result, monthly_users=100_000)
    assert bs.code == result.decision.code and DECISION_LABELS[bs.code][0] in bs.markdown
    assert bs.impact.extra_orders == pytest.approx(result.base.conversion.diff * 100_000)
    assert bs.impact.extra_orders_low <= bs.impact.extra_orders <= bs.impact.extra_orders_high
    low = bs.markdown.lower()
    assert "true_conversion" not in low and "truth.json" not in low
    assert bs.next_steps and bs.findings


def test_every_decision_code_has_label():
    assert {
        "SHIP",
        "SHIP_WITH_MONITORING",
        "NEED_MORE_DATA",
        "INVESTIGATE",
        "DO_NOT_SHIP",
        "INVALID_EXPERIMENT",
    } == set(DECISION_LABELS)


def test_filters_select_rows_and_drop_unused_categories(adv_df):
    key = (("device_type", ("mobile",)), ("is_new_user", ("1",)))
    out = services.apply_filters(adv_df, key)
    assert (out["device_type"] == "mobile").all() and (out["is_new_user"] == 1).all()
    assert 0 < len(out) < len(adv_df)
    assert services.apply_filters(adv_df, (("country", ()),)).empty
    assert "device_type: " in services.summarize_filters(key)


def test_size_check_blocks_tiny_subsets(adv_df):
    tiny = services.apply_filters(
        adv_df,
        (
            ("country", ("US",)),
            ("traffic_source", ("email",)),
            ("device_type", ("tablet",)),
            ("age_group", ("55+",)),
        ),
    )
    with pytest.raises(services.AnalysisUnavailableError):
        services._check_size(tiny, 1000)
    with pytest.raises(services.AnalysisUnavailableError):
        services._check_size(adv_df.iloc[0:0], 1000)


def test_db_health_reports_unavailable_without_leaking_secrets(monkeypatch):
    monkeypatch.setattr(
        health, "database_url", lambda: "postgresql+psycopg2://user:secret@127.0.0.1:1/none"
    )
    status = health.check_database(timeout=1)
    assert not status.ok and "недоступен" in status.message and "secret" not in status.message
