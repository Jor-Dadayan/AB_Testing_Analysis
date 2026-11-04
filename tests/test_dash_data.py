import numpy as np
import pytest

from experimentlab.analysis import dashboard_data as dd
from experimentlab.analysis import power
from experimentlab.analysis.conversion import two_proportion_ztest
from experimentlab.analysis.guardrails import evaluate_guardrails
from experimentlab.analysis.revenue import welch_ttest


def _groups(df):
    g = df["group_name"].astype(str)
    return df[g == "control"], df[g == "treatment"]


def test_conversion_comparison_matches_ztest(adv_df):
    c, t = _groups(adv_df)
    ref = two_proportion_ztest(int(c.converted.sum()), len(c), int(t.converted.sum()), len(t))
    cmp_ = dd.compare_metric(adv_df, "conversion")
    assert cmp_.diff == pytest.approx(ref.diff) and cmp_.p_value == pytest.approx(ref.p_value)
    assert cmp_.ci_low == pytest.approx(ref.ci_low)


def test_arpu_and_aov_use_welch_on_correct_population(adv_df):
    c, t = _groups(adv_df)
    arpu = dd.compare_metric(adv_df, "arpu")
    assert arpu.p_value == pytest.approx(welch_ttest(c.revenue, t.revenue).p_value)
    aov = dd.compare_metric(adv_df, "aov")
    assert aov.n_control == int(c.converted.sum()) and aov.n_treatment == int(t.converted.sum())


def test_unknown_metric_rejected(adv_df):
    with pytest.raises(ValueError):
        dd.compare_metric(adv_df, "nps")


def test_group_ci_contains_value(adv_df):
    for metric in ("conversion", "arpu", "aov"):
        f = dd.group_metric_ci(adv_df, metric)
        assert len(f) == 2 and ((f.ci_low < f.value) & (f.value < f.ci_high)).all()


def test_cumulative_series_end_at_overall_value(adv_df):
    daily = dd.daily_metrics(adv_df, "conversion")
    last = daily.sort_values("signup_date").groupby("group_name")["cumulative"].last()
    c, t = _groups(adv_df)
    assert last["control"] == pytest.approx(c.converted.mean())
    assert last["treatment"] == pytest.approx(t.converted.mean())


def test_cumulative_difference_last_row_equals_full_test(adv_df):
    cd = dd.cumulative_conversion_difference(adv_df)
    assert cd.iloc[-1]["diff"] == pytest.approx(dd.compare_metric(adv_df, "conversion").diff)
    assert (cd["ci_low"] <= cd["diff"]).all() and (cd["diff"] <= cd["ci_high"]).all()


def test_revenue_histogram_shares_sum_to_one(adv_df):
    h = dd.revenue_histogram(adv_df)
    assert h.groupby("group")["share"].sum().round(6).eq(1.0).all()


def test_power_helpers_match_power_module():
    sc = dd.power_scenario(0.10, 0.01, 0.05, 0.8, 3500)
    assert sc.n_control == power.sample_size_two_proportions(0.10, 0.11, 0.05, 0.8)
    assert sc.days_full_weeks % 7 == 0
    with pytest.raises(ValueError):
        dd.power_scenario(0.10, 0.95, 0.05, 0.8, 3500)
    curve = dd.power_curve(0.10, [0.01], 20_000)
    assert curve["power"].is_monotonic_increasing
    mde = dd.mde_curve(0.10, 20_000)
    assert mde["mde"].is_monotonic_decreasing


def test_guardrail_normalization_uses_margin_units(adv_df, adv_cfg):
    guards = evaluate_guardrails(adv_df, adv_cfg["advanced"]["guardrails"])
    norm = dd.guardrail_normalized(guards)
    assert set(norm["name"]) == {"arpu", "aov"}  # не измеренные метрики исключены
    assert (norm["bound"] <= norm["effect"] + 1e-9).all()
    passed = norm[norm["status"] == "PASS"]
    assert (passed["bound"] > -1).all()
    assert np.isfinite(norm[["effect", "bound"]].to_numpy()).all()
