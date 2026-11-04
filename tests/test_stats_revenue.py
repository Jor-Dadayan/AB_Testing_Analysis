import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from experimentlab.analysis.revenue import (
    arppu,
    arpu,
    bootstrap_mean_diff_ci,
    clt_issues,
    mann_whitney,
    methods_agree,
    revenue_summary,
    total_revenue,
    welch_ttest,
)


def test_arpu_arppu_total_with_zeros():
    control = [0] * 90 + [10] * 10
    treatment = [0] * 80 + [10] * 20
    assert arpu(control) == pytest.approx(1.0)
    assert arpu(treatment) == pytest.approx(2.0)
    assert arppu(control) == pytest.approx(10.0)
    assert total_revenue(treatment) == pytest.approx(200.0)
    s = revenue_summary(control, treatment)
    assert s["treatment"]["payers"] == 20
    assert s["control"]["payer_share"] == pytest.approx(0.1)


def test_non_finite_values_rejected():
    with pytest.raises(ValueError):
        arpu([1.0, float("nan")])


def test_welch_matches_scipy_and_statsmodels():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 400), rng.normal(10.5, 5, 300)
    r = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.t == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    lo, hi = CompareMeans(DescrStatsW(b), DescrStatsW(a)).tconfint_diff(usevar="unequal")
    assert r.ci_low == pytest.approx(lo)
    assert r.ci_high == pytest.approx(hi)
    assert 299 <= r.df <= 698


def test_welch_one_sided_half_of_two_sided():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 200), rng.normal(0.4, 1, 200)
    two = welch_ttest(a, b)
    one = welch_ttest(a, b, alternative="greater")
    assert one.p_value == pytest.approx(two.p_value / 2)


def test_welch_zero_variance_raises():
    with pytest.raises(ValueError):
        welch_ttest([1, 1, 1], [1, 1, 1])


def test_clt_issue_flagged_for_small_skewed_sample_only():
    rng = np.random.default_rng(4)
    assert clt_issues(rng.lognormal(0, 2, 50), "x")
    assert not clt_issues(rng.normal(0, 1, 1000), "x")


def test_mann_whitney_probability_of_superiority():
    rng = np.random.default_rng(5)
    shifted = mann_whitney(rng.normal(0, 1, 500), rng.normal(1, 1, 500))
    assert shifted.prob_superiority > 0.7 and shifted.p_value < 0.001
    x = [1.0, 2.0, 3.0, 4.0]
    assert mann_whitney(x, x).prob_superiority == pytest.approx(0.5)


def test_bootstrap_is_reproducible_and_close_to_welch():
    rng = np.random.default_rng(6)
    a, b = rng.normal(10, 3, 2000), rng.normal(10.3, 3, 2000)
    first = bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=11)
    again = bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=11)
    other = bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=12)
    assert first == again
    assert first != other
    welch = welch_ttest(a, b)
    assert (first.ci_high - first.ci_low) == pytest.approx(welch.ci_high - welch.ci_low, rel=0.1)


def test_bootstrap_interval_covers_true_difference():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 1500), rng.normal(0.5, 1, 1500)
    boot = bootstrap_mean_diff_ci(a, b, n_boot=1000, seed=1)
    assert boot.ci_low < 0.5 < boot.ci_high


def test_methods_agree_for_clear_effect_and_for_noise():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0, 1, 800), rng.normal(0.6, 1, 800)
    assert methods_agree(welch_ttest(a, b), bootstrap_mean_diff_ci(a, b, 500, seed=1))
    c = rng.normal(0, 1, 800)
    assert methods_agree(welch_ttest(a, c), bootstrap_mean_diff_ci(a, c, 500, seed=1))
