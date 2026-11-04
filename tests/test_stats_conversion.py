import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from experimentlab.analysis.conversion import (
    p_value_from_z,
    srm_check,
    two_proportion_ztest,
)


def test_ztest_matches_hand_calculation():
    r = two_proportion_ztest(100, 1000, 120, 1000)
    assert r.diff == pytest.approx(0.02)
    assert r.se_pooled == pytest.approx(0.013993, abs=1e-5)
    assert r.z == pytest.approx(1.4293, abs=1e-3)
    assert r.p_value == pytest.approx(0.1529, abs=1e-3)
    assert r.ci_low == pytest.approx(-0.007412, abs=1e-4)
    assert r.ci_high == pytest.approx(0.047412, abs=1e-4)
    assert r.rel_lift == pytest.approx(0.2)


def test_ztest_matches_statsmodels():
    z, p = proportions_ztest([120, 100], [1000, 1000])
    r = two_proportion_ztest(100, 1000, 120, 1000)
    assert r.z == pytest.approx(z, rel=1e-6)
    assert r.p_value == pytest.approx(p, rel=1e-6)


def test_p_value_from_z():
    assert p_value_from_z(1.959964) == pytest.approx(0.05, abs=1e-5)
    assert p_value_from_z(0.0, "greater") == pytest.approx(0.5)
    assert p_value_from_z(1.644854, "greater") == pytest.approx(0.05, abs=1e-5)
    assert p_value_from_z(-1.644854, "less") == pytest.approx(0.05, abs=1e-5)


def test_one_sided_is_half_of_two_sided_when_effect_positive():
    two = two_proportion_ztest(100, 1000, 130, 1000)
    one = two_proportion_ztest(100, 1000, 130, 1000, alternative="greater")
    assert one.p_value == pytest.approx(two.p_value / 2)


def test_symmetry_when_groups_swapped():
    a = two_proportion_ztest(100, 1000, 130, 1000)
    b = two_proportion_ztest(130, 1000, 100, 1000)
    assert a.z == pytest.approx(-b.z)
    assert a.p_value == pytest.approx(b.p_value)


def test_relative_lift_interval_contains_estimate():
    r = two_proportion_ztest(1000, 10_000, 1100, 10_000)
    assert r.rel_lift_ci_low < r.rel_lift < r.rel_lift_ci_high


def test_wider_interval_for_smaller_alpha():
    a = two_proportion_ztest(100, 1000, 120, 1000, alpha=0.10)
    b = two_proportion_ztest(100, 1000, 120, 1000, alpha=0.01)
    assert (b.ci_high - b.ci_low) > (a.ci_high - a.ci_low)


@pytest.mark.parametrize("args", [(5, 0, 1, 10), (11, 10, 1, 10), (0, 100, 0, 100)])
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        two_proportion_ztest(*args)


def test_small_counts_flag_assumption_issue():
    assert two_proportion_ztest(5, 1000, 8, 1000).assumption_issues
    assert not two_proportion_ztest(100, 1000, 120, 1000).assumption_issues


def test_srm_detects_mismatch_and_accepts_balance():
    assert srm_check(50_000, 50_000).passed
    assert not srm_check(49_000, 51_000).passed
    assert srm_check(50_300, 49_700).passed  # небольшое случайное отклонение


def test_type_one_error_close_to_alpha():
    rng = np.random.default_rng(1)
    n, sims = 5000, 3000
    xc, xt = rng.binomial(n, 0.10, sims), rng.binomial(n, 0.10, sims)
    rejected = np.mean(
        [
            two_proportion_ztest(int(a), n, int(b), n).p_value < 0.05
            for a, b in zip(xc, xt, strict=True)
        ]
    )
    assert 0.035 < rejected < 0.065


def test_confidence_interval_coverage_close_to_95_percent():
    rng = np.random.default_rng(2)
    n, sims, true_delta = 5000, 3000, 0.01
    xc, xt = rng.binomial(n, 0.10, sims), rng.binomial(n, 0.11, sims)
    results = [two_proportion_ztest(int(a), n, int(b), n) for a, b in zip(xc, xt, strict=True)]
    coverage = np.mean([r.ci_low <= true_delta <= r.ci_high for r in results])
    assert 0.93 < coverage < 0.97
