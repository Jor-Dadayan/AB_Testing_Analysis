import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from experimentlab.analysis.conversion import two_proportion_ztest
from experimentlab.analysis.power import (
    estimate_duration_days,
    mde_two_means,
    mde_two_proportions,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)


def test_sample_size_textbook_value():
    n = sample_size_two_proportions(0.10, 0.11, 0.05, 0.80)
    assert 14_700 < n < 14_800


def test_sample_size_close_to_statsmodels():
    n = sample_size_two_proportions(0.10, 0.11, 0.05, 0.80)
    ref = NormalIndPower().solve_power(
        effect_size=proportion_effectsize(0.11, 0.10), alpha=0.05, power=0.80, ratio=1.0
    )
    assert abs(n - ref) / ref < 0.01


def test_power_round_trip():
    n = sample_size_two_proportions(0.10, 0.11, 0.05, 0.80)
    achieved = power_two_proportions(0.10, 0.11, n, 0.05)
    assert 0.80 <= achieved < 0.801


def test_power_equals_alpha_when_no_effect():
    assert power_two_proportions(0.10, 0.10, 5000, 0.05) == pytest.approx(0.05, abs=1e-6)


def test_power_grows_with_n_and_effect():
    assert power_two_proportions(0.1, 0.11, 20_000) > power_two_proportions(0.1, 0.11, 5_000)
    assert power_two_proportions(0.1, 0.12, 5_000) > power_two_proportions(0.1, 0.11, 5_000)


def test_one_sided_needs_fewer_users():
    two = sample_size_two_proportions(0.10, 0.11, alternative="two-sided")
    one = sample_size_two_proportions(0.10, 0.11, alternative="greater")
    assert one < two


def test_mde_round_trip():
    n = sample_size_two_proportions(0.10, 0.11, 0.05, 0.80)
    mde = mde_two_proportions(0.10, n, 0.05, 0.80)
    assert 0.0099 < mde <= 0.010001


def test_mde_shrinks_with_larger_sample():
    assert mde_two_proportions(0.10, 50_000) < mde_two_proportions(0.10, 10_000)


def test_unequal_allocation_needs_more_total_users():
    equal = sample_size_two_proportions(0.10, 0.11, ratio=1.0)
    skew = sample_size_two_proportions(0.10, 0.11, ratio=0.25)
    assert skew * 1.25 > equal * 2


def test_no_effect_sample_size_is_undefined():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.10, 0.10)


def test_sample_size_for_means_known_value():
    assert sample_size_two_means(sd=1.0, delta=0.5) == 63
    assert mde_two_means(sd=1.0, n_control=63) == pytest.approx(0.5, abs=2e-3)


def test_duration_estimation():
    assert estimate_duration_days(14_751, 3_500, round_to_full_weeks=False) == 9
    assert estimate_duration_days(14_751, 3_500) == 14
    with pytest.raises(ValueError):
        estimate_duration_days(1000, 0)


def test_predicted_power_matches_simulation():
    rng = np.random.default_rng(3)
    n, sims = 5000, 3000
    xc, xt = rng.binomial(n, 0.10, sims), rng.binomial(n, 0.11, sims)
    empirical = np.mean(
        [
            two_proportion_ztest(int(a), n, int(b), n).p_value < 0.05
            for a, b in zip(xc, xt, strict=True)
        ]
    )
    assert empirical == pytest.approx(power_two_proportions(0.10, 0.11, n), abs=0.04)
