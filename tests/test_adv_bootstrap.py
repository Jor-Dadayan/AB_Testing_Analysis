import numpy as np
import pytest

from experimentlab.analysis.bootstrap import (
    bootstrap_arpu_diff,
    bootstrap_conversion_diff,
    bootstrap_conversion_relative_lift,
    bootstrap_statistic_diff,
    excludes_zero,
)
from experimentlab.analysis.conversion import two_proportion_ztest
from experimentlab.analysis.revenue import bootstrap_mean_diff_ci


def test_conversion_bootstrap_close_to_analytic_interval():
    b = bootstrap_conversion_diff(1000, 10_000, 1100, 10_000, n_boot=20_000, seed=1)
    z = two_proportion_ztest(1000, 10_000, 1100, 10_000)
    assert b.diff == pytest.approx(0.01)
    assert (b.ci_high - b.ci_low) == pytest.approx(z.ci_high - z.ci_low, rel=0.05)


def test_reproducible_and_seed_sensitive():
    a = bootstrap_conversion_diff(100, 1000, 120, 1000, 2000, seed=5)
    assert a == bootstrap_conversion_diff(100, 1000, 120, 1000, 2000, seed=5)
    assert a != bootstrap_conversion_diff(100, 1000, 120, 1000, 2000, seed=6)


def test_bootstrap_interval_coverage_near_95_percent():
    rng = np.random.default_rng(10)
    hits, sims, true_delta = 0, 500, 0.01
    for i in range(sims):
        xc, xt = int(rng.binomial(5000, 0.10)), int(rng.binomial(5000, 0.11))
        b = bootstrap_conversion_diff(xc, 5000, xt, 5000, 1000, seed=i)
        hits += b.ci_low <= true_delta <= b.ci_high
    assert 0.92 < hits / sims < 0.98


def test_relative_lift_point_and_interval():
    b = bootstrap_conversion_relative_lift(1000, 10_000, 1100, 10_000, 5000, seed=2)
    assert b.diff == pytest.approx(0.10)
    assert b.ci_low < 0.10 < b.ci_high


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bootstrap_conversion_diff(11, 10, 1, 10)
    with pytest.raises(ValueError):
        bootstrap_conversion_relative_lift(0, 100, 5, 100)


def test_generic_default_equals_mean_bootstrap():
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 3, 800), rng.normal(10.5, 3, 800)
    g = bootstrap_statistic_diff(a, b, n_boot=300, seed=4)
    m = bootstrap_mean_diff_ci(a, b, n_boot=300, seed=4)
    assert g.ci_low == pytest.approx(m.ci_low) and g.ci_high == pytest.approx(m.ci_high)
    assert bootstrap_arpu_diff(a, b, 300, seed=4) == m


def test_generic_statistic_median_detects_shift():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 500), rng.normal(1, 1, 500)
    res = bootstrap_statistic_diff(a, b, lambda m: np.median(m, axis=1), n_boot=500, seed=1)
    assert excludes_zero(res) and res.diff > 0.5


def test_binomial_shortcut_matches_resampling_of_binary_vector():
    rng = np.random.default_rng(6)
    c = (rng.random(2000) < 0.10).astype(float)
    t = (rng.random(2000) < 0.12).astype(float)
    full = bootstrap_statistic_diff(c, t, n_boot=2000, seed=1)
    fast = bootstrap_conversion_diff(int(c.sum()), 2000, int(t.sum()), 2000, 2000, seed=1)
    assert (full.ci_high - full.ci_low) == pytest.approx(fast.ci_high - fast.ci_low, rel=0.15)
