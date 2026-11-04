import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from experimentlab.analysis.hypothesis import holm_adjust
from experimentlab.analysis.multiple_testing import (
    benjamini_hochberg_adjust,
    bonferroni_adjust,
    compare_corrections,
    family_wise_error_probability,
)


@pytest.mark.parametrize(
    "ours,method",
    [
        (bonferroni_adjust, "bonferroni"),
        (holm_adjust, "holm"),
        (benjamini_hochberg_adjust, "fdr_bh"),
    ],
)
def test_matches_statsmodels(ours, method):
    p = np.random.default_rng(0).uniform(0, 0.2, 15)
    assert ours(p) == pytest.approx(multipletests(p, method=method)[1])


def test_bh_known_values():
    assert benjamini_hochberg_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.02]
    )


def test_bonferroni_caps_at_one():
    assert bonferroni_adjust([0.5, 0.9]).tolist() == [1.0, 1.0]


def test_ordering_bh_le_holm_le_bonferroni():
    p = np.random.default_rng(1).uniform(0, 1, 30)
    bh, holm, bonf = benjamini_hochberg_adjust(p), holm_adjust(p), bonferroni_adjust(p)
    assert np.all(bh <= holm + 1e-12) and np.all(holm <= bonf + 1e-12)


def test_family_wise_error_probability():
    assert family_wise_error_probability(1) == pytest.approx(0.05)
    assert family_wise_error_probability(20) == pytest.approx(0.6415, abs=1e-3)


def test_global_null_simulation():
    p = np.random.default_rng(2).uniform(size=(2000, 20))
    raw = (p < 0.05).any(axis=1).mean()
    bonf = np.mean([(bonferroni_adjust(r) < 0.05).any() for r in p])
    bh = np.mean([(benjamini_hochberg_adjust(r) < 0.05).any() for r in p])
    assert 0.58 < raw < 0.70  # без поправки ложная находка почти у каждого второго
    assert 0.03 < bonf < 0.07
    assert 0.03 < bh < 0.07


def test_compare_corrections_table_and_validation():
    out = compare_corrections(["a", "b", "c"], [0.001, 0.02, 0.6])
    assert list(out["label"]) == ["a", "b", "c"]
    assert out.loc[0, "sig_bonferroni"] and not out.loc[2, "sig_raw"]
    assert out.loc[1, "sig_raw"] and not out.loc[1, "sig_bonferroni"]
    with pytest.raises(ValueError):
        compare_corrections(["a"], [1.5])
    assert compare_corrections([], []).empty
