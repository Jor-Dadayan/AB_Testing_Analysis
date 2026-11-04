import pandas as pd
import pytest
from scipy import stats

from experimentlab.analysis.srm import srm_by_segment, srm_report


def test_balanced_groups_pass_without_warning():
    r = srm_report(50_000, 50_000)
    assert r.passed and r.warning is None
    assert r.expected_control == r.expected_treatment == 50_000


def test_mismatch_detected_with_warning():
    r = srm_report(49_000, 51_000)
    assert not r.passed
    assert "SRM" in r.warning and "НЕЛЬЗЯ" in r.warning


def test_matches_scipy_chisquare():
    r = srm_report(49_000, 51_000)
    ref = stats.chisquare([49_000, 51_000], [50_000, 50_000])
    assert r.chi2 == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_unequal_planned_split():
    assert srm_report(70_000, 30_000, expected_treatment_share=0.3).passed


def test_segment_srm_flags_only_broken_segment():
    df = pd.DataFrame({
        "group_name": ["control"] * 5000 + ["treatment"] * 5000 + ["control"] * 6000 + ["treatment"] * 4000,
        "device": ["a"] * 10_000 + ["b"] * 10_000,
    })
    out = srm_by_segment(df, ["device"]).set_index("level")
    assert not out.loc["a", "flagged"] and out.loc["b", "flagged"]
