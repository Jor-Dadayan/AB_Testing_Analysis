import numpy as np
import pytest

from experimentlab.analysis.hypothesis import build_hypotheses, error_rates, holm_adjust
from experimentlab.analysis.interpretation import (
    classify_effect,
    p_value_statement,
    recommend,
    sufficiency,
)

THR = 0.005


def _classify(diff, lo, hi, p):
    return classify_effect(diff, lo, hi, p, 0.05, THR)


def test_practical_significance():
    v = _classify(0.010, 0.006, 0.014, 1e-5)
    assert v.significant and v.practical == "practical" and v.direction == "positive"
    assert not v.ci_includes_zero


def test_significant_but_practically_uncertain():
    assert _classify(0.008, 0.004, 0.012, 0.001).practical == "uncertain"


def test_significant_but_negligible():
    assert _classify(0.002, 0.0005, 0.0035, 0.01).practical == "negligible"


def test_harmful_effect():
    v = _classify(-0.010, -0.014, -0.006, 1e-5)
    assert v.practical == "harmful" and v.direction == "negative"


def test_not_significant_interval_inside_threshold():
    v = _classify(0.001, -0.002, 0.004, 0.5)
    assert not v.significant and v.ci_includes_zero
    assert v.practical == "no_practical_effect"


def test_not_significant_wide_interval_is_inconclusive():
    v = _classify(0.004, -0.010, 0.018, 0.5)
    assert v.practical == "inconclusive"
    assert "не означает отсутствия эффекта" in v.summary


def test_text_never_accepts_null_from_high_p_value():
    for v in (_classify(0.001, -0.002, 0.004, 0.5), _classify(0.004, -0.010, 0.018, 0.5)):
        assert "принима" not in v.summary.lower()
        assert "H0 верна" not in v.summary


def test_p_value_statement_does_not_claim_probability_of_h0():
    text = p_value_statement(0.03, 0.05).lower()
    assert "не вероятность того, что h0 верна" in text


def test_sufficiency_flags():
    ok = sufficiency(50_000, 14_751, 0.0053, 0.005, 0.01)
    assert ok["data_sufficient"] and not ok["can_detect_threshold"]
    small = sufficiency(5_000, 14_751, 0.017, 0.005, 0.01)
    assert not small["data_sufficient"]


def test_recommendation_rules():
    good = _classify(0.010, 0.006, 0.014, 1e-5)
    noise = _classify(0.004, -0.010, 0.018, 0.5)
    bad = _classify(-0.010, -0.014, -0.006, 1e-5)
    assert recommend(True, good, noise, True, True)[0] == "SHIP"
    assert recommend(True, noise, noise, True, False)[0] == "NEED_MORE_DATA"
    assert recommend(True, bad, noise, True, True)[0] == "DO_NOT_SHIP"
    assert recommend(False, good, noise, True, True)[0] == "INVALID_EXPERIMENT"
    arpu_down = _classify(-0.5, -0.8, -0.2, 0.001)
    assert recommend(True, good, arpu_down, True, True)[0] == "INVESTIGATE"


def test_holm_adjustment_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.all(holm_adjust([0.5, 0.9]) <= 1.0)


def test_error_rates_and_hypotheses():
    assert error_rates(0.05, 0.8) == {"type_1_error": 0.05, "type_2_error": pytest.approx(0.2)}
    assert len(build_hypotheses()) == 4
