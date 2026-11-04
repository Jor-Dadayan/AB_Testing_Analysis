import numpy as np
import pandas as pd
import pytest

from experimentlab.analysis.guardrails import evaluate_guardrail, noninferiority

MEAN_SPEC = {
    "name": "aov",
    "label": "AOV",
    "column": "value",
    "kind": "mean",
    "population": "all",
    "direction": "higher_is_better",
    "margin_rel": 0.05,
    "required": True,
}
RATE_SPEC = {
    "name": "refund",
    "label": "Возвраты",
    "column": "refunded",
    "kind": "proportion",
    "population": "all",
    "direction": "lower_is_better",
    "margin_abs": 0.02,
    "required": False,
}


def frame(c, t, col):
    return pd.DataFrame(
        {
            "group_name": ["control"] * len(c) + ["treatment"] * len(t),
            "converted": 1,
            col: np.concatenate([c, t]),
        }
    )


def test_noninferiority_statuses():
    assert noninferiority(0.0, 0.01, float("inf"), 0.05)[0] == "PASS"
    assert noninferiority(-0.06, 0.01, float("inf"), 0.05)[0] == "INCONCLUSIVE"
    assert noninferiority(-0.08, 0.01, float("inf"), 0.05)[0] == "FAIL"


def test_missing_column_is_not_evaluated():
    df = pd.DataFrame({"group_name": ["control", "treatment"] * 5, "converted": 1})
    r = evaluate_guardrail(df, RATE_SPEC)
    assert r.status == "NOT_EVALUATED" and "нет столбца" in r.detail


def test_mean_guardrail_pass_and_fail():
    rng = np.random.default_rng(0)
    same = frame(rng.normal(100, 10, 3000), rng.normal(100, 10, 3000), "value")
    assert evaluate_guardrail(same, MEAN_SPEC).status == "PASS"
    worse = frame(rng.normal(100, 10, 2000), rng.normal(80, 10, 2000), "value")
    assert evaluate_guardrail(worse, MEAN_SPEC).status == "FAIL"


def test_noisy_small_sample_is_inconclusive_not_pass():
    rng = np.random.default_rng(1)
    df = frame(rng.normal(100, 80, 40), rng.normal(97, 80, 40), "value")
    assert evaluate_guardrail(df, MEAN_SPEC).status == "INCONCLUSIVE"


def test_lower_is_better_proportion():
    ok = frame(np.r_[np.ones(200), np.zeros(9800)], np.r_[np.ones(200), np.zeros(9800)], "refunded")
    assert evaluate_guardrail(ok, RATE_SPEC).status == "PASS"
    bad = frame(
        np.r_[np.ones(200), np.zeros(9800)], np.r_[np.ones(500), np.zeros(9500)], "refunded"
    )
    r = evaluate_guardrail(bad, RATE_SPEC)
    assert r.status == "FAIL" and r.bound > r.limit > 0


def test_payers_population_filters_rows():
    df = pd.DataFrame(
        {
            "group_name": ["control"] * 10 + ["treatment"] * 10,
            "converted": [1, 0] * 10,
            "value": np.arange(20, dtype=float) + 1,
        }
    )
    r = evaluate_guardrail(df, {**MEAN_SPEC, "population": "payers"})
    assert r.n_control == 5 and r.n_treatment == 5


def test_zero_variance_is_inconclusive():
    df = frame(np.zeros(100), np.zeros(100), "refunded")
    assert evaluate_guardrail(df, RATE_SPEC).status == "INCONCLUSIVE"


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        evaluate_guardrail(
            pd.DataFrame({"group_name": ["control"]}), {**MEAN_SPEC, "direction": "x"}
        )
