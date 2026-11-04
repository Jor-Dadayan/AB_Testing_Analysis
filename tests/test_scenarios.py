"""Три сценария эксперимента проходят весь аналитический конвейер."""

import copy

import pytest

from experimentlab.analysis import eda
from experimentlab.analysis.advanced_engine import run_advanced
from experimentlab.data.generator import generate_users

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", params=["positive", "no_effect", "negative"])
def scenario_run(request, adv_cfg):
    cfg = copy.deepcopy(adv_cfg)
    cfg["scenario"] = request.param
    df, truth = generate_users(cfg)
    return request.param, truth, run_advanced(eda._derive_from_csv(df), cfg, source="test")


def test_estimate_is_close_to_planted_effect(scenario_run):
    _, truth, result = scenario_run
    c = result.base.conversion
    assert abs(c.diff - truth["true_conversion_delta"]) < 4 * c.se_unpooled


def test_decision_is_consistent_with_scenario(scenario_run):
    name, _, result = scenario_run
    assert result.srm.passed
    code, c = result.decision.code, result.base.conversion
    if name == "positive":
        assert c.diff > 0 and c.p_value < 0.05
        assert code not in {"DO_NOT_SHIP", "INVALID_EXPERIMENT"}
    elif name == "no_effect":
        assert code not in {"SHIP", "SHIP_WITH_MONITORING"}
    else:
        assert c.diff < 0 and code == "DO_NOT_SHIP"
