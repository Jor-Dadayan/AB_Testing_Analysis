"""Бизнес-правила данных и конфигурации."""

import pytest

from experimentlab.analysis.decision import DEFAULT_CRITERIA
from experimentlab.config import load_config
from experimentlab.data.generator import generate_users


@pytest.fixture(scope="module")
def generated():
    cfg = load_config()
    cfg["n_users"] = 20_000
    return generate_users(cfg)[0]


def test_revenue_exists_only_for_buyers(generated):
    assert ((generated["revenue"] > 0) == (generated["converted"] == 1)).all()
    assert (generated["revenue"] >= 0).all() and (generated["cart_value"] > 0).all()


def test_purchase_never_precedes_signup(generated):
    buyers = generated[generated["converted"] == 1]
    assert (buyers["purchase_ts"] >= buyers["signup_ts"]).all()
    assert generated.loc[generated["converted"] == 0, "purchase_ts"].isna().all()


def test_one_row_and_one_group_per_user(generated):
    assert generated["user_id"].is_unique
    assert set(generated["group"]) == {"control", "treatment"}
    assert (generated["group"] == "treatment").mean() == pytest.approx(0.5, abs=1e-9)


def test_config_thresholds_are_consistent():
    cfg = load_config()
    a, adv = cfg["analysis"], cfg["advanced"]
    assert 0 < a["alpha"] < 0.2 and 0.5 <= a["power"] < 1
    assert 0 < a["practical_threshold_abs"] <= a["planned_mde_abs"]
    assert 0 < cfg["treatment_share"] < 1
    assert 0 < a["srm_alpha"] <= a["alpha"]
    assert set(cfg["scenarios"]) == {"positive", "no_effect", "negative"}
    assert set(adv["success_criteria"]) <= set(DEFAULT_CRITERIA)


def test_guardrail_specs_are_well_formed():
    for g in load_config()["advanced"]["guardrails"]:
        assert {"name", "column", "kind", "direction"} <= g.keys()
        assert g["kind"] in {"mean", "proportion"}
        assert g["direction"] in {"higher_is_better", "lower_is_better"}
        assert g.get("margin_abs", 0) > 0 or g.get("margin_rel", 0) > 0, g["name"]
