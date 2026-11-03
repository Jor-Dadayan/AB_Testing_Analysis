import pytest

from experimentlab.config import load_config
from experimentlab.data.generator import generate_users, run_generation
from experimentlab.data.quality import run_quality_checks


@pytest.fixture
def cfg():
    c = load_config()
    c["n_users"] = 100_000
    return c


def _lift(df):
    m = df.groupby("group")["converted"].mean()
    return m["treatment"] - m["control"]


def test_reproducible(cfg):
    a, _ = generate_users(cfg)
    b, _ = generate_users(cfg)
    assert a.equals(b)


def test_other_seed_differs(cfg):
    a, _ = generate_users(cfg)
    cfg["seed"] = 7
    b, _ = generate_users(cfg)
    assert not a.equals(b)


def test_one_row_per_user_and_split(cfg):
    df, _ = generate_users(cfg)
    assert len(df) == 100_000
    assert df["user_id"].is_unique
    assert (df["group"] == "treatment").sum() == 50_000


def test_quality_checks_pass(cfg):
    df, _ = generate_users(cfg)
    failed = [c.name for c in run_quality_checks(df, cfg) if not c.passed]
    assert failed == []


@pytest.mark.parametrize("scenario,low,high", [
    ("positive", 0.0, 0.02),
    ("no_effect", -0.01, 0.01),
    ("negative", -0.02, 0.0),
])
def test_scenarios_recover_effect(cfg, scenario, low, high):
    cfg["scenario"] = scenario
    df, truth = generate_users(cfg)
    assert low < _lift(df) < high
    delta = truth["true_conversion_treatment"] - truth["true_conversion_control"]
    assert delta == pytest.approx(truth["true_conversion_delta"], abs=1e-4)


def test_rerun_is_idempotent(cfg, tmp_path):
    cfg["n_users"] = 5_000
    csv, truth = tmp_path / "u.csv", tmp_path / "t.json"
    assert run_generation(cfg, csv, truth) == "generated"
    assert run_generation(cfg, csv, truth) == "skipped"
    assert run_generation(cfg, csv, truth, force=True) == "generated"
