import json

import pytest

from experimentlab.analysis.engine import run_analysis
from experimentlab.analysis.report import render_report
from experimentlab.config import load_config
from experimentlab.data.generator import generate_users


@pytest.fixture(scope="module")
def prepared():
    cfg = load_config()
    cfg["n_users"] = 20_000
    cfg["analysis"]["n_bootstrap"] = 200
    df, _ = generate_users(cfg)
    df = df.rename(columns={"group": "group_name"})
    return df, cfg


def test_engine_end_to_end(prepared):
    df, cfg = prepared
    res = run_analysis(df, cfg, source="test")
    assert res.meta["n_control"] + res.meta["n_treatment"] == 20_000
    assert res.srm.passed
    assert res.conversion.n_control == res.meta["n_control"]
    assert res.recommendation in {"SHIP", "SHIP_WITH_MONITORING", "DO_NOT_SHIP",
                                  "NEED_MORE_DATA", "INVESTIGATE", "INVALID_EXPERIMENT"}


def test_engine_is_deterministic(prepared):
    df, cfg = prepared
    a, b = run_analysis(df, cfg), run_analysis(df, cfg)
    assert a.conversion == b.conversion
    assert a.arpu_bootstrap == b.arpu_bootstrap


def test_result_is_json_serializable(prepared):
    df, cfg = prepared
    payload = json.dumps(run_analysis(df, cfg).to_dict(), default=str)
    assert "conversion" in json.loads(payload)


def test_report_has_sections_and_hides_simulation_truth(prepared):
    df, cfg = prepared
    text = render_report(run_analysis(df, cfg, source="test"))
    for header in ("## 1. Резюме", "## 5. Конверсия", "## 7. Мощность", "## 8. Статистический вывод"):
        assert header in text
    assert "truth" not in text.lower() and "true_conversion" not in text


def test_engine_requires_both_groups(prepared):
    df, cfg = prepared
    with pytest.raises(ValueError):
        run_analysis(df[df["group_name"] == "control"], cfg)
