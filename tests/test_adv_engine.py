import json

import pytest

from experimentlab.analysis.advanced_engine import run_advanced
from experimentlab.analysis.advanced_report import render_advanced_report


@pytest.fixture(scope="module")
def result(adv_df, adv_cfg):
    return run_advanced(adv_df, adv_cfg, source="test")


def test_engine_end_to_end(result):
    assert result.srm.passed
    status = {g.name: g.status for g in result.guardrails}
    assert status["refund_rate"] == "NOT_EVALUATED" and status["checkout_abandonment"] == "NOT_EVALUATED"
    assert status["arpu"] == "PASS" and status["aov"] == "PASS"
    assert result.segments.n_tests >= 20
    assert result.decision.code in {"SHIP", "SHIP_WITH_MONITORING", "NEED_MORE_DATA", "DO_NOT_SHIP",
                                    "INVESTIGATE", "INVALID_EXPERIMENT"}


def test_json_serializable(result):
    payload = json.loads(json.dumps(result.to_dict()))
    assert {"decision", "segments", "guardrails", "sensitivity", "base"} <= payload.keys()


def test_report_has_sections_and_hides_truth(result):
    text = render_advanced_report(result)
    for header in ("## 1. Итоговое решение", "## 2. Проверка SRM", "## 4. Сегментный анализ",
                   "## 5. Множественные сравнения", "## 6. Guardrail-метрики",
                   "## 7. Анализ чувствительности", "## 8. Качество эксперимента"):
        assert header in text
    assert "true_conversion" not in text and "truth.json" not in text
