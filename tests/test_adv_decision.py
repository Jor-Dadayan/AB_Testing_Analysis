from experimentlab.analysis.decision import decide
from experimentlab.analysis.experiment_quality import Finding
from experimentlab.analysis.guardrails import GuardrailResult
from experimentlab.analysis.interpretation import classify_effect

NAN = float("nan")
GOOD = classify_effect(0.010, 0.006, 0.014, 1e-5, 0.05, 0.005)
UNCERTAIN = classify_effect(0.008, 0.004, 0.012, 0.001, 0.05, 0.005)
NOISE = classify_effect(0.004, -0.010, 0.018, 0.5, 0.05, 0.005)
NO_EFFECT = classify_effect(0.001, -0.002, 0.004, 0.5, 0.05, 0.005)
HARM = classify_effect(-0.010, -0.014, -0.006, 1e-5, 0.05, 0.005)
OK_Q = [Finding("balance", "ok", "ok")]


def gr(status, required=True, name="g"):
    return GuardrailResult(
        name,
        name,
        status,
        required,
        "higher_is_better",
        "mean",
        1,
        1,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        0.05,
        status,
    )


def run(
    conv=GOOD, guards=None, robust=True, holm=(), bh=(), srm=True, quality=OK_Q, sufficient=True
):
    return decide(
        srm_passed=srm,
        quality_findings=quality,
        conversion=conv,
        guardrails=guards if guards is not None else [gr("PASS")],
        conversion_robust=robust,
        segment_harm_holm=list(holm),
        segment_harm_bh=list(bh),
        data_sufficient=sufficient,
    )


def test_all_criteria_met_ships():
    assert run().code == "SHIP"


def test_significant_primary_with_failed_guardrail_is_not_ship():
    assert run(guards=[gr("FAIL")]).code == "INVESTIGATE"


def test_inconclusive_or_unmeasured_required_guardrail_caps_at_monitoring():
    assert run(guards=[gr("INCONCLUSIVE")]).code == "SHIP_WITH_MONITORING"
    assert run(guards=[gr("NOT_EVALUATED")]).code == "SHIP_WITH_MONITORING"


def test_unmeasured_optional_guardrail_warns_but_does_not_block():
    d = run(guards=[gr("PASS"), gr("NOT_EVALUATED", required=False, name="refunds")])
    assert d.code == "SHIP" and any("refunds" in w for w in d.warnings)


def test_unstable_result_is_investigated():
    assert run(robust=False).code == "INVESTIGATE"


def test_confirmed_segment_harm_blocks_but_bh_only_hint_does_not():
    assert run(holm=["device_type: desktop"]).code == "INVESTIGATE"
    d = run(bh=["device_type: desktop"])
    assert d.code == "SHIP" and d.warnings


def test_invalid_experiment_overrides_everything():
    assert run(srm=False).code == "INVALID_EXPERIMENT"
    assert run(quality=[Finding("balance", "critical", "x")]).code == "INVALID_EXPERIMENT"


def test_primary_outcomes():
    assert run(conv=UNCERTAIN).code == "SHIP_WITH_MONITORING"
    assert run(conv=NOISE).code == "NEED_MORE_DATA"
    assert run(conv=NO_EFFECT).code == "DO_NOT_SHIP"
    assert run(conv=HARM).code == "DO_NOT_SHIP"


def test_criteria_table_lists_every_check():
    names = [c.name for c in run().criteria]
    assert any("SRM" in n for n in names) and any("Guardrail" in n for n in names)
    assert any("практическая" in n for n in names) and any("Сегменты" in n for n in names)
