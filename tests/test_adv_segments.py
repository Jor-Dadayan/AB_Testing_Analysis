import pandas as pd
import pytest

from experimentlab.analysis.segments import analyze_segments, significant_levels


def build(levels: dict) -> pd.DataFrame:
    """levels: уровень -> (n_c, x_c, n_t, x_t); данные детерминированы, без случайности."""
    parts = []
    for lvl, (nc, xc, nt, xt) in levels.items():
        parts.append(pd.DataFrame({"group_name": "control", "device_type": lvl,
                                   "converted": [1] * xc + [0] * (nc - xc)}))
        parts.append(pd.DataFrame({"group_name": "treatment", "device_type": lvl,
                                   "converted": [1] * xt + [0] * (nt - xt)}))
    return pd.concat(parts, ignore_index=True)


def test_effect_in_one_segment_survives_correction_and_omnibus_detects_it():
    df = build({"mobile": (20_000, 2000, 20_000, 3000),
                "desktop": (20_000, 2000, 20_000, 2000),
                "tablet": (20_000, 2000, 20_000, 2000)})
    res = analyze_segments(df, ["device_type"])
    t = res.table.set_index("level")
    assert res.n_tests == 3
    assert t.loc["mobile", "sig_holm"] and not t.loc["desktop", "sig_holm"]
    assert t.loc["mobile", "ci_low_bonf"] < t.loc["mobile", "ci_low"]  # скорр. интервал шире
    assert res.heterogeneity.loc[0, "p_value"] < 1e-10


def test_identical_segments_have_no_findings():
    df = build({k: (10_000, 1000, 10_000, 1000) for k in ("a", "b", "c")})
    res = analyze_segments(df, ["device_type"])
    assert not res.table["sig_raw"].any()
    assert res.heterogeneity.loc[0, "p_value"] > 0.99


def test_small_segments_are_skipped_not_tested():
    df = build({"big": (10_000, 1000, 10_000, 1000), "big2": (10_000, 1000, 10_000, 1000),
                "tiny": (100, 10, 100, 12)})
    res = analyze_segments(df, ["device_type"], min_users=500)
    assert res.n_tests == 2 and res.skipped[0]["level"] == "tiny"


def test_harmful_segment_is_reported():
    df = build({"mobile": (20_000, 2000, 20_000, 2000), "desktop": (20_000, 3000, 20_000, 2000)})
    assert significant_levels(analyze_segments(df, ["device_type"]).table, "holm", "negative") \
        == ["device_type: desktop"]


def test_raw_significant_but_not_after_multiplicity_correction():
    levels = {f"l{i}": (5000, 500, 5000, 500) for i in range(9)}
    levels["target"] = (5000, 500, 5000, 575)  # +1.5 п.п., сырой p около 0.015
    t = analyze_segments(build(levels), ["device_type"]).table.set_index("level")
    assert t.loc["target", "sig_raw"]
    assert not (t.loc["target", "sig_bonferroni"] or t.loc["target", "sig_holm"] or t.loc["target", "sig_bh"])
    assert t.loc["target", "p_value"] == pytest.approx(0.0155, abs=0.002)
    assert t.loc["target", "p_bonferroni"] == pytest.approx(10 * t.loc["target", "p_value"])
