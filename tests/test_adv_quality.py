import numpy as np
import pandas as pd

from experimentlab.analysis.experiment_quality import (
    check_balance,
    check_design,
    check_duration,
    check_sample,
    check_stability,
)

NUM = ["age", "sessions_prior_30d", "cart_items", "cart_value", "signup_hour"]
CAT = ["country", "device_type", "traffic_source", "is_new_user", "age_group", "is_weekend"]


def test_balance_ok_on_randomized_data(adv_df):
    findings, table = check_balance(adv_df, NUM, CAT)
    assert findings[0].severity == "ok" and table["balanced"].all()


def test_balance_detects_broken_randomization(adv_df):
    d = adv_df.copy()
    d.loc[d["group_name"] == "treatment", "age"] += 15
    assert check_balance(d, NUM, CAT)[0][0].severity == "critical"


def test_duration_ok_for_four_full_weeks():
    dates = pd.Series(pd.date_range("2025-01-06", periods=28).repeat(3))
    assert all(f.severity == "ok" for f in check_duration(dates, required_days=14))


def test_duration_warnings():
    dates = pd.Series(pd.date_range("2025-01-06", periods=10).repeat(3))
    codes = {f.code for f in check_duration(dates, required_days=14)}
    assert {"duration_weeks", "duration_min", "duration_required"} <= codes
    gap = pd.Series(pd.date_range("2025-01-06", periods=28).repeat(3))
    gap = gap[gap != pd.Timestamp("2025-01-10")]
    assert "duration_gaps" in {f.code for f in check_duration(gap, 14)}


def test_sample_findings():
    codes = {f.code: f.severity for f in check_sample(5000, 14_751, 0.012, 0.005, 0.2, 0.03)}
    assert codes["sample_planned"] == "warning" and codes["sample_practical"] == "warning"
    ok = check_sample(50_000, 14_751, 0.004, 0.005, 0.01, 0.03)
    assert len(ok) == 1 and ok[0].severity == "ok"


def _stab_frame(early_treat_rate_pct, late_treat_rate_pct):
    days = pd.date_range("2025-01-06", periods=28)
    parts = []
    for half, rate in ((days[:14], early_treat_rate_pct), (days[14:], late_treat_rate_pct)):
        for g, pct_ in (("control", 10), ("treatment", rate)):
            idx = np.arange(10_000)
            parts.append(pd.DataFrame({"group_name": g, "signup_date": np.tile(half, 10_000 // 14 + 1)[:10_000],
                                       "converted": (idx % 100 < pct_).astype(int)}))
    return pd.concat(parts, ignore_index=True)


def test_stability_flags_fading_effect_and_accepts_stable_one():
    fading, info = check_stability(_stab_frame(13, 10))
    assert fading.severity == "warning" and info["diff_early"] > info["diff_late"]
    stable, _ = check_stability(_stab_frame(10, 10))
    assert stable.severity == "ok"


def test_design_baseline_mismatch():
    cfg = {"treatment_share": 0.5, "baseline_conversion": 0.10}
    assert check_design(cfg, 0.10)[0].severity == "ok"
    assert check_design(cfg, 0.14)[0].severity == "warning"
