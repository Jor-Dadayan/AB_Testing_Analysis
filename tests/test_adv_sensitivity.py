import numpy as np
import pandas as pd

from experimentlab.analysis.sensitivity import (
    MethodEstimate,
    adjusted_effect,
    conversion_estimates,
    outlier_influence,
    revenue_estimates,
    robustness,
)


def test_conversion_methods_agree_on_generated_data(adv_df):
    est = conversion_estimates(adv_df, 0.05, 2000, seed=1)
    values = [e.estimate for e in est]
    assert max(values) - min(values) < 0.005
    assert robustness(est, "conversion")[0]


def test_adjustment_does_not_widen_interval_much(adv_df):
    est = {e.method: e for e in conversion_estimates(adv_df, 0.05, 2000, seed=1)}
    z, reg = est["Z-тест долей (первичный)"], est["Регрессия с ковариатами (HC1)"]
    assert (reg.ci_high - reg.ci_low) <= (z.ci_high - z.ci_low) * 1.02
    tau, lo, hi, _ = adjusted_effect(adv_df)
    assert lo < tau < hi


def _est(method, sig, value=0.01):
    return MethodEstimate("m", method, True, value, None, None, None, sig)


def test_robustness_rules():
    assert robustness([_est("a", True), _est("b", True)], "m")[0]
    ok, why = robustness([_est("a", True), _est("b", False)], "m")
    assert not ok and "расхождение" in why
    assert not robustness([_est("a", True, 0.01), _est("b", True, -0.01)], "m")[0]
    assert not robustness([], "m")[0]


def test_outlier_changes_raw_estimate_but_not_winsorized():
    rng = np.random.default_rng(0)
    n = 20_000

    def draw():
        return np.where(rng.random(n) < 0.1, rng.lognormal(3.5, 0.5, n), 0.0)

    rt = draw()
    rt[0] = 100_000.0
    df = pd.DataFrame(
        {"group_name": ["control"] * n + ["treatment"] * n, "revenue": np.concatenate([draw(), rt])}
    )
    est = {e.method: e for e in revenue_estimates(df, 0.05, 200, 1, [0.99])}
    raw = est["Тест Уэлча (вторичный)"].estimate
    win = next(e for k, e in est.items() if "винзоризации" in k).estimate
    assert raw - win > 1.0
    assert outlier_influence(df)["treatment"]["max_order_share"] > 0.5
