import pandas as pd

from experimentlab.analysis import eda
from experimentlab.config import load_config
from experimentlab.data.generator import generate_users


def _frame() -> pd.DataFrame:
    cfg = load_config()
    cfg["n_users"] = 5000
    df, _ = generate_users(cfg)
    return df.rename(columns={"group": "group_name"})


def test_randomized_groups_are_balanced():
    table = eda.balance_table(_frame(), ["age", "cart_value"], ["device_type", "country"])
    assert table["balanced"].all()


def test_balance_detects_imbalance():
    df = _frame()
    df.loc[df["group_name"] == "treatment", "age"] += 15
    table = eda.balance_table(df, ["age"], []).set_index("variable")
    assert not table.loc["age", "balanced"]


def test_iqr_summary_flags_extreme_value():
    result = eda.iqr_summary(pd.Series([1, 2, 3, 4, 5, 100]))
    assert result["n_outliers"] == 1


def test_conversion_ci_contains_rate():
    out = eda.conversion_ci(_frame(), "group_name")
    assert ((out["ci_low"] < out["rate"]) & (out["rate"] < out["ci_high"])).all()
