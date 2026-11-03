import pandas as pd
import pytest

from experimentlab.config import load_config
from experimentlab.data.generator import generate_users
from experimentlab.db.validation import validate


@pytest.fixture(scope="module")
def clean_df(tmp_path_factory):
    cfg = load_config()
    cfg["n_users"] = 2000
    df, _ = generate_users(cfg)
    path = tmp_path_factory.mktemp("val") / "users.csv"
    df.to_csv(path, index=False)
    return pd.read_csv(path)


def _has(errors, fragment):
    return any(fragment in e for e in errors)


def test_clean_data_passes(clean_df):
    assert validate(clean_df) == []


def test_missing_column(clean_df):
    assert _has(validate(clean_df.drop(columns=["revenue"])), "нет столбцов")


def test_duplicates(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[0]]])
    assert _has(validate(df), "дубликаты")


def test_null_in_required_column(clean_df):
    df = clean_df.copy()
    df.loc[0, "country"] = None
    assert _has(validate(df), "NULL")


def test_wrong_type(clean_df):
    df = clean_df.copy()
    df["age"] = df["age"].astype(str)
    assert _has(validate(df), "тип")


def test_unknown_category(clean_df):
    df = clean_df.copy()
    df.loc[0, "device_type"] = "smart_fridge"
    assert _has(validate(df), "device_type")


def test_negative_revenue(clean_df):
    df = clean_df.copy()
    i = df.index[df["converted"] == 1][0]
    df.loc[i, "revenue"] = -5.0
    assert _has(validate(df), "revenue")


def test_revenue_without_purchase(clean_df):
    df = clean_df.copy()
    i = df.index[df["converted"] == 0][0]
    df.loc[i, "revenue"] = 10.0
    assert _has(validate(df), "согласован")


def test_purchase_before_signup(clean_df):
    df = clean_df.copy()
    i = df.index[df["converted"] == 1][0]
    df.loc[i, "purchase_ts"] = "2000-01-01 00:00:00"
    assert _has(validate(df), "раньше")
