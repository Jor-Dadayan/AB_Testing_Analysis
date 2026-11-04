import pytest

from experimentlab.analysis import eda
from experimentlab.config import load_config
from experimentlab.data.generator import generate_users


@pytest.fixture(scope="session")
def adv_cfg():
    cfg = load_config()
    cfg["n_users"] = 100_000
    cfg["analysis"]["n_bootstrap"] = 200
    cfg["advanced"]["n_bootstrap_conversion"] = 2_000
    return cfg


@pytest.fixture(scope="session")
def adv_df(adv_cfg):
    df, _ = generate_users(adv_cfg)
    return eda._derive_from_csv(df)  # добавляет group_name, age_group и другие поля
