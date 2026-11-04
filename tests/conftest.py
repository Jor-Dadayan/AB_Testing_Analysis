import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from experimentlab.analysis import eda
from experimentlab.config import database_url, load_config
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


@pytest.fixture
def pg_engine():
    """Отдельная схема PostgreSQL на каждый тест. Без базы: пропуск, а при REQUIRE_DB=1 падение."""
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(database_url(), connect_args={"connect_timeout": 3})
    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError:
        admin.dispose()
        message = "PostgreSQL недоступен (docker compose up -d db)"
        if os.getenv("REQUIRE_DB") == "1":
            pytest.fail(message)
        pytest.skip(message)
    engine = create_engine(database_url(), connect_args={"options": f"-csearch_path={schema}"})
    yield engine
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()
