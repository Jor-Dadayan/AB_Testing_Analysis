"""Интеграционные тесты: работают в отдельной схеме PostgreSQL, пропускаются без базы."""

import uuid

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from experimentlab.config import database_url, load_config
from experimentlab.data.generator import generate_users
from experimentlab.db.checks import reconcile
from experimentlab.db.loader import load_csv
from experimentlab.db.queries import build_analytics

N = 3000


@pytest.fixture
def engine():
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(database_url())
    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError:
        pytest.skip("PostgreSQL недоступен (docker compose up -d --wait)")
    eng = create_engine(database_url(), connect_args={"options": f"-csearch_path={schema}"})
    yield eng
    eng.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()


@pytest.fixture
def csv_path(tmp_path):
    cfg = load_config()
    cfg["n_users"] = N
    df, _ = generate_users(cfg)
    path = tmp_path / "users.csv"
    df.to_csv(path, index=False)
    return path


def _count(engine, table):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()


def test_load_is_idempotent_and_reconciles(engine, csv_path):
    assert load_csv(csv_path, engine) == "loaded"
    assert load_csv(csv_path, engine) == "skipped"
    assert load_csv(csv_path, engine, force=True) == "loaded"
    assert _count(engine, "users") == N  # дубликатов нет
    build_analytics(engine)
    failed = [c for c in reconcile(engine, csv_path) if not c.passed]
    assert failed == []


def test_invalid_data_is_rejected_and_logged(engine, csv_path):
    df = pd.read_csv(csv_path)
    bad = csv_path.with_name("bad.csv")
    pd.concat([df, df.iloc[[0]]]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_csv(bad, engine)
    assert _count(engine, "users") == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status FROM load_log")).scalar_one() == "failed"


def test_database_constraint_blocks_bad_row(engine, csv_path):
    load_csv(csv_path, engine)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE user_outcomes SET revenue = 5 WHERE user_id = "
                "(SELECT MIN(user_id) FROM user_outcomes WHERE NOT converted)"
            )
        )
