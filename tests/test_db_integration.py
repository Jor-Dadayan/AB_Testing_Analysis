"""Интеграционные тесты PostgreSQL: схема, ограничения, идемпотентность, откат, запросы."""

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from experimentlab.config import SQL_DIR, load_config
from experimentlab.data.generator import generate_users
from experimentlab.db import loader
from experimentlab.db.checks import reconcile
from experimentlab.db.loader import load_csv
from experimentlab.db.queries import build_analytics, list_queries, read_query

pytestmark = pytest.mark.integration
N = 3000
ALLOW_EMPTY = {"q02_segments_having"}  # при 3000 пользователей сегментов >= 1000 может не быть
INSERT_USER = text(
    "INSERT INTO users (user_id, group_id, signup_ts, device_type, country, traffic_source, "
    "is_new_user, age, sessions_prior_30d, cart_items, cart_value) "
    "VALUES (:uid, :gid, now(), :dev, :country, 'organic', true, :age, 0, 1, 10)"
)


def _write_csv(path, n_users, seed=None):
    cfg = load_config()
    cfg["n_users"] = n_users
    if seed is not None:
        cfg["seed"] = seed
    df, _ = generate_users(cfg)
    df.to_csv(path, index=False)
    return path


def _scalar(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar_one()


@pytest.fixture
def csv_path(tmp_path):
    return _write_csv(tmp_path / "users.csv", N)


@pytest.fixture
def loaded(pg_engine, csv_path):
    load_csv(csv_path, pg_engine)
    build_analytics(pg_engine)
    return pg_engine


def test_load_is_idempotent_and_reconciles(pg_engine, csv_path):
    assert load_csv(csv_path, pg_engine) == "loaded"
    assert load_csv(csv_path, pg_engine) == "skipped"
    assert load_csv(csv_path, pg_engine, force=True) == "loaded"
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM users") == N  # дубликатов нет
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM load_log WHERE status = 'success'") == 2
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM load_log WHERE status = 'running'") == 0
    build_analytics(pg_engine)
    assert [c.name for c in reconcile(pg_engine, csv_path) if not c.passed] == []


def test_invalid_data_is_rejected_and_logged(pg_engine, csv_path):
    df = pd.read_csv(csv_path)
    bad = csv_path.with_name("bad.csv")
    pd.concat([df, df.iloc[[0]]]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_csv(bad, pg_engine)
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM users") == 0
    assert _scalar(pg_engine, "SELECT status FROM load_log") == "failed"


def test_failed_load_rolls_back_everything(pg_engine, csv_path, tmp_path, monkeypatch):
    broken = tmp_path / "sql"
    broken.mkdir()
    for name in ("01_schema.sql", "02_upsert_from_staging.sql"):
        (broken / name).write_text((SQL_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    with open(broken / "02_upsert_from_staging.sql", "a", encoding="utf-8") as f:
        f.write("\nSELECT 1/0;\n")  # ошибка ПОСЛЕ успешных вставок
    monkeypatch.setattr(loader, "SQL_DIR", broken)
    with pytest.raises(DBAPIError):
        load_csv(csv_path, pg_engine)
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM users") == 0
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM user_outcomes") == 0
    assert _scalar(pg_engine, "SELECT to_regclass('stg_users') IS NULL") is True
    assert _scalar(pg_engine, "SELECT status FROM load_log") == "failed"


def test_reload_with_smaller_file_removes_stale_users(pg_engine, tmp_path):
    big = _write_csv(tmp_path / "big.csv", 3000)
    small = _write_csv(tmp_path / "small.csv", 2000, seed=7)
    load_csv(big, pg_engine)
    load_csv(small, pg_engine)
    build_analytics(pg_engine)
    assert _scalar(pg_engine, "SELECT COUNT(*) FROM users") == 2000
    assert [c.name for c in reconcile(pg_engine, small) if not c.passed] == []


@pytest.mark.parametrize(
    "params",
    [
        dict(uid=10**9, gid=99, dev="mobile", country="US", age=30),
        dict(uid=10**9, gid=1, dev="fridge", country="US", age=30),
        dict(uid=10**9, gid=1, dev="mobile", country="US", age=5),
        dict(uid=10**9, gid=1, dev="mobile", country=None, age=30),
        dict(uid=1, gid=1, dev="mobile", country="US", age=30),
    ],
    ids=["foreign_key", "check_device", "check_age", "not_null", "primary_key"],
)
def test_database_rejects_invalid_users(loaded, params):
    with pytest.raises(IntegrityError), loaded.begin() as conn:
        conn.execute(INSERT_USER, params)


@pytest.mark.parametrize(
    "update",
    [
        "UPDATE user_outcomes SET revenue = 0 WHERE user_id = "
        "(SELECT MIN(user_id) FROM user_outcomes WHERE converted)",
        "UPDATE user_outcomes SET revenue = 5 WHERE user_id = "
        "(SELECT MIN(user_id) FROM user_outcomes WHERE NOT converted)",
    ],
    ids=["buyer_without_revenue", "non_buyer_with_revenue"],
)
def test_business_rule_is_enforced_by_database(loaded, update):
    with pytest.raises(IntegrityError), loaded.begin() as conn:
        conn.execute(text(update))


def test_analytics_table_matches_csv(loaded, csv_path):
    assert _scalar(loaded, "SELECT COUNT(*) - COUNT(DISTINCT user_id) FROM user_analytics") == 0
    csv = pd.read_csv(csv_path)
    expected = csv.groupby("group").agg(n=("user_id", "size"), buyers=("converted", "sum"))
    with loaded.connect() as conn:
        actual = pd.read_sql(
            text(
                "SELECT group_name, COUNT(*) AS n, SUM(converted) AS buyers "
                "FROM user_analytics GROUP BY group_name ORDER BY group_name"
            ),
            conn,
        ).set_index("group_name")
    assert actual["n"].astype(int).to_dict() == expected["n"].to_dict()
    assert actual["buyers"].astype(int).to_dict() == expected["buyers"].astype(int).to_dict()


def test_all_analytic_queries_run_and_q01_matches_csv(loaded, csv_path):
    for name in list_queries():
        result = read_query(loaded, name)
        assert isinstance(result, pd.DataFrame), name
        assert name in ALLOW_EMPTY or not result.empty, f"{name} вернул пустой результат"
    q01 = read_query(loaded, "q01_group_summary").set_index("group_name")
    csv = pd.read_csv(csv_path).groupby("group")["converted"].mean()
    for group, rate in csv.items():
        assert float(q01.loc[group, "conversion"]) == pytest.approx(rate, abs=6e-5)
