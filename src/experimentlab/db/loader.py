"""Идемпотентная загрузка CSV в PostgreSQL: валидация, staging, upsert, сверка."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from experimentlab.config import SQL_DIR
from experimentlab.db.queries import run_script
from experimentlab.db.validation import DB_COLUMNS, validate

log = logging.getLogger("experimentlab.db.loader")
STAGING_TABLE = "stg_users"


def file_sha256(path: Path) -> str:
    """Хеш файла: по нему определяем, что этот файл уже загружен."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы и имена к виду staging-таблицы."""
    df = raw[DB_COLUMNS].rename(columns={"group": "group_name"}).copy()
    df["signup_ts"] = pd.to_datetime(df["signup_ts"])
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    df["is_new_user"] = df["is_new_user"].astype(bool)
    df["converted"] = df["converted"].astype(bool)
    return df


def _already_loaded(engine: Engine, sha: str) -> bool:
    """True, если этот файл уже загружен и в базе столько же строк, сколько было тогда."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT rows_in_db FROM load_log "
            "WHERE file_sha256 = :h AND status = 'success' "
            "ORDER BY load_id DESC LIMIT 1"), {"h": sha}).scalar_one_or_none()
        if rows is None:
            return False
        n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
    return n_users == rows


def _start_log(engine: Engine, sha: str, n_csv: int) -> int:
    with engine.begin() as conn:
        return conn.execute(text(
            "INSERT INTO load_log (file_sha256, rows_in_csv, status) "
            "VALUES (:h, :n, 'running') RETURNING load_id"),
            {"h": sha, "n": n_csv}).scalar_one()


def _finish_log(engine: Engine, load_id: int, status: str, n_db: int | None,
                error: str | None) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE load_log SET finished_at = now(), status = :s, "
            "rows_in_db = :n, error = :e WHERE load_id = :i"),
            {"s": status, "n": n_db, "e": error, "i": load_id})


def load_csv(csv_path: Path | str, engine: Engine, force: bool = False) -> str:
    """Загружает CSV в БД. Возвращает 'loaded' или 'skipped'."""
    csv_path = Path(csv_path)
    sha = file_sha256(csv_path)
    run_script(engine, SQL_DIR / "01_schema.sql")

    if not force and _already_loaded(engine, sha):
        log.info("Файл уже загружен (хеш совпал), пропуск")
        return "skipped"

    raw = pd.read_csv(csv_path)
    load_id = _start_log(engine, sha, len(raw))

    problems = validate(raw)
    if problems:
        for p in problems:
            log.error("Валидация: %s", p)
        _finish_log(engine, load_id, "failed", None, "; ".join(problems)[:1000])
        raise ValueError("Валидация не пройдена: " + "; ".join(problems))

    df = prepare(raw)
    try:
        with engine.begin() as conn:  # одна транзакция: всё или ничего
            df.to_sql(STAGING_TABLE, conn, if_exists="replace", index=False,
                      chunksize=5000, method="multi")
            conn.exec_driver_sql((SQL_DIR / "02_upsert_from_staging.sql")
                                 .read_text(encoding="utf-8"))
            conn.exec_driver_sql(f"DROP TABLE {STAGING_TABLE}")
            n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
            n_out = conn.execute(text("SELECT COUNT(*) FROM user_outcomes")).scalar_one()
            if n_users != len(df) or n_out != len(df):
                raise RuntimeError(
                    f"Расхождение: CSV={len(df)}, users={n_users}, outcomes={n_out}")
    except Exception as exc:
        log.exception("Загрузка не удалась, транзакция откатена")
        _finish_log(engine, load_id, "failed", None, repr(exc)[:1000])
        raise

    _finish_log(engine, load_id, "success", n_users, None)
    log.info("Загружено строк: %s", n_users)
    return "loaded"
