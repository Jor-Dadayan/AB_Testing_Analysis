"""Чтение и запуск SQL-файлов. Сам SQL живёт в sql/, не в Python."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from experimentlab.config import SQL_DIR

QUERIES_DIR = SQL_DIR / "queries"


def run_script(engine: Engine, path: Path | str) -> None:
    """Выполняет SQL-файл (несколько команд) в одной транзакции."""
    sql = Path(path).read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def build_analytics(engine: Engine) -> None:
    """Пересоздаёт таблицу user_analytics."""
    run_script(engine, SQL_DIR / "03_user_analytics.sql")


def query_file(engine: Engine, path: Path | str) -> pd.DataFrame:
    """Выполняет SELECT из файла и возвращает DataFrame."""
    sql = Path(path).read_text(encoding="utf-8")
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def list_queries() -> list[str]:
    return sorted(p.stem for p in QUERIES_DIR.glob("*.sql"))


def read_query(engine: Engine, name: str) -> pd.DataFrame:
    """Результат запроса sql/queries/<name>.sql."""
    return query_file(engine, QUERIES_DIR / f"{name}.sql")
