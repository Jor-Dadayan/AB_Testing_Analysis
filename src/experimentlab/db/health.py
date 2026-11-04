"""Проверка доступности PostgreSQL для dashboard. Пароли в сообщения не попадают."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from experimentlab.config import database_url

log = logging.getLogger("experimentlab.db.health")


@dataclass(frozen=True)
class DBStatus:
    ok: bool
    message: str
    n_users: int | None = None


def check_database(timeout: int = 3) -> DBStatus:
    """Подключение, наличие таблицы user_analytics и непустое содержимое."""
    engine = None
    try:
        engine = create_engine(database_url(), connect_args={"connect_timeout": timeout})
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT to_regclass('public.user_analytics') IS NOT NULL")
            ).scalar_one()
            if not exists:
                return DBStatus(
                    False,
                    "База доступна, но таблицы user_analytics нет "
                    "(выполните scripts/load_to_db.py)",
                )
            n = int(conn.execute(text("SELECT COUNT(*) FROM user_analytics")).scalar_one())
    except SQLAlchemyError as exc:
        log.warning("PostgreSQL недоступен: %s", type(exc).__name__)
        return DBStatus(False, "PostgreSQL недоступен (проверьте: docker compose ps)")
    finally:
        if engine is not None:
            engine.dispose()
    if n == 0:
        return DBStatus(False, "Таблица user_analytics пуста (выполните scripts/load_to_db.py)", 0)
    return DBStatus(True, "PostgreSQL подключён", n)
