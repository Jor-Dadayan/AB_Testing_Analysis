"""Создание подключения SQLAlchemy."""

from sqlalchemy import Engine, create_engine

from experimentlab.config import database_url


def get_engine(url=None) -> Engine:
    """pool_pre_ping отбрасывает «умершие» соединения перед использованием."""
    return create_engine(url or database_url(), pool_pre_ping=True)
