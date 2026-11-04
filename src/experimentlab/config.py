"""Конфигурация проекта: YAML-эксперимент и подключение к БД."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
SQL_DIR = PROJECT_ROOT / "sql"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path | str | None = None) -> dict:
    """Читает YAML-конфиг и возвращает словарь."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def database_url() -> URL | str:
    """Адрес БД из .env (или из переменной DATABASE_URL)."""
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "lab"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "experimentlab"),
    )
