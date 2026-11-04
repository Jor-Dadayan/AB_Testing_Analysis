"""Единая настройка логирования. Уровень задаётся переменной LOG_LEVEL."""

from __future__ import annotations

import logging
import os
import sys

from experimentlab.config import PROJECT_ROOT

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def resolve_log_level(value: str | None) -> int:
    """'DEBUG' -> 10; пустое или неизвестное значение -> INFO."""
    level = logging.getLevelName((value or "INFO").strip().upper())
    return level if isinstance(level, int) else logging.INFO


def configure_logging(log_file: str | None = None) -> None:
    """Вывод в stderr и, при необходимости, в logs/<log_file>."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        path = PROJECT_ROOT / "logs" / log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=resolve_log_level(os.getenv("LOG_LEVEL")),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
