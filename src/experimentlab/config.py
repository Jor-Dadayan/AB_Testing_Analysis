"""Чтение конфигурации эксперимента."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Читает YAML-конфиг и возвращает словарь."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
