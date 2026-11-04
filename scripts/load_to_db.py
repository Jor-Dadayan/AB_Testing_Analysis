"""Полный цикл: CSV -> PostgreSQL -> user_analytics -> сверка."""

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy.exc import OperationalError, SQLAlchemyError

from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.db.checks import print_checks, reconcile
from experimentlab.db.connection import get_engine
from experimentlab.db.loader import load_csv
from experimentlab.db.queries import build_analytics
from experimentlab.logging_setup import configure_logging

log = logging.getLogger("experimentlab.load")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="путь к CSV (по умолчанию из конфига)")
    ap.add_argument("--force", action="store_true", help="загрузить, даже если файл уже загружен")
    args = ap.parse_args()

    configure_logging("load.log")
    cfg = load_config()
    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / cfg["paths"]["raw_csv"]
    if not csv_path.exists():
        log.error("Файл данных не найден: %s (выполните scripts/generate_data.py)", csv_path)
        return 2

    engine = get_engine()
    try:
        status = load_csv(csv_path, engine, force=args.force)
        build_analytics(engine)
        print(f"[{status}] {csv_path}")
        return 0 if print_checks(reconcile(engine, csv_path)) else 1
    except OperationalError:
        log.error("Нет подключения к PostgreSQL. Проверьте: docker compose ps")
        return 1
    except (ValueError, RuntimeError):
        return 1  # причина уже записана в лог и в таблицу load_log
    except SQLAlchemyError as exc:
        log.error("Ошибка базы данных: %s", type(exc).__name__)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
