"""Полный цикл: CSV -> PostgreSQL -> user_analytics -> сверка."""
import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy.exc import OperationalError

from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.db.checks import print_checks, reconcile
from experimentlab.db.connection import get_engine
from experimentlab.db.loader import load_csv
from experimentlab.db.queries import build_analytics

log = logging.getLogger("experimentlab.load")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="путь к CSV (по умолчанию из конфига)")
    ap.add_argument("--force", action="store_true", help="загрузить даже если файл уже загружен")
    args = ap.parse_args()

    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(PROJECT_ROOT / "logs" / "load.log", encoding="utf-8")],
    )
    cfg = load_config()
    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / cfg["paths"]["raw_csv"]

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
        return 1  # причина уже записана в лог и в load_log
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
