"""Запуск аналитических запросов: python scripts/run_queries.py [имя_запроса]."""
import argparse

from experimentlab.db.connection import get_engine
from experimentlab.db.queries import list_queries, read_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="например q03_lift_cte_case")
    args = ap.parse_args()

    engine = get_engine()
    for name in [args.name] if args.name else list_queries():
        print(f"\n=== {name} ===")
        print(read_query(engine, name).to_string(index=False, max_rows=12))
    engine.dispose()


if __name__ == "__main__":
    main()
