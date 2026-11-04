"""Сверка исходного CSV с базой и проверки целостности."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from experimentlab.check_report import Check, print_check_report
from experimentlab.config import SQL_DIR
from experimentlab.db.queries import query_file

TABLES = ("users", "user_outcomes", "user_analytics")


def reconcile(engine: Engine, csv_path: Path | str) -> list[Check]:
    """Сравнивает CSV и БД по числу строк и контрольным суммам, затем SQL-проверки."""
    csv = pd.read_csv(csv_path, usecols=["user_id", "converted", "revenue"])
    n_csv = len(csv)
    checks = [Check("CSV: user_id уникален", bool(csv["user_id"].is_unique))]

    with engine.connect() as conn:
        for table in TABLES:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            checks.append(
                Check(f"строк в {table} равно строкам в CSV", n == n_csv, f"БД={n}, CSV={n_csv}")
            )
        buyers_db, revenue_db = conn.execute(
            text(
                "SELECT COUNT(*) FILTER (WHERE converted), COALESCE(SUM(revenue), 0) "
                "FROM user_outcomes"
            )
        ).one()

    buyers_csv = int(csv["converted"].sum())
    revenue_csv = float(csv["revenue"].sum())
    checks.append(
        Check(
            "число покупателей совпадает",
            buyers_db == buyers_csv,
            f"БД={buyers_db}, CSV={buyers_csv}",
        )
    )
    checks.append(
        Check(
            "суммарная выручка совпадает",
            math.isclose(float(revenue_db), revenue_csv, abs_tol=0.01),
            f"БД={float(revenue_db):.2f}, CSV={revenue_csv:.2f}",
        )
    )

    integrity = query_file(engine, SQL_DIR / "04_integrity_checks.sql")
    for row in integrity.itertuples(index=False):
        checks.append(
            Check(
                f"SQL-проверка {row.check_name} = 0",
                int(row.violations) == 0,
                f"нарушений: {row.violations}",
            )
        )
    return checks


def print_checks(checks: list[Check]) -> bool:
    """Печатает отчёт сверки, возвращает True, если всё пройдено."""
    return print_check_report(checks, "Reconciliation")
