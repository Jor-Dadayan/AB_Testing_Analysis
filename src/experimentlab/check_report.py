"""Общий формат проверок (датасет, сверка БД). Единственное место, где печатается отчёт."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def print_check_report(checks: list[Check], title: str) -> bool:
    """Печатает проверки. Пустой список считается провалом: проверять было нечего."""
    for c in checks:
        extra = f"  ({c.detail})" if c.detail else ""
        print(f"[{'OK  ' if c.passed else 'FAIL'}] {c.name}{extra}")
    ok = sum(c.passed for c in checks)
    success = bool(checks) and ok == len(checks)
    print(f"\n{title} {'PASSED' if success else 'FAILED'}: {ok}/{len(checks)} checks ok")
    return success
