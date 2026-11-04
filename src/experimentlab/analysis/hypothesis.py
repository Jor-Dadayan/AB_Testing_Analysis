"""Формулировки гипотез и вспомогательные функции проверки гипотез."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HypothesisSpec:
    metric: str
    role: str
    h0: str
    h1: str
    test: str


def build_hypotheses() -> list[HypothesisSpec]:
    """План анализа, зафиксированный ДО просмотра результатов."""
    return [
        HypothesisSpec(
            "Конверсия",
            "первичная (по ней принимается решение)",
            "p_T = p_C",
            "p_T ≠ p_C",
            "Z-тест двух долей, двусторонний",
        ),
        HypothesisSpec(
            "ARPU",
            "вторичная (контроль выручки)",
            "μ_T = μ_C",
            "μ_T ≠ μ_C",
            "Тест Уэлча, двусторонний",
        ),
        HypothesisSpec(
            "ARPU",
            "проверка устойчивости",
            "μ_T = μ_C",
            "μ_T ≠ μ_C",
            "Bootstrap-интервал (percentile)",
        ),
        HypothesisSpec(
            "Распределение выручки",
            "диагностика (не про среднее)",
            "P(Y_T > Y_C) = 0.5",
            "P(Y_T > Y_C) ≠ 0.5",
            "Манн-Уитни",
        ),
    ]


def error_rates(alpha: float, power: float) -> dict[str, float]:
    """Ошибка I рода = α (ложно найти эффект). Ошибка II рода = 1 - мощность (пропустить)."""
    return {"type_1_error": alpha, "type_2_error": 1.0 - power}


def holm_adjust(p_values) -> np.ndarray:
    """Поправка Холма на множественные сравнения (контролирует FWER).

    Понадобится при сегментном анализе: много сегментов = много шансов на случайную находку.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(p.size)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (p.size - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
