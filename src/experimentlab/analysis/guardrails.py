"""Guardrail-метрики: тест «не хуже допустимого предела» (non-inferiority)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

NAN = float("nan")


@dataclass(frozen=True)
class GuardrailResult:
    name: str
    label: str
    status: str  # PASS / FAIL / INCONCLUSIVE / NOT_EVALUATED
    required: bool
    direction: str
    kind: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    diff: float
    margin_abs: float
    bound: float  # односторонняя граница ДИ в сторону возможного вреда
    limit: float  # допустимый предел в тех же единицах
    alpha: float
    detail: str


def noninferiority(effect: float, se: float, dof: float, margin: float, alpha: float = 0.05):
    """effect в шкале «больше = лучше». Возвращает (статус, нижняя граница, верхняя граница)."""
    if se <= 0 or margin <= 0:
        raise ValueError("se и margin должны быть больше 0")
    crit = stats.norm.ppf(1 - alpha) if not np.isfinite(dof) else stats.t.ppf(1 - alpha, dof)
    low, up = effect - crit * se, effect + crit * se
    status = "PASS" if low > -margin else "FAIL" if up < -margin else "INCONCLUSIVE"
    return status, float(low), float(up)


def _not_evaluated(spec: dict, detail: str) -> GuardrailResult:
    return GuardrailResult(
        name=spec["name"],
        label=spec.get("label", spec["name"]),
        status="NOT_EVALUATED",
        required=bool(spec.get("required", False)),
        direction=spec["direction"],
        kind=spec["kind"],
        n_control=0,
        n_treatment=0,
        mean_control=NAN,
        mean_treatment=NAN,
        diff=NAN,
        margin_abs=NAN,
        bound=NAN,
        limit=NAN,
        alpha=NAN,
        detail=detail,
    )


def evaluate_guardrail(
    df: pd.DataFrame, spec: dict, alpha: float = 0.05, group_col: str = "group_name"
) -> GuardrailResult:
    """Оценивает одну guardrail-метрику по описанию из конфига."""
    if spec["direction"] not in ("higher_is_better", "lower_is_better"):
        raise ValueError("direction: higher_is_better или lower_is_better")
    col = spec["column"]
    if col not in df.columns:
        return _not_evaluated(spec, f"в данных нет столбца «{col}»: метрика не измерена")
    d = df[df["converted"] == 1] if spec.get("population", "all") == "payers" else df
    d = d.dropna(subset=[col])
    g = d[group_col].astype(str)
    c = d.loc[g == "control", col].to_numpy(float)
    t = d.loc[g == "treatment", col].to_numpy(float)
    if c.size < 2 or t.size < 2:
        return _not_evaluated(spec, "слишком мало наблюдений для оценки")

    diff = float(t.mean() - c.mean())
    a, b = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    se = math.sqrt(a + b)
    margin = (
        float(spec["margin_abs"])
        if "margin_abs" in spec
        else float(spec["margin_rel"]) * abs(float(c.mean()))
    )
    if margin <= 0:
        raise ValueError(f"{spec['name']}: допустимый предел должен быть больше 0")
    sign = 1.0 if spec["direction"] == "higher_is_better" else -1.0
    base = dict(
        name=spec["name"],
        label=spec.get("label", spec["name"]),
        required=bool(spec.get("required", False)),
        direction=spec["direction"],
        kind=spec["kind"],
        n_control=int(c.size),
        n_treatment=int(t.size),
        mean_control=float(c.mean()),
        mean_treatment=float(t.mean()),
        diff=diff,
        margin_abs=margin,
        limit=-sign * margin,
        alpha=alpha,
    )
    if se == 0:
        return GuardrailResult(
            status="INCONCLUSIVE", bound=NAN, detail="нулевая дисперсия: оценить нельзя", **base
        )
    dof = (
        (a + b) ** 2 / (a**2 / (c.size - 1) + b**2 / (t.size - 1))
        if spec["kind"] == "mean"
        else float("inf")
    )
    status, low, _ = noninferiority(sign * diff, se, dof, margin, alpha)
    verdict = {
        "PASS": "ухудшение сверх допустимого предела исключено",
        "FAIL": "ухудшение сверх допустимого предела подтверждено",
        "INCONCLUSIVE": "данные не исключают ухудшения сверх предела "
        "(это не доказательство ни вреда, ни безопасности)",
    }[status]
    return GuardrailResult(status=status, bound=float(sign * low), detail=verdict, **base)


def evaluate_guardrails(
    df: pd.DataFrame, specs: list[dict], alpha: float = 0.05
) -> list[GuardrailResult]:
    return [evaluate_guardrail(df, s, alpha) for s in specs]
