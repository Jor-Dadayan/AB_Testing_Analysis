"""Поправки на множественные сравнения: Bonferroni, Holm, Benjamini-Hochberg."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experimentlab.analysis.hypothesis import holm_adjust


def _as_p(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values должны быть одномерным массивом чисел из [0, 1]")
    return p


def bonferroni_adjust(p_values) -> np.ndarray:
    """p_adj = min(1, m * p). Контроль вероятности хотя бы одной ложной находки (FWER)."""
    p = _as_p(p_values)
    return np.minimum(1.0, p * p.size)


def benjamini_hochberg_adjust(p_values) -> np.ndarray:
    """q-values BH: контроль ожидаемой доли ложных находок среди находок (FDR)."""
    p = _as_p(p_values)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(1.0, adjusted)
    return out


def family_wise_error_probability(m: int, alpha: float = 0.05) -> float:
    """Вероятность хотя бы одной ложной находки при m независимых тестах без поправки."""
    return 1.0 - (1.0 - alpha) ** m


def compare_corrections(
    labels, p_values, alpha: float = 0.05, fdr_q: float | None = None
) -> pd.DataFrame:
    """Исходные и скорректированные p-values рядом. Для BH порог равен fdr_q (по умолчанию alpha)."""
    cols = ["label", "p_raw", "p_bonferroni", "p_holm", "q_bh",
            "sig_raw", "sig_bonferroni", "sig_holm", "sig_bh"]
    p = _as_p(p_values)
    labels = list(labels)
    if len(labels) != p.size:
        raise ValueError("labels и p_values разной длины")
    if p.size == 0:
        return pd.DataFrame(columns=cols)
    q = fdr_q if fdr_q is not None else alpha
    bonf, holm, bh = bonferroni_adjust(p), holm_adjust(p), benjamini_hochberg_adjust(p)
    return pd.DataFrame({
        "label": labels, "p_raw": p, "p_bonferroni": bonf, "p_holm": holm, "q_bh": bh,
        "sig_raw": p < alpha, "sig_bonferroni": bonf < alpha,
        "sig_holm": holm < alpha, "sig_bh": bh < q,
    })
