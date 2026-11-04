"""Слой доступа dashboard: загрузка данных, фильтры, кеш, запуск анализа.

Здесь нет статистики: все расчёты выполняют модули experimentlab.analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from experimentlab.analysis import eda
from experimentlab.analysis.advanced_engine import AdvancedResult, run_advanced
from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.db.health import DBStatus, check_database

log = logging.getLogger("experimentlab.app")
FilterKey = tuple[tuple[str, tuple[str, ...]], ...]

FILTER_SPECS = {
    "device_type": ("Устройство", list(eda.ORDERS["device_type"])),
    "country": ("Страна", list(eda.ORDERS["country"])),
    "traffic_source": ("Канал трафика", list(eda.ORDERS["traffic_source"])),
    "age_group": ("Возраст", list(eda.ORDERS["age_group"])),
    "is_new_user": ("Тип пользователя", ["0", "1"]),
}
VALUE_LABELS = {"is_new_user": {"0": "Вернувшиеся", "1": "Новые"}}


class DataUnavailableError(Exception):
    """Нет источника данных (ни базы, ни файла)."""


class AnalysisUnavailableError(Exception):
    """Анализ невозможен для выбранных данных (мало пользователей, ошибка расчёта)."""


@dataclass
class Context:
    cfg: dict
    experiment_id: str
    experiment_label: str
    source: str
    db: DBStatus
    filter_key: FilterKey
    filter_summary: str
    df: pd.DataFrame | None
    result: AdvancedResult | None  # результат для текущего просмотра (с фильтрами)
    official: AdvancedResult | None  # результат по всей аудитории
    error: str | None = None
    error_kind: str | None = None  # data / analysis
    extra: dict = field(default_factory=dict)

    @property
    def is_filtered(self) -> bool:
        return bool(self.filter_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_data(ttl=30, show_spinner=False)
def db_status() -> DBStatus:
    return check_database()


def list_experiments(cfg: dict) -> list[dict]:
    return cfg.get("experiments") or [
        {"id": "primary", "label": "Основной эксперимент", "source": "primary"}
    ]


def _load_primary(cfg: dict) -> tuple[pd.DataFrame, str]:
    status = db_status()
    if status.ok:
        try:
            return eda.load_analytics_frame()
        except SQLAlchemyError:
            log.warning("Чтение из PostgreSQL не удалось, используется CSV")
    path = PROJECT_ROOT / cfg["paths"]["raw_csv"]
    if not path.exists():
        raise DataUnavailableError(
            f"Нет данных: {status.message}, и файла {cfg['paths']['raw_csv']} тоже нет."
        )
    return eda.load_csv_frame(path), f"CSV: {cfg['paths']['raw_csv']} ({status.message})"


@st.cache_data(show_spinner="Загрузка данных…")
def load_experiment(exp_id: str) -> tuple[pd.DataFrame, str]:
    cfg = load_config()
    spec = next((e for e in list_experiments(cfg) if e["id"] == exp_id), None)
    if spec is None:
        raise DataUnavailableError(f"Неизвестный эксперимент: {exp_id}")
    if spec["source"] == "primary":
        df, source = _load_primary(cfg)
    else:
        path = PROJECT_ROOT / spec["path"]
        if not path.exists():
            raise DataUnavailableError(
                f"Файл {spec['path']} не найден. Создайте демо-эксперименты: "
                "python scripts/build_experiments.py"
            )
        df, source = eda.load_csv_frame(path), f"CSV: {spec['path']}"
    if df.empty:
        raise DataUnavailableError("Источник данных вернул пустую таблицу.")
    return df, source


def apply_filters(df: pd.DataFrame, key: FilterKey) -> pd.DataFrame:
    """Отбор строк по выбранным значениям признаков (пустой выбор = нет данных)."""
    mask = pd.Series(True, index=df.index)
    for dim, values in key:
        mask &= df[dim].astype(str).isin(values)
    out = df[mask].copy()
    for col in out.select_dtypes("category").columns:
        out[col] = out[col].cat.remove_unused_categories()
    return out


def _check_size(df: pd.DataFrame, min_users: int) -> None:
    if df.empty:
        raise AnalysisUnavailableError(
            "После применения фильтров не осталось ни одного пользователя."
        )
    counts = df["group_name"].astype(str).value_counts()
    for g in ("control", "treatment"):
        n = int(counts.get(g, 0))
        if n < min_users:
            raise AnalysisUnavailableError(
                f"В группе {g} только {n:,} пользователей (минимум для анализа: {min_users:,})."
            )


@st.cache_data(show_spinner="Выполняется статистический анализ…")
def run_cached(exp_id: str, key: FilterKey) -> AdvancedResult:
    """Запускает анализ этапов 5-6. Для подвыборок bootstrap облегчён."""
    cfg = load_config()
    dash = cfg.get("dashboard", {})
    df, source = load_experiment(exp_id)
    if key:
        df = apply_filters(df, key)
        cfg["analysis"]["n_bootstrap"] = int(dash.get("filtered_n_bootstrap", 500))
        cfg["advanced"]["n_bootstrap_conversion"] = int(
            dash.get("filtered_n_bootstrap_conversion", 5000)
        )
        source += " (подвыборка)"
    _check_size(df, int(dash.get("min_group_users", 1000)))
    try:
        return run_advanced(df, cfg, source)
    except Exception as exc:  # граница интерфейса: любая ошибка расчёта становится сообщением
        log.exception("Анализ не выполнен")
        raise AnalysisUnavailableError(
            f"Анализ не удалось выполнить для этих данных ({type(exc).__name__}: {exc})"
        ) from exc


def summarize_filters(key: FilterKey) -> str:
    parts = []
    for dim, values in key:
        label = FILTER_SPECS[dim][0]
        names = [VALUE_LABELS.get(dim, {}).get(v, v) for v in values]
        parts.append(f"{label}: {', '.join(names) or 'ничего'}")
    return "; ".join(parts)


def build_context(exp_id: str, key: FilterKey) -> Context:
    cfg = get_config()
    label = next((e["label"] for e in list_experiments(cfg) if e["id"] == exp_id), exp_id)
    base = dict(
        cfg=cfg,
        experiment_id=exp_id,
        experiment_label=label,
        db=db_status(),
        filter_key=key,
        filter_summary=summarize_filters(key),
    )
    try:
        df_all, source = load_experiment(exp_id)
    except DataUnavailableError as exc:
        return Context(
            source="—",
            df=None,
            result=None,
            official=None,
            error=str(exc),
            error_kind="data",
            **base,
        )
    try:
        official = run_cached(exp_id, ())
    except AnalysisUnavailableError as exc:
        return Context(
            source=source,
            df=df_all,
            result=None,
            official=None,
            error=str(exc),
            error_kind="analysis",
            **base,
        )
    if not key:
        return Context(source=source, df=df_all, result=official, official=official, **base)
    df = apply_filters(df_all, key)
    try:
        result = run_cached(exp_id, key)
    except AnalysisUnavailableError as exc:
        return Context(
            source=source,
            df=df,
            result=None,
            official=official,
            error=str(exc),
            error_kind="analysis",
            **base,
        )
    return Context(source=source, df=df, result=result, official=official, **base)
