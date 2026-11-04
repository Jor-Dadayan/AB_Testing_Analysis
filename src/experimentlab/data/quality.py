"""Проверки качества сгенерированного датасета."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from experimentlab.data import generator as g


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def run_quality_checks(df: pd.DataFrame, cfg: dict) -> list[Check]:
    """Возвращает список проверок. Истинный эффект здесь не используется."""
    d = df.copy()
    for c in ("signup_ts", "purchase_ts", "signup_date"):
        d[c] = pd.to_datetime(d[c])
    res: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        res.append(Check(name, bool(ok), detail))

    n = len(d)
    add("число строк равно n_users", n == cfg["n_users"], str(n))
    add("user_id уникален", d["user_id"].is_unique)
    required = [c for c in d.columns if c != "purchase_ts"]
    add("нет пропусков (кроме purchase_ts)", d[required].isna().sum().sum() == 0)
    add("группы: только control/treatment", set(d["group"]) == {"control", "treatment"})

    # Рандомизация
    n_t = (d["group"] == "treatment").sum()
    exp_t = n * cfg["treatment_share"]
    p_srm = stats.chisquare([n - n_t, n_t], [n - exp_t, exp_t]).pvalue
    add("доли групп соответствуют плану (SRM)", p_srm > 0.001, f"p={p_srm:.3f}")
    for col in ("device_type", "traffic_source", "country", "is_new_user"):
        p_bal = stats.chi2_contingency(pd.crosstab(d["group"], d[col]))[1]
        add(f"баланс групп по {col}", p_bal > 0.001, f"p={p_bal:.3f}")

    # Допустимые значения
    add("device_type в допустимом наборе", set(d["device_type"]) <= set(g.DEVICES))
    add("country в допустимом наборе", set(d["country"]) <= set(g.COUNTRIES))
    add("traffic_source в допустимом наборе", set(d["traffic_source"]) <= set(g.SOURCES))
    add("converted равен 0/1", set(d["converted"]) <= {0, 1})
    add("age в диапазоне 18-75", d["age"].between(18, 75).all())
    add("signup_hour в диапазоне 0-23", d["signup_hour"].between(0, 23).all())

    # Время
    start = pd.Timestamp(cfg["start_date"])
    end = start + pd.Timedelta(days=cfg["n_days"])
    add("даты регистрации в периоде теста", d["signup_ts"].between(start, end).all())
    buyers = d["converted"] == 1
    add("purchase_ts заполнен только у купивших", (d["purchase_ts"].notna() == buyers).all())
    add(
        "покупка не раньше регистрации",
        (d.loc[buyers, "purchase_ts"] >= d.loc[buyers, "signup_ts"]).all(),
    )

    # Выручка
    add("выручка неотрицательна", (d["revenue"] >= 0).all())
    add("выручка > 0 только у купивших", ((d["revenue"] > 0) == buyers).all())
    zero_share = (d["revenue"] == 0).mean()
    add("доля нулевой выручки не менее 70%", zero_share >= 0.70, f"{zero_share:.1%}")
    pay = d.loc[buyers, "revenue"]
    skew = stats.skew(pay)
    add(
        "выручка покупателей скошена вправо (mean > median)",
        pay.mean() > pay.median() and skew > 0.5,
        f"skew={skew:.2f}",
    )
    add(
        "конверсия в разумных пределах 3-30%",
        0.03 <= d["converted"].mean() <= 0.30,
        f"{d['converted'].mean():.2%}",
    )

    # Логичные корреляции
    cv = d.groupby("device_type")["cart_value"].mean()
    add("корзина desktop дороже, чем mobile", cv["desktop"] > cv["mobile"])
    ses = d.groupby("is_new_user")["sessions_prior_30d"].mean()
    add("у вернувшихся больше прошлых сессий", ses[0] > ses[1])
    cs = d.groupby("traffic_source")["converted"].mean()
    add("конверсия email выше, чем social", cs["email"] > cs["social"])
    add(
        "выручка коррелирует со стоимостью корзины",
        d.loc[buyers, ["revenue", "cart_value"]].corr().iloc[0, 1] > 0.5,
    )
    return res


def print_report(checks: list[Check]) -> bool:
    """Печатает отчёт, возвращает True, если все проверки пройдены."""
    for c in checks:
        mark = "OK  " if c.passed else "FAIL"
        extra = f"  ({c.detail})" if c.detail else ""
        print(f"[{mark}] {c.name}{extra}")
    ok = sum(c.passed for c in checks)
    status = "PASSED" if ok == len(checks) else "FAILED"
    print(f"\nQuality report {status}: {ok}/{len(checks)} checks ok")
    return ok == len(checks)
