"""Пустые данные, NULL, неверные входы, сломанная рандомизация."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from experimentlab.analysis.advanced_engine import run_advanced
from experimentlab.analysis.conversion import srm_check, two_proportion_ztest
from experimentlab.analysis.descriptive import describe_series
from experimentlab.analysis.engine import run_analysis
from experimentlab.app import services
from experimentlab.check_report import Check, print_check_report
from experimentlab.config import load_config
from experimentlab.data.generator import generate_users
from experimentlab.db.validation import NOT_NULL_COLUMNS, validate
from experimentlab.logging_setup import resolve_log_level


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    cfg = load_config()
    cfg["n_users"] = 2000
    df, _ = generate_users(cfg)
    path = tmp_path_factory.mktemp("edge") / "users.csv"
    df.to_csv(path, index=False)
    return pd.read_csv(path)


# ---------- валидация данных, пустые данные и NULL
def test_validate_empty_inputs(raw):
    assert validate(raw.iloc[0:0]) == ["датасет пуст"]
    assert validate(pd.DataFrame())[0].startswith("нет столбцов")


@pytest.mark.parametrize("column", NOT_NULL_COLUMNS)
def test_null_in_every_required_column_is_reported(raw, column):
    df = raw.copy()
    df[column] = df[column].astype(object)
    df.loc[0, column] = None
    assert any("NULL" in e for e in validate(df))


def test_missing_purchase_time_for_buyer_is_inconsistent(raw):
    df = raw.copy()
    df.loc[df.index[df["converted"] == 1][0], "purchase_ts"] = None
    assert any("согласован" in e for e in validate(df))


def test_analysis_rejects_empty_frame(adv_df, adv_cfg):
    with pytest.raises(ValueError, match="пуста"):
        run_analysis(adv_df.iloc[0:0], adv_cfg)


def test_analysis_rejects_null_conversion_instead_of_silently_dropping_it(adv_df, adv_cfg):
    df = adv_df.copy()
    df["converted"] = df["converted"].astype(float)
    df.loc[df.index[0], "converted"] = np.nan
    with pytest.raises(ValueError, match="NULL"):
        run_analysis(df, adv_cfg)


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("group_name", "variant_c", "групп"),
        ("revenue", -1.0, "отрицательн"),
        ("converted", 2, "0 и 1"),
    ],
)
def test_analysis_rejects_invalid_values(adv_df, adv_cfg, column, value, message):
    df = adv_df.copy()
    df.loc[df.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        run_analysis(df, adv_cfg)


def test_analysis_rejects_missing_column(adv_df, adv_cfg):
    with pytest.raises(ValueError, match="нет столбцов"):
        run_analysis(adv_df.drop(columns=["revenue"]), adv_cfg)


def test_statistics_reject_degenerate_samples():
    with pytest.raises(ValueError):
        describe_series([None, np.nan])
    with pytest.raises(ValueError):
        two_proportion_ztest(0, 100, 0, 100)


# ---------- некорректное распределение групп (SRM)
def test_srm_input_validation_and_extreme_split():
    with pytest.raises(ValueError):
        srm_check(0, 0)
    with pytest.raises(ValueError):
        srm_check(10, 10, expected_treatment_share=1.5)
    assert not srm_check(90_000, 10_000).passed


@pytest.mark.slow
def test_broken_randomization_invalidates_experiment(adv_df, adv_cfg):
    rng = np.random.default_rng(0)
    lost = adv_df.index[(adv_df["group_name"] == "treatment") & (rng.random(len(adv_df)) < 0.3)]
    result = run_advanced(adv_df.drop(lost), adv_cfg, source="test")
    assert not result.srm.passed
    assert result.decision.code == "INVALID_EXPERIMENT"
    assert "НЕЛЬЗЯ" in result.srm.warning


# ---------- независимая сверка математики
def test_pooled_ztest_squared_equals_pearson_chi2():
    r = two_proportion_ztest(120, 1000, 150, 1100)
    chi2 = chi2_contingency([[120, 880], [150, 950]], correction=False)[0]
    assert r.z**2 == pytest.approx(chi2, rel=1e-9)


# ---------- прочее
def test_unknown_experiment_gives_domain_error():
    with pytest.raises(services.DataUnavailableError):
        services.load_experiment("no_such_experiment")


def test_check_report_formatting_and_empty_is_failure(capsys):
    assert print_check_report([Check("a", True), Check("b", True, "ok")], "Report")
    assert "Report PASSED: 2/2 checks ok" in capsys.readouterr().out
    assert not print_check_report([], "Report")
    assert "FAILED" in capsys.readouterr().out


def test_log_level_resolution():
    assert resolve_log_level("debug") == 10
    assert resolve_log_level("nonsense") == 20
    assert resolve_log_level(None) == 20
