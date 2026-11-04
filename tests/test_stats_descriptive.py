import pandas as pd
import pytest

from experimentlab.analysis.descriptive import (
    describe_by_group,
    describe_series,
    distribution_diagnostics,
)


def test_describe_series_known_values():
    r = describe_series([1, 2, 2, 3, 10])
    assert r["mean"] == pytest.approx(3.6)
    assert r["median"] == 2
    assert r["mode"] == 2
    assert r["variance"] == pytest.approx(13.3)
    assert r["std"] == pytest.approx(13.3**0.5)
    assert (r["min"], r["max"]) == (1, 10)
    assert r["iqr"] == pytest.approx(1.0)
    assert r["p25"] == 2 and r["p75"] == 3


def test_empty_sample_raises():
    with pytest.raises(ValueError):
        describe_series([])


def test_describe_by_group_shape():
    df = pd.DataFrame({"group_name": ["control"] * 3 + ["treatment"] * 3,
                       "x": [1, 2, 3, 10, 20, 30]})
    out = describe_by_group(df, ["x"])
    assert out.loc[("x", "control"), "mean"] == 2
    assert out.loc[("x", "treatment"), "mean"] == 20


def test_distribution_diagnostics_zero_share_and_skew():
    d = distribution_diagnostics([0] * 90 + [1] * 5 + [50] * 5)
    assert d["zero_share"] == pytest.approx(0.9)
    assert d["skewness"] > 1
