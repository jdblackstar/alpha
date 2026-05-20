from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib.metrics.growth import cagr, trailing_cagr


def test_cagr_returns_compound_annual_growth_rate() -> None:
    assert np.isclose(cagr(100.0, 121.0, 2.0), 0.10)


def test_cagr_requires_positive_inputs() -> None:
    with pytest.raises(ValueError, match="start_value must be positive"):
        cagr(0.0, 121.0, 2.0)

    with pytest.raises(ValueError, match="end_value must be positive"):
        cagr(100.0, 0.0, 2.0)

    with pytest.raises(ValueError, match="years must be positive"):
        cagr(100.0, 121.0, 0.0)


def test_trailing_cagr_uses_calendar_lookback() -> None:
    values = pd.Series(
        [10.0, 20.0, 40.0],
        index=pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01"]),
    )

    result = trailing_cagr(values, years=1.0)

    assert np.isnan(result.iloc[0])
    assert np.isclose(result.iloc[1], 1.0)
    assert np.isclose(result.iloc[2], 1.0)


def test_trailing_cagr_accepts_datetime_like_index() -> None:
    values = pd.Series(
        [10.0, 20.0],
        index=["2022-01-01", "2023-01-01"],
    )

    result = trailing_cagr(values, years=1.0)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert np.isclose(result.iloc[1], 1.0)


def test_trailing_cagr_empty_after_dropna_returns_empty_float_series() -> None:
    values = pd.Series(
        [np.nan, np.nan],
        index=pd.to_datetime(["2022-01-01", "2023-01-01"]),
    )

    result = trailing_cagr(values, years=1.0)

    assert result.empty
    assert result.dtype == float


def test_trailing_cagr_requires_series_input() -> None:
    with pytest.raises(TypeError, match="values must be a pandas Series"):
        trailing_cagr([10.0, 20.0], years=1.0)  # type: ignore[arg-type]


def test_trailing_cagr_requires_numeric_values() -> None:
    values = pd.Series(
        [10.0, "bad", 20.0],
        index=pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"]),
    )

    with pytest.raises(TypeError, match="numeric values"):
        trailing_cagr(values, years=1.0)


def test_trailing_cagr_rejects_non_finite_values() -> None:
    values = pd.Series(
        [10.0, np.inf, 20.0],
        index=pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"]),
    )

    with pytest.raises(ValueError, match="finite values"):
        trailing_cagr(values, years=1.0)


def test_trailing_cagr_requires_positive_values() -> None:
    values = pd.Series(
        [10.0, 0.0, 20.0],
        index=pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"]),
    )

    with pytest.raises(ValueError, match="values must be positive"):
        trailing_cagr(values, years=1.0)


def test_trailing_cagr_requires_datetime_like_index() -> None:
    values = pd.Series([10.0, 20.0], index=["not-a-date", "also-not-a-date"])

    with pytest.raises(TypeError, match="datetime-like"):
        trailing_cagr(values, years=1.0)


def test_trailing_cagr_requires_positive_years() -> None:
    values = pd.Series(
        [10.0, 20.0],
        index=pd.to_datetime(["2022-01-01", "2023-01-01"]),
    )

    with pytest.raises(ValueError, match="years must be positive"):
        trailing_cagr(values, years=0.0)
