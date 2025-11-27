from __future__ import annotations

import pandas as pd
import pytest

from qlib.factors.macd import MACD, MACDResult


def _make_ohlcv(close: list[float]) -> pd.DataFrame:
    """Create minimal OHLCV dataframe from close prices."""
    dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"close": close}, index=dates)


def test_macd_returns_named_tuple() -> None:
    """compute() should return a MACDResult with three components."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)

    assert isinstance(result, MACDResult)
    assert isinstance(result.line, pd.Series)
    assert isinstance(result.signal, pd.Series)
    assert isinstance(result.histogram, pd.Series)


def test_macd_tuple_unpacking() -> None:
    """MACDResult should support tuple unpacking."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    line, signal, histogram = macd.compute(data)

    assert isinstance(line, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(histogram, pd.Series)


def test_macd_to_frame() -> None:
    """to_frame() should return a DataFrame with all components."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)
    df = result.to_frame()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["macd", "signal", "histogram"]
    assert len(df) == len(data)


def test_macd_line_is_fast_minus_slow_ema() -> None:
    """MACD line should equal fast EMA minus slow EMA."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD(fast=12, slow=26, signal_period=9)
    result = macd.compute(data)

    # Manually compute EMAs
    close_series = data["close"]
    fast_ema = close_series.ewm(span=12, adjust=False).mean()
    slow_ema = close_series.ewm(span=26, adjust=False).mean()
    expected_line = fast_ema - slow_ema

    pd.testing.assert_series_equal(
        result.line, expected_line.rename("macd"), check_names=True
    )


def test_macd_signal_is_ema_of_line() -> None:
    """Signal line should be EMA of the MACD line."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD(fast=12, slow=26, signal_period=9)
    result = macd.compute(data)

    # Signal should be EMA of the line
    expected_signal = result.line.ewm(span=9, adjust=False).mean()

    pd.testing.assert_series_equal(
        result.signal, expected_signal.rename("signal"), check_names=True
    )


def test_macd_histogram_is_line_minus_signal() -> None:
    """Histogram should equal MACD line minus signal line."""
    close = [100.0 + i * 0.5 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)

    expected_histogram = result.line - result.signal

    pd.testing.assert_series_equal(
        result.histogram, expected_histogram.rename("histogram"), check_names=True
    )


def test_macd_uptrend_positive_line() -> None:
    """In a steady uptrend, MACD line should be positive."""
    # Strong upward movement
    close = [100.0 + i * 2.0 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)

    # After warmup, MACD line should be positive (fast > slow)
    valid_values = result.line.iloc[30:].dropna()
    assert all(valid_values > 0)


def test_macd_downtrend_negative_line() -> None:
    """In a steady downtrend, MACD line should be negative."""
    # Strong downward movement
    close = [200.0 - i * 2.0 for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)

    # After warmup, MACD line should be negative (fast < slow)
    valid_values = result.line.iloc[30:].dropna()
    assert all(valid_values < 0)


def test_macd_series_have_correct_names() -> None:
    """Output series should have descriptive names."""
    close = [100.0 + i for i in range(50)]
    data = _make_ohlcv(close)

    macd = MACD()
    result = macd.compute(data)

    assert result.line.name == "macd"
    assert result.signal.name == "signal"
    assert result.histogram.name == "histogram"


def test_macd_respects_custom_periods() -> None:
    """Custom periods should produce different results."""
    close = [100.0 + i * 0.5 for i in range(100)]
    data = _make_ohlcv(close)

    macd_default = MACD(fast=12, slow=26, signal_period=9)
    macd_custom = MACD(fast=8, slow=17, signal_period=5)

    result_default = macd_default.compute(data)
    result_custom = macd_custom.compute(data)

    # Results should differ
    assert not result_default.line.equals(result_custom.line)
    assert not result_default.signal.equals(result_custom.signal)


def test_macd_invalid_periods_raises() -> None:
    """Invalid period values should raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        MACD(fast=0, slow=26, signal_period=9)

    with pytest.raises(ValueError, match="positive"):
        MACD(fast=12, slow=-1, signal_period=9)

    with pytest.raises(ValueError, match="positive"):
        MACD(fast=12, slow=26, signal_period=0)


def test_macd_fast_must_be_less_than_slow() -> None:
    """Fast period must be smaller than slow period."""
    with pytest.raises(ValueError, match="smaller"):
        MACD(fast=26, slow=12, signal_period=9)

    with pytest.raises(ValueError, match="smaller"):
        MACD(fast=20, slow=20, signal_period=9)
