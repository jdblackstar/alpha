from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib.factors.rsi import RSI


def _make_ohlcv(close: list[float]) -> pd.DataFrame:
    """Create minimal OHLCV dataframe from close prices."""
    dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"close": close}, index=dates)


def test_rsi_all_gains_approaches_100() -> None:
    """When price only goes up, RSI should be 100."""
    # Steady upward movement: 100, 101, 102, ..., 120
    close = [100.0 + i for i in range(21)]
    data = _make_ohlcv(close)

    factor = RSI(period=14)
    result = factor.compute(data)

    # After warmup, RSI should be 100 (no losses)
    valid_values = result.dropna()
    assert len(valid_values) > 0
    assert all(valid_values == 100.0)


def test_rsi_all_losses_approaches_0() -> None:
    """When price only goes down, RSI should be 0."""
    # Steady downward movement: 120, 119, 118, ..., 100
    close = [120.0 - i for i in range(21)]
    data = _make_ohlcv(close)

    factor = RSI(period=14)
    result = factor.compute(data)

    # After warmup, RSI should be 0 (no gains)
    valid_values = result.dropna()
    assert len(valid_values) > 0
    assert all(valid_values == 0.0)


def test_rsi_balanced_movement_near_50() -> None:
    """When gains and losses are equal, RSI should be around 50."""
    # Alternating +1, -1 pattern
    close = [100.0]
    for i in range(30):
        delta = 1.0 if i % 2 == 0 else -1.0
        close.append(close[-1] + delta)
    data = _make_ohlcv(close)

    factor = RSI(period=14)
    result = factor.compute(data)

    # RSI should hover around 50 when movement is balanced
    valid_values = result.dropna()
    assert len(valid_values) > 0
    mean_rsi = valid_values.mean()
    assert 45.0 < mean_rsi < 55.0


def test_rsi_bounded_between_0_and_100() -> None:
    """RSI values should always be in [0, 100]."""
    # Random-ish price movements
    np.random.seed(42)
    close = [100.0]
    for _ in range(50):
        close.append(close[-1] * (1 + np.random.uniform(-0.05, 0.05)))
    data = _make_ohlcv(close)

    factor = RSI(period=14)
    result = factor.compute(data)

    valid_values = result.dropna()
    assert all(valid_values >= 0.0)
    assert all(valid_values <= 100.0)


def test_rsi_flat_prices_returns_50() -> None:
    """When price doesn't move, RSI should be 50 (neutral)."""
    close = [100.0] * 20
    data = _make_ohlcv(close)

    factor = RSI(period=14)
    result = factor.compute(data)

    valid_values = result.dropna()
    assert len(valid_values) > 0
    assert all(valid_values == 50.0)


def test_rsi_respects_period_parameter() -> None:
    """Different periods should produce different warmup lengths."""
    close = [100.0 + i * 0.5 for i in range(30)]
    data = _make_ohlcv(close)

    rsi_short = RSI(period=5).compute(data)
    rsi_long = RSI(period=20).compute(data)

    # Shorter period should have fewer NaN values
    assert rsi_short.isna().sum() < rsi_long.isna().sum()


def test_rsi_invalid_period_raises() -> None:
    """Period must be positive."""
    with pytest.raises(ValueError, match="positive"):
        RSI(period=0)

    with pytest.raises(ValueError, match="positive"):
        RSI(period=-5)


def test_rsi_series_has_correct_name() -> None:
    """Output series should be named 'rsi'."""
    data = _make_ohlcv([100.0 + i for i in range(20)])
    factor = RSI(period=14)
    result = factor.compute(data)
    assert result.name == "rsi"

