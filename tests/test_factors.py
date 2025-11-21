from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from qlib.factors.momentum import Momentum
from qlib.factors.sma import SMACrossover
from qlib.factors.volatility import Volatility


def _sample_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0, 110.0, 108.0, 111.0], index=dates)
    return pd.DataFrame({"close": close})


def test_momentum_pct_change_matches_pandas() -> None:
    data = _sample_data()
    factor = Momentum(lookback=2)
    result = factor.compute(data)
    expected = data["close"].pct_change(2)
    expected.name = "momentum"
    pdt.assert_series_equal(result, expected)


def test_sma_crossover_returns_fast_minus_slow() -> None:
    data = _sample_data()
    factor = SMACrossover(fast=2, slow=4)
    result = factor.compute(data)
    close = data["close"]
    fast = close.rolling(2, min_periods=2).mean()
    slow = close.rolling(4, min_periods=4).mean()
    expected = (fast - slow).rename("sma_crossover")
    pdt.assert_series_equal(result, expected)


def test_volatility_matches_rolling_std() -> None:
    data = _sample_data()
    factor = Volatility(window=3)
    result = factor.compute(data)
    expected = data["close"].pct_change().rolling(3, min_periods=3).std().rename("volatility")
    pdt.assert_series_equal(result, expected)
