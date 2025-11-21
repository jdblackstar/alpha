from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from qlib.backtesting import Backtester


def _prices() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.Series([100.0, 101.0, 102.0, 99.0, 100.0], index=dates)


def test_backtester_matches_manual_returns() -> None:
    prices = _prices()
    signal = pd.Series([0.0, 0.5, 1.0, -0.5, 0.0], index=prices.index)
    bt = Backtester(prices)
    result = bt.run(signal)

    expected = signal.shift(1).clip(-1, 1) * prices.pct_change()
    expected.name = "strategy_returns"
    pdt.assert_series_equal(result, expected)


def test_backtester_applies_commission_costs() -> None:
    prices = _prices()
    signal = pd.Series([0.0, 1.0, -1.0, 1.0, -1.0], index=prices.index)
    bt = Backtester(prices, commission_bps=10.0)  # 0.10%
    result = bt.run(signal)

    raw = signal.shift(1).clip(-1, 1) * prices.pct_change()
    turnover = signal.shift(1).clip(-1, 1).diff().abs().fillna(0.0)
    expected = raw - turnover * 0.001  # 10 bps
    expected.name = "strategy_returns"
    pdt.assert_series_equal(result, expected)


def test_backtester_raises_when_no_overlap() -> None:
    prices = _prices()
    signal = pd.Series([1.0, 1.0], index=pd.date_range("2023-12-01", periods=2, freq="D"))
    bt = Backtester(prices)

    try:
        bt.run(signal)
    except ValueError as exc:
        assert "timestamps" in str(exc)
    else:
        raise AssertionError("Backtester should raise when signal and prices do not overlap")
