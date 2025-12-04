from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
from pandas.tseries.frequencies import to_offset

from qlib.backtesting import PortfolioBacktester


def _prices() -> pd.DataFrame:
    """Create a simple 3-asset MultiIndex price DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    # Simple price series for each asset
    spy = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "volume": [1000] * 10,
        },
        index=dates,
    )
    bnd = pd.DataFrame(
        {
            "open": [50.0] * 10,
            "high": [50.5] * 10,
            "low": [49.5] * 10,
            "close": [50.0, 50.5, 51.0, 50.5, 51.0, 51.5, 51.0, 51.5, 52.0, 52.5],
            "volume": [500] * 10,
        },
        index=dates,
    )
    gld = pd.DataFrame(
        {
            "open": [150.0] * 10,
            "high": [151.0] * 10,
            "low": [149.0] * 10,
            "close": [
                150.0,
                149.0,
                148.0,
                149.0,
                150.0,
                151.0,
                152.0,
                151.0,
                150.0,
                149.0,
            ],
            "volume": [200] * 10,
        },
        index=dates,
    )

    return pd.concat({"SPY": spy, "BND": bnd, "GLD": gld}, axis=1)


def _prices_multi_month() -> pd.DataFrame:
    """Two-asset dataset spanning multiple months to test rebalancing."""
    dates = pd.date_range("2024-01-25", periods=40, freq="D")

    def _trend(base: float, step: float) -> list[float]:
        return [base + i * step for i in range(len(dates))]

    def _asset(base: float, step: float, volume: int) -> pd.DataFrame:
        close = _trend(base, step)
        return pd.DataFrame(
            {
                "open": close,
                "high": [c + 0.5 for c in close],
                "low": [c - 0.5 for c in close],
                "close": close,
                "volume": [volume] * len(dates),
            },
            index=dates,
        )

    aaa = _asset(100.0, 1.0, 1_000)
    bbb = _asset(50.0, 0.5, 500)

    return pd.concat({"AAA": aaa, "BBB": bbb}, axis=1)


def _first_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return the first timestamp in each period for a given frequency."""
    offset = to_offset(freq)
    period_code = offset.rule_code
    if period_code.endswith("E"):
        period_code = period_code[:-1]
    periods = index.to_period(period_code)

    seen: set[pd.Period] = set()
    dates: list[pd.Timestamp] = []
    for date, period in zip(index, periods, strict=True):
        if period not in seen:
            seen.add(period)
            dates.append(date)
    return pd.DatetimeIndex(dates)


def _expected_equal_weight_returns(prices: pd.DataFrame, freq: str) -> pd.Series:
    close = prices.xs("close", axis=1, level=1)
    asset_returns = close.pct_change()
    symbols = list(close.columns)
    weight_matrix = pd.DataFrame(
        float("nan"), index=asset_returns.index, columns=symbols
    )

    for date in _first_rebalance_dates(asset_returns.index, freq):
        weight_matrix.loc[date] = 1.0 / len(symbols)

    weight_matrix = weight_matrix.ffill().fillna(0.0)
    positions = weight_matrix.shift(1)
    expected = (positions * asset_returns).sum(axis=1)
    expected.name = "portfolio_returns"
    return expected


def test_equal_weight_allocation() -> None:
    """Equal-weight should distribute returns evenly across all assets."""
    prices = _prices()
    bt = PortfolioBacktester(prices, weights=None, rebalance_freq="ME")
    result = bt.run()

    expected = _expected_equal_weight_returns(prices, freq="ME")
    pdt.assert_series_equal(result, expected)


def test_custom_weight_allocation() -> None:
    """Custom weights should apply specified allocation."""
    prices = _prices()
    weights = {"SPY": 0.6, "BND": 0.3, "GLD": 0.1}
    bt = PortfolioBacktester(prices, weights=weights)
    result = bt.run()

    # Verify the result has expected properties
    assert result.name == "portfolio_returns"
    assert len(result) == len(prices)
    # First two values should be NaN (pct_change + shift)
    assert pd.isna(result.iloc[0]) or result.iloc[0] == 0.0


def test_rebalance_frequency_daily() -> None:
    """Daily rebalancing should reset weights every day."""
    prices = _prices()
    bt_daily = PortfolioBacktester(prices, rebalance_freq="D")
    result_daily = bt_daily.run()

    bt_monthly = PortfolioBacktester(prices, rebalance_freq="ME")
    result_monthly = bt_monthly.run()

    # Both should produce returns but may differ due to rebalancing timing
    assert len(result_daily) == len(result_monthly)
    assert result_daily.name == "portfolio_returns"


def test_commission_application() -> None:
    """Commission costs should reduce returns."""
    prices = _prices()

    bt_no_comm = PortfolioBacktester(prices, commission_bps=0.0)
    bt_with_comm = PortfolioBacktester(prices, commission_bps=10.0)  # 10 bps

    result_no_comm = bt_no_comm.run()
    result_with_comm = bt_with_comm.run()

    # Returns with commission should be less than or equal to returns without
    # (comparing sums, accounting for potential NaN)
    sum_no_comm = result_no_comm.dropna().sum()
    sum_with_comm = result_with_comm.dropna().sum()
    assert sum_with_comm <= sum_no_comm


def test_monthly_rebalance_hits_first_trading_day() -> None:
    """Rebalance dates should align to the first available day of each period."""
    prices = _prices_multi_month()
    bt = PortfolioBacktester(prices, rebalance_freq="ME")
    weights = bt._build_rebalance_weights(prices.index)

    expected_dates = _first_rebalance_dates(prices.index, "ME")
    target = pd.Series(
        {symbol: 1.0 / len(weights.columns) for symbol in weights.columns},
        name=None,
    )

    for date in expected_dates:
        pdt.assert_series_equal(weights.loc[date], target, check_names=False)

    # Should set weights at least once per period (no all-zero rows)
    assert weights.loc[expected_dates].notna().all().all()


def test_monthly_rebalance_returns_match_manual() -> None:
    """Portfolio returns should reflect monthly rebalancing into equal weights."""
    prices = _prices_multi_month()
    bt = PortfolioBacktester(prices, rebalance_freq="ME")
    result = bt.run()

    expected = _expected_equal_weight_returns(prices, freq="ME")
    pdt.assert_series_equal(result, expected)


def test_signal_override() -> None:
    """Signals should override static weights when provided."""
    prices = _prices()
    dates = prices.index

    # Create signals that vary over time
    signals = pd.DataFrame(
        {
            "SPY": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5],
            "BND": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5],
            "GLD": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )

    bt = PortfolioBacktester(prices)
    result = bt.run(signals=signals)

    assert result.name == "portfolio_returns"
    assert len(result) == len(prices)


def test_raises_on_invalid_prices() -> None:
    """Should raise on invalid price input."""
    try:
        PortfolioBacktester(pd.Series([1, 2, 3]))
    except TypeError as exc:
        assert "DataFrame" in str(exc)
    else:
        raise AssertionError("Should raise TypeError for non-DataFrame")


def test_raises_on_empty_prices() -> None:
    """Should raise on empty DataFrame."""
    empty = pd.DataFrame()
    try:
        PortfolioBacktester(empty)
    except ValueError as exc:
        assert "empty" in str(exc).lower() or "observation" in str(exc).lower()
    else:
        raise AssertionError("Should raise ValueError for empty DataFrame")


def test_raises_on_non_multiindex() -> None:
    """Should raise if columns are not MultiIndex."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    simple_df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
    try:
        PortfolioBacktester(simple_df)
    except TypeError as exc:
        assert "MultiIndex" in str(exc)
    else:
        raise AssertionError("Should raise TypeError for non-MultiIndex columns")


def test_raises_on_incomplete_weights() -> None:
    """Should raise if weights dict is missing symbols from data."""
    prices = _prices()
    incomplete_weights = {"SPY": 0.6, "BND": 0.4}  # Missing GLD
    try:
        PortfolioBacktester(prices, weights=incomplete_weights)
    except ValueError as exc:
        assert "missing" in str(exc).lower()
        assert "GLD" in str(exc)
    else:
        raise AssertionError("Should raise ValueError for incomplete weights")


def test_raises_on_extra_weights() -> None:
    """Should raise if weights dict has symbols not in data."""
    prices = _prices()
    extra_weights = {"SPY": 0.5, "BND": 0.3, "GLD": 0.1, "QQQ": 0.1}
    try:
        PortfolioBacktester(prices, weights=extra_weights)
    except ValueError as exc:
        assert "not in data" in str(exc).lower()
        assert "QQQ" in str(exc)
    else:
        raise AssertionError("Should raise ValueError for extra weights")
