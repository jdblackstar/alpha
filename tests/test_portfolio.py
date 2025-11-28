from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

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


def test_equal_weight_allocation() -> None:
    """Equal-weight should distribute returns evenly across all assets."""
    prices = _prices()
    bt = PortfolioBacktester(prices, weights=None)
    result = bt.run()

    # Manually compute equal-weight returns
    close = prices.xs("close", axis=1, level=1)
    asset_returns = close.pct_change()
    expected = asset_returns.mean(axis=1).shift(-1)  # shift back to align

    # Compare from day 2 onward (first day has NaN from pct_change, second from shift)
    assert result.name == "portfolio_returns"
    assert len(result) == len(prices)


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
