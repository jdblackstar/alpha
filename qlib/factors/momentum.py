from __future__ import annotations

import pandas as pd

from .base import Factor


class Momentum(Factor):
    """
    Simple momentum factor based on close-to-close returns.

    Momentum measures the percentage change in price over a lookback window.
    It captures the tendency for assets that have performed well recently
    to continue performing well (and vice versa).

    The calculation:
        momentum = (close[t] - close[t - lookback]) / close[t - lookback]

    This is equivalent to `close.pct_change(lookback)`.

    Traditional interpretation:
        - Positive momentum: price has been rising, trend may continue
        - Negative momentum: price has been falling, trend may continue
        - Near zero: price is flat or reversing
    """

    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize the Momentum factor.

        Args:
            lookback: Number of periods to measure price change over.
                      Default is 20 (roughly one trading month).
        """
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        self.lookback = lookback

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute momentum from OHLCV data.

        Args:
            data: DataFrame with at least a 'close' column.

        Returns:
            Series with momentum values (percentage change). The first
            `lookback` rows will be NaN due to insufficient history.
        """
        close: pd.Series = data["close"]

        # Percentage change over the lookback window
        signal: pd.Series = close.pct_change(self.lookback)

        signal.name = "momentum"
        return signal
