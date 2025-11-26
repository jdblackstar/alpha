from __future__ import annotations

import pandas as pd

from .base import Factor


class SMACrossover(Factor):
    """
    Simple moving average crossover signal on the close price.

    This factor computes the difference between a fast (short-term) and
    slow (long-term) simple moving average. When the fast MA is above
    the slow MA, the signal is positive — indicating upward momentum.

    The calculation:
        1. fast_ma = rolling mean of close over `fast` periods
        2. slow_ma = rolling mean of close over `slow` periods
        3. signal = fast_ma - slow_ma

    Traditional interpretation:
        - Positive signal: fast MA > slow MA, bullish trend
        - Negative signal: fast MA < slow MA, bearish trend
        - Crossover from negative to positive: buy signal
        - Crossover from positive to negative: sell signal

    Note:
        The raw difference can be large for high-priced stocks. Consider
        normalizing by price or slow_ma if comparing across assets.
    """

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        """
        Initialize the SMA Crossover factor.

        Args:
            fast: Lookback window for the fast (short-term) moving average.
                  Default is 10 periods.
            slow: Lookback window for the slow (long-term) moving average.
                  Default is 30 periods.
        """
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow windows must be positive integers")
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the SMA crossover signal from OHLCV data.

        Args:
            data: DataFrame with at least a 'close' column.

        Returns:
            Series with the difference (fast_ma - slow_ma). The first
            `slow` rows will be NaN due to insufficient history.
        """
        close: pd.Series = data["close"]

        # Simple moving averages with min_periods to avoid partial windows
        fast_ma: pd.Series = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma: pd.Series = close.rolling(self.slow, min_periods=self.slow).mean()

        # Positive when fast is above slow (bullish), negative when below
        signal: pd.Series = fast_ma - slow_ma

        signal.name = "sma_crossover"
        return signal
