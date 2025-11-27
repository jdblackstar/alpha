from __future__ import annotations

from typing import NamedTuple

import pandas as pd


class MACDResult(NamedTuple):
    """
    Container for MACD indicator components.

    Attributes:
        line: The MACD line (fast EMA minus slow EMA). Positive values
              indicate upward momentum; negative values indicate downward.
        signal: The signal line (EMA of the MACD line). Used to generate
                buy/sell signals when the MACD line crosses it.
        histogram: The difference between MACD line and signal line.
                   Shows the momentum of the MACD itself — early warning
                   of trend changes.

    Example:
        result = macd.compute(data)

        # Attribute access
        result.line.plot()
        result.histogram.plot(kind='bar')

        # Tuple unpacking
        line, signal, hist = result

        # Convert to DataFrame
        df = result.to_frame()
    """

    line: pd.Series
    signal: pd.Series
    histogram: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """
        Convert the MACD components to a DataFrame.

        Returns:
            DataFrame with columns 'macd', 'signal', and 'histogram',
            indexed by the original datetime index.
        """
        return pd.DataFrame(
            {
                "macd": self.line,
                "signal": self.signal,
                "histogram": self.histogram,
            }
        )


class MACD:
    """
    Moving Average Convergence Divergence indicator.

    MACD measures momentum by comparing two exponential moving averages
    of price. It produces three related signals:

    1. **MACD Line**: Fast EMA minus Slow EMA
       - Positive = upward momentum
       - Negative = downward momentum

    2. **Signal Line**: EMA of the MACD Line
       - Smooths the MACD to reduce noise
       - Crossovers generate trading signals

    3. **Histogram**: MACD Line minus Signal Line
       - Shows rate of change of momentum
       - Shrinking histogram = momentum fading (early reversal warning)

    The calculation:
        fast_ema = EMA(close, fast)
        slow_ema = EMA(close, slow)
        macd_line = fast_ema - slow_ema
        signal_line = EMA(macd_line, signal_period)
        histogram = macd_line - signal_line

    Traditional parameters (Appel's original):
        - fast: 12 periods
        - slow: 26 periods
        - signal: 9 periods

    Note:
        This is NOT a Factor subclass because it returns three series
        instead of one. Use MACDResult.histogram if you need a single
        signal for backtesting.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> None:
        """
        Initialize the MACD indicator.

        Args:
            fast: Lookback for the fast (short-term) EMA. Default is 12.
            slow: Lookback for the slow (long-term) EMA. Default is 26.
            signal_period: Lookback for the signal line EMA. Default is 9.
        """
        if fast <= 0 or slow <= 0 or signal_period <= 0:
            raise ValueError("all periods must be positive integers")
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")

        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def compute(self, data: pd.DataFrame) -> MACDResult:
        """
        Compute MACD components from OHLCV data.

        Args:
            data: DataFrame with at least a 'close' column.

        Returns:
            MACDResult containing line, signal, and histogram series.
            The MACD line will have NaN for the first `slow - 1` rows.
            The signal and histogram will have NaN for the first
            `slow + signal_period - 2` rows due to the EMA warmup period.
        """
        close: pd.Series = data["close"]

        # Exponential moving averages of price
        # Using adjust=False for consistency with standard MACD implementations
        # min_periods ensures we don't compute from insufficient data
        fast_ema = close.ewm(span=self.fast, min_periods=self.fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow, min_periods=self.slow, adjust=False).mean()

        # MACD line: difference between fast and slow EMAs
        # Will be NaN until both EMAs have enough data (slow - 1 rows)
        line = fast_ema - slow_ema
        line.name = "macd"

        # Signal line: EMA of the MACD line
        signal = line.ewm(
            span=self.signal_period, min_periods=self.signal_period, adjust=False
        ).mean()
        signal.name = "signal"

        # Histogram: difference between MACD and signal
        histogram = line - signal
        histogram.name = "histogram"

        return MACDResult(line=line, signal=signal, histogram=histogram)
