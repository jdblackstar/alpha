from __future__ import annotations

import pandas as pd

from .base import Factor


class RSI(Factor):
    """
    Relative Strength Index using Wilder's exponential smoothing.

    RSI measures the speed and magnitude of recent price changes to identify
    overbought or oversold conditions. Values range from 0 to 100.

    The calculation:
        1. Compute daily price changes
        2. Separate into gains (positive changes) and losses (negative changes)
        3. Smooth both using Wilder's EMA (alpha = 1/period)
        4. RS = avg_gain / avg_loss
        5. RSI = 100 - (100 / (1 + RS))

    Traditional interpretation:
        - RSI > 70: potentially overbought (price may reverse down)
        - RSI < 30: potentially oversold (price may reverse up)
        - RSI ~ 50: neutral momentum
    """

    def __init__(self, period: int = 14) -> None:
        """
        Initialize the RSI factor.

        Args:
            period: Lookback window for Wilder's smoothing. Default is 14,
                    which was Wilder's original recommendation.
        """
        if period <= 0:
            raise ValueError("period must be a positive integer")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute RSI from OHLCV data.

        Args:
            data: DataFrame with at least a 'close' column.

        Returns:
            Series with RSI values in [0, 100]. The first `period` rows will
            be NaN due to insufficient data for the smoothing calculation.
        """
        close: pd.Series = data["close"]
        delta = close.diff()

        # Separate gains and losses (both as positive values)
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing: EMA with alpha = 1/period
        # This gives more weight to recent observations while still
        # incorporating the full history
        alpha = 1.0 / self.period
        avg_gain = gains.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        # RS = Relative Strength = average gain / average loss
        # RSI transforms RS to a 0-100 scale
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Edge case: when price hasn't moved (both avg_gain and avg_loss are 0),
        # RS is undefined. We return 50 to indicate neutral momentum.
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~both_zero, 50.0)

        rsi.name = "rsi"
        return rsi
