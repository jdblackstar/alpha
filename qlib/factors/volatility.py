from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class Volatility(Factor):
    """
    Rolling volatility of price returns.

    Volatility measures the dispersion of returns over a lookback window.
    Higher volatility indicates larger price swings and greater uncertainty.

    The calculation:
        1. Compute returns: either simple (pct_change) or log returns
        2. Calculate rolling standard deviation over `window` periods

    Traditional interpretation:
        - High volatility: larger expected price moves, higher risk
        - Low volatility: smaller expected price moves, lower risk
        - Rising volatility: uncertainty increasing, often during selloffs
        - Falling volatility: market calming, often during rallies

    Note:
        This is realized (historical) volatility. It measures what happened,
        not what will happen. Consider annualizing by multiplying by
        sqrt(252) for daily data if comparing to implied volatility.
    """

    def __init__(self, window: int = 20, use_log_returns: bool = False) -> None:
        """
        Initialize the Volatility factor.

        Args:
            window: Lookback window for the rolling standard deviation.
                    Default is 20 (roughly one trading month).
            use_log_returns: If True, compute volatility of log returns
                             instead of simple percentage returns.
                             Log returns are additive across time and
                             symmetric for gains/losses.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.use_log_returns = use_log_returns

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute rolling volatility from OHLCV data.

        Args:
            data: DataFrame with at least a 'close' column.

        Returns:
            Series with volatility values (standard deviation of returns).
            The first `window` rows will be NaN due to insufficient history.
        """
        close: pd.Series = data["close"]

        # Compute returns based on the selected method
        if self.use_log_returns:
            # Log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            returns: pd.Series = np.log(close).diff()
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = close.pct_change()

        # Rolling standard deviation of returns
        signal: pd.Series = returns.rolling(self.window, min_periods=self.window).std()

        signal.name = "volatility"
        return signal
