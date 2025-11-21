from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class Volatility(Factor):
    """Rolling volatility of price returns."""

    def __init__(self, window: int = 20, use_log_returns: bool = False) -> None:
        if window <= 0:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.use_log_returns = use_log_returns

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Return the rolling standard deviation of returns."""
        close: pd.Series = data["close"]
        if self.use_log_returns:
            returns: pd.Series = np.log(close).diff()
        else:
            returns = close.pct_change()
        signal: pd.Series = returns.rolling(self.window, min_periods=self.window).std()
        signal.name = "volatility"
        return signal
