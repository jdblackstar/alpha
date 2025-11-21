from __future__ import annotations

import pandas as pd

from .base import Factor


class SMACrossover(Factor):
    """Simple moving average crossover on the close price."""

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow windows must be positive integers")
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Return the fast MA minus the slow MA."""
        close: pd.Series = data["close"]
        fast_ma: pd.Series = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma: pd.Series = close.rolling(self.slow, min_periods=self.slow).mean()
        signal: pd.Series = fast_ma - slow_ma
        signal.name = "sma_crossover"
        return signal
