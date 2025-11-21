from __future__ import annotations

import pandas as pd

from .base import Factor


class Momentum(Factor):
    """Simple momentum factor based on close-to-close returns."""

    def __init__(self, lookback: int = 20) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        self.lookback = lookback

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Return the percentage change over the given lookback window."""
        close: pd.Series = data["close"]
        signal: pd.Series = close.pct_change(self.lookback)
        signal.name = "momentum"
        return signal
