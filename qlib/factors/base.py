from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """
    Abstract base class for all factors.

    A factor is a quantitative signal computed from market data (typically
    OHLCV) that may have predictive power for future returns. Each factor
    takes a DataFrame of price data and returns a Series of signal values
    aligned with the input's index.

    Subclasses must implement the `compute` method.

    Example:
        class MyFactor(Factor):
            def compute(self, data: pd.DataFrame) -> pd.Series:
                return data["close"].pct_change(5)
    """

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the factor signal from OHLCV data.

        Args:
            data: DataFrame with OHLCV columns (at minimum 'close').
                  Must have a datetime index.

        Returns:
            Series of signal values, indexed to match the input data.
            Early rows may contain NaN if the factor requires a warmup period.
        """
        raise NotImplementedError
