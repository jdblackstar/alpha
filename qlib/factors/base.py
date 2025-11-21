from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """
    Base interface for all factors.

    Every factor consumes OHLCV data and returns a single-column signal that
    aligns with the provided index.
    """

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the factor signal from the provided OHLCV data.
        """
        raise NotImplementedError
