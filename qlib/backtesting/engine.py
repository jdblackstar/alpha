from __future__ import annotations

import pandas as pd


class Backtester:
    """Simple single-asset backtester for daily close signals."""

    def __init__(
        self,
        prices: pd.Series,
        *,
        max_position: float = 1.0,
        commission_bps: float = 0.0,
    ) -> None:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if prices.empty:
            raise ValueError("prices must contain at least one observation")
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        if commission_bps < 0:
            raise ValueError("commission_bps cannot be negative")

        self.prices: pd.Series = prices.sort_index()
        self.max_position = float(max_position)
        self.commission_bps = float(commission_bps)

    def run(self, signal: pd.Series) -> pd.Series:
        """Run the backtest and return daily strategy returns."""
        if not isinstance(signal, pd.Series):
            raise TypeError("signal must be a pandas Series")

        signal_aligned, prices_aligned = signal.align(self.prices, join="inner")
        if prices_aligned.empty:
            raise ValueError("signal and prices do not share any timestamps")

        positions = signal_aligned.shift(1).clip(-self.max_position, self.max_position)
        returns = prices_aligned.pct_change()
        pnl = positions * returns

        if self.commission_bps:
            turnover = positions.diff().abs().fillna(0.0)
            costs = turnover * (self.commission_bps / 10_000.0)
            pnl = pnl - costs

        pnl.name = "strategy_returns"
        return pnl
