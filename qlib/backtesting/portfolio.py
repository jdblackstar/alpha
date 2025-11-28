from __future__ import annotations

from typing import Optional

import pandas as pd


class PortfolioBacktester:
    """Multi-asset portfolio backtester with rebalancing support."""

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: Optional[dict[str, float]] = None,
        rebalance_freq: str = "ME",
        commission_bps: float = 0.0,
    ) -> None:
        """
        Initialize the portfolio backtester.

        Args:
            prices: DataFrame with MultiIndex columns (symbol, field).
                    Must include "close" prices for each symbol.
            weights: Target allocation weights per symbol. If None, uses
                     equal-weight across all symbols. Weights should sum to 1.
            rebalance_freq: Rebalancing frequency. "D" daily, "W" weekly,
                            "ME" monthly, "QE" quarterly.
            commission_bps: Commission cost in basis points applied to turnover.
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if prices.empty:
            raise ValueError("prices must contain at least one observation")
        if not isinstance(prices.columns, pd.MultiIndex):
            raise TypeError("prices must have MultiIndex columns (symbol, field)")
        if commission_bps < 0:
            raise ValueError("commission_bps cannot be negative")

        self.prices = prices.sort_index()
        self.rebalance_freq = rebalance_freq
        self.commission_bps = float(commission_bps)

        # Extract symbols from the first level of the column MultiIndex
        self._symbols = list(prices.columns.get_level_values(0).unique())

        # Set up weights
        if weights is None:
            # Equal-weight allocation
            n = len(self._symbols)
            self._weights = {s: 1.0 / n for s in self._symbols}
        else:
            self._weights = weights

    def run(self, signals: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Run the backtest and return portfolio daily returns.

        Args:
            signals: Optional DataFrame of per-asset signals to override static
                     weights. Columns should match symbols. Values are used as
                     weights (normalized each row).

        Returns:
            Series of daily portfolio returns.
        """
        # Extract close prices for each symbol
        close_prices = self.prices.xs("close", axis=1, level=1)
        asset_returns = close_prices.pct_change()

        # Build the weight matrix
        if signals is not None:
            # Use signals as weights, normalized per row
            weight_matrix = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
            weight_matrix = weight_matrix.reindex(asset_returns.index).ffill().fillna(0)
        else:
            weight_matrix = self._build_rebalance_weights(asset_returns.index)

        # Shift weights by 1 day (trade on signal, settle next day)
        positions = weight_matrix.shift(1)

        # Compute portfolio returns
        portfolio_returns = (positions * asset_returns).sum(axis=1)

        # Apply commission costs on turnover
        if self.commission_bps:
            turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
            costs = turnover * (self.commission_bps / 10_000.0)
            portfolio_returns = portfolio_returns - costs

        portfolio_returns.name = "portfolio_returns"
        return portfolio_returns

    def _build_rebalance_weights(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build a weight matrix that rebalances at the specified frequency."""
        # Create a series of target weights
        target = pd.Series(self._weights)

        # Initialize weight matrix with zeros
        weight_matrix = pd.DataFrame(0.0, index=index, columns=self._symbols)

        # Find rebalance dates
        rebalance_dates = (
            index.to_series().resample(self.rebalance_freq).first().dropna()
        )

        # Set target weights on rebalance dates
        for date in rebalance_dates.values:
            if date in weight_matrix.index:
                weight_matrix.loc[date] = target

        # Forward-fill weights between rebalance dates
        weight_matrix = weight_matrix.replace(0.0, float("nan"))
        weight_matrix = weight_matrix.ffill().fillna(0.0)

        return weight_matrix
