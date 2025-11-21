from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(rets: pd.Series, annualization: int = 252) -> float:
    """Return the annualized Sharpe ratio."""
    if rets.empty:
        return float("nan")
    vol = rets.std()
    if vol == 0:
        return float("nan")
    return (rets.mean() / vol) * np.sqrt(annualization)


def sortino(rets: pd.Series, annualization: int = 252) -> float:
    """Return the annualized Sortino ratio."""
    if rets.empty:
        return float("nan")
    downside = rets[rets < 0].std()
    if downside == 0:
        return float("nan")
    return (rets.mean() / downside) * np.sqrt(annualization)


def max_drawdown(rets: pd.Series) -> float:
    """Return the maximum drawdown of the cumulative return curve."""
    if rets.empty:
        return float("nan")
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
