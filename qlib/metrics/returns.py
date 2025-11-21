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
    """Return the annualized Sortino ratio assuming a zero target return."""
    clean = rets.dropna()
    if clean.empty:
        return float("nan")
    downside = np.minimum(clean, 0.0)
    downside_variance = (downside**2).mean()
    if downside_variance == 0:
        return float("nan")
    downside_deviation = float(np.sqrt(downside_variance))
    return (clean.mean() / downside_deviation) * np.sqrt(annualization)


def max_drawdown(rets: pd.Series) -> float:
    """Return the maximum drawdown of the cumulative return curve."""
    if rets.empty:
        return float("nan")
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
