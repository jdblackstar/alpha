from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> float:
    _validate_annualization(annualization)
    _validate_annual_rate("annual_risk_free_rate", annual_risk_free_rate)
    clean = _clean_returns(returns)
    if len(clean) <= 1:
        return float("nan")

    rf_per_period = (1 + annual_risk_free_rate) ** (1 / annualization) - 1
    excess = clean - rf_per_period

    vol = excess.std()
    if np.isclose(vol, 0.0):
        return float("nan")

    return (excess.mean() / vol) * np.sqrt(annualization)


def sortino(
    returns: pd.Series,
    annual_target_return: float = 0.0,
    annualization: int = 252,
) -> float:
    """Return annualized Sortino ratio using downside deviation below annual target."""
    _validate_annualization(annualization)
    _validate_annual_rate("annual_target_return", annual_target_return)
    clean = _clean_returns(returns)
    n_periods = len(clean)
    if n_periods <= 1:
        return float("nan")

    target_per_period = (1 + annual_target_return) ** (1 / annualization) - 1

    excess = clean - target_per_period
    downside = np.minimum(excess, 0.0)

    downside_variance = np.sum(downside**2) / (n_periods - 1)
    downside_deviation = np.sqrt(downside_variance)

    if np.isclose(downside_deviation, 0.0):
        return float("nan")

    return (excess.mean() / downside_deviation) * np.sqrt(annualization)


def max_drawdown(returns: pd.Series) -> float:
    """Return max drawdown as a negative decimal, including initial capital baseline."""
    clean = _clean_returns(returns)
    if len(clean) <= 1:
        return float("nan")

    equity_curve = (1.0 + clean).cumprod()
    equity_curve = pd.concat(
        [pd.Series([1.0], index=[clean.index[0]]), equity_curve]
    )

    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0

    return drawdown.min()


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series of periodic simple returns")

    clean = returns.dropna()
    if clean.empty:
        return clean.astype(float)

    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.isna().any():
        raise TypeError("returns must contain numeric return values")

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("returns must contain only finite return values")

    return numeric


def _validate_annualization(annualization: int) -> None:
    if annualization <= 0:
        raise ValueError("annualization must be positive")


def _validate_annual_rate(name: str, value: float) -> None:
    if value <= -1.0:
        raise ValueError(f"{name} must be greater than -1.0")
