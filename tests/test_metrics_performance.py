from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib.metrics.performance import max_drawdown, sharpe, sortino


def _returns() -> pd.Series:
    return pd.Series([0.01, -0.02, 0.015, 0.0, 0.01])


def test_sharpe_matches_manual() -> None:
    returns = _returns()
    expected = (returns.mean() / returns.std()) * np.sqrt(252)
    assert np.isclose(sharpe(returns), expected)


def test_sortino_uses_downside_deviation() -> None:
    returns = _returns()
    clean = returns.dropna()
    downside = np.minimum(clean, 0.0)
    downside_dev = np.sqrt(np.sum(downside**2) / (len(clean) - 1))
    expected = (clean.mean() / downside_dev) * np.sqrt(252)
    assert np.isclose(sortino(returns), expected)


def test_sortino_returns_nan_without_downside() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino(returns))


def test_max_drawdown_returns_minimum() -> None:
    returns = pd.Series([0.1, 0.05, -0.2, 0.01])
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    expected = ((cumulative - peak) / peak).min()
    assert np.isclose(max_drawdown(returns), expected)


def test_metrics_require_series_input() -> None:
    with pytest.raises(TypeError, match="returns must be a pandas Series"):
        sharpe([0.01, 0.02])  # type: ignore[arg-type]


def test_metrics_require_numeric_returns() -> None:
    returns = pd.Series([0.01, "bad", 0.02])

    with pytest.raises(TypeError, match="numeric return values"):
        sortino(returns)


def test_metrics_reject_non_finite_returns() -> None:
    returns = pd.Series([0.01, np.inf, 0.02])

    with pytest.raises(ValueError, match="finite return values"):
        max_drawdown(returns)


def test_sharpe_requires_positive_annualization() -> None:
    with pytest.raises(ValueError, match="annualization must be positive"):
        sharpe(_returns(), annualization=0)


def test_sharpe_requires_finite_annualization() -> None:
    with pytest.raises(ValueError, match="annualization must be finite"):
        sharpe(_returns(), annualization=np.inf)


def test_sharpe_requires_numeric_annualization() -> None:
    with pytest.raises(TypeError, match="annualization must be numeric"):
        sharpe(_returns(), annualization=True)


def test_sortino_requires_positive_annualization() -> None:
    with pytest.raises(ValueError, match="annualization must be positive"):
        sortino(_returns(), annualization=0)


def test_sharpe_requires_risk_free_rate_above_negative_one() -> None:
    with pytest.raises(ValueError, match="annual_risk_free_rate"):
        sharpe(_returns(), annual_risk_free_rate=-1.0)


def test_sharpe_requires_finite_risk_free_rate() -> None:
    with pytest.raises(ValueError, match="annual_risk_free_rate must be finite"):
        sharpe(_returns(), annual_risk_free_rate=np.nan)


def test_sharpe_requires_numeric_risk_free_rate() -> None:
    with pytest.raises(TypeError, match="annual_risk_free_rate must be numeric"):
        sharpe(_returns(), annual_risk_free_rate=True)


def test_sortino_requires_target_return_above_negative_one() -> None:
    with pytest.raises(ValueError, match="annual_target_return"):
        sortino(_returns(), annual_target_return=-1.0)


def test_sortino_requires_finite_target_return() -> None:
    with pytest.raises(ValueError, match="annual_target_return must be finite"):
        sortino(_returns(), annual_target_return=np.inf)
