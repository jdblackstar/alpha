from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.metrics.returns import max_drawdown, sharpe, sortino


def _returns() -> pd.Series:
    return pd.Series([0.01, -0.02, 0.015, 0.0, 0.01])


def test_sharpe_matches_manual() -> None:
    rets = _returns()
    expected = (rets.mean() / rets.std()) * np.sqrt(252)
    assert np.isclose(sharpe(rets), expected)


def test_sortino_uses_downside_deviation() -> None:
    rets = _returns()
    clean = rets.dropna()
    downside = np.minimum(clean, 0.0)
    downside_dev = np.sqrt((downside**2).mean())
    expected = (clean.mean() / downside_dev) * np.sqrt(252)
    assert np.isclose(sortino(rets), expected)


def test_sortino_returns_nan_without_downside() -> None:
    rets = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino(rets))


def test_max_drawdown_returns_minimum() -> None:
    rets = pd.Series([0.1, 0.05, -0.2, 0.01])
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    expected = ((cumulative - peak) / peak).min()
    assert np.isclose(max_drawdown(rets), expected)
