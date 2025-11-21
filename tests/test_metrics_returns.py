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


def test_sortino_handles_downside() -> None:
    rets = _returns()
    downside = rets[rets < 0].std()
    expected = (rets.mean() / downside) * np.sqrt(252)
    result = sortino(rets)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert np.isclose(result, expected)


def test_max_drawdown_returns_minimum() -> None:
    rets = pd.Series([0.1, 0.05, -0.2, 0.01])
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    expected = ((cumulative - peak) / peak).min()
    assert np.isclose(max_drawdown(rets), expected)
