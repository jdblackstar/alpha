from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from qlib.validation import validate_positive


def cagr(start_value: float, end_value: float, years: float) -> float:
    """Return compound annual growth rate from start to end value."""
    for name, value in [
        ("start_value", start_value),
        ("end_value", end_value),
        ("years", years),
    ]:
        validate_positive(name, value)
    return (end_value / start_value) ** (1.0 / years) - 1.0


def trailing_cagr(values: pd.Series, *, years: float) -> pd.Series:
    """Return trailing CAGR at each timestamp using a calendar lookback."""
    validate_positive("years", years)
    clean = _clean_values(values)
    output = pd.Series(float("nan"), index=clean.index, name=f"trailing_{years:g}y_cagr")

    index = clean.index
    lookback_days = round(years * 365.25)
    for pos, timestamp in enumerate(index):
        target = timestamp - pd.Timedelta(days=lookback_days)
        prior_pos = index.searchsorted(target, side="right") - 1
        if prior_pos < 0 or prior_pos >= pos:
            continue
        prior_value = clean.iloc[prior_pos]
        current_value = clean.iloc[pos]
        output.iloc[pos] = cagr(prior_value, current_value, years)
    return output


def _clean_values(values: pd.Series) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series of positive values")

    clean = values.dropna()
    if clean.empty:
        return _with_datetime_index(clean.astype(float))

    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.isna().any():
        raise TypeError("values must contain numeric values")

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("values must contain only finite values")

    if (numeric <= 0).any():
        raise ValueError("values must be positive")

    return _with_datetime_index(numeric)


def _with_datetime_index(values: pd.Series) -> pd.Series:
    clean = values.sort_index()
    if not isinstance(clean.index, pd.DatetimeIndex):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            index = pd.to_datetime(clean.index, errors="coerce")
        if index.isna().any():
            raise TypeError("values index must be datetime-like for trailing CAGR")
        clean.index = index
        clean = clean.sort_index()
    return clean
