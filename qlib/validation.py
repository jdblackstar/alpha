from __future__ import annotations

from numbers import Real

import numpy as np


def validate_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def validate_positive(name: str, value: float) -> None:
    validate_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_negative(name: str, value: float) -> None:
    validate_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
