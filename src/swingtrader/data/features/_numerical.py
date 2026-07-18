"""Numerical helpers for feature calculations."""

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and replace nonfinite or zero-denominator results with NA."""
    if not isinstance(numerator, pd.Series) or not isinstance(denominator, pd.Series):
        raise TypeError("Both numerator and denominator must be pandas Series objects.")

    valid_denominator = denominator.ne(0) & np.isfinite(denominator)
    result = numerator.div(denominator.where(valid_denominator))
    return result.where(np.isfinite(result))


def exponential_moving_average(values: pd.Series, *, length: int) -> pd.Series:
    """Calculate an exponential moving average over one ordered series.

    Uses pandas ``ewm`` with ``span=length``, ``adjust=False``, and
    ``min_periods=length`` so the first ``length - 1`` observations remain
    missing until the window is full.
    """
    return values.ewm(span=length, adjust=False, min_periods=length).mean()
