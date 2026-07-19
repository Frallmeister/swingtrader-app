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


def wilder_moving_average(values: pd.Series, *, length: int) -> pd.Series:
    """Calculate Wilder's smoothed moving average (RMA) over one ordered series.

    Uses pandas ``ewm`` with ``alpha=1 / length``, ``adjust=False``, and
    ``min_periods=length`` so the first ``length - 1`` observations remain
    missing until the window is full. This is the recursive smoothing Wilder
    defined for indicators such as ATR and RSI, and it decays more slowly than
    :func:`exponential_moving_average` for the same ``length``.

    This recursive form is seeded from the first observation rather than the
    canonical Wilder definition, which seeds the first output with the simple
    average of the first ``length`` observations. The two forms converge quickly
    as more observations accrue, but early values differ slightly from a
    canonical implementation.
    """
    return values.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
