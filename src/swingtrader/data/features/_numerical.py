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


def consecutive_true_count(condition: pd.Series) -> pd.Series:
    """Count consecutive periods for which a condition is true.

    The count starts at one when the condition becomes true, increments
    while it remains true, and resets to zero when it becomes false.
    Missing values break the current run and remain missing in the result.

    Parameters
    ----------
    condition
        Boolean condition ordered chronologically.

    Returns
    -------
    pd.Series
        Nullable integer series containing the consecutive true count.

    Examples
    --------
    A condition of::

        <NA>, False, True, True, True, False, True

    produces::

        <NA>, 0, 1, 2, 3, 0, 1
    """
    condition = condition.astype("boolean")
    active = condition.fillna(False)

    # Each False or missing observation begins a new run group.
    run_id = (~active).cumsum()

    counts = active.astype("int64").groupby(run_id).cumsum().astype("Int64")

    return counts.where(condition.notna())


def linreg(
    values: pd.Series,
    *,
    length: int,
    offset: int = 0,
) -> pd.Series:
    """Calculate TradingView-style rolling linear regression.

    Fits an ordinary least-squares line to each rolling window and returns
    the fitted value at position ``length - 1 - offset``.

    Parameters
    ----------
    values
        Input series ordered from oldest to newest.
    length
        Number of observations in each regression window.
    offset
        Offset from the newest observation. An offset of zero evaluates
        the fitted line at the newest position.

    Returns
    -------
    pd.Series
        Rolling fitted values with the same index as ``values``.
    """
    if length < 1:
        raise ValueError("length must be at least 1")

    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denominator = np.dot(x_centered, x_centered)

    evaluation_position = length - 1 - offset

    # For length=1, every regression consists of one constant observation.
    if denominator == 0:
        return values.rolling(
            window=length,
            min_periods=length,
        ).mean()

    # The fitted value can be expressed directly as a weighted sum of y.
    # This avoids running np.polyfit for every rolling window.
    weights = (
        np.full(length, 1.0 / length) + (evaluation_position - x_mean) * x_centered / denominator
    )

    return values.rolling(
        window=length,
        min_periods=length,
    ).apply(
        lambda window: np.dot(window, weights),
        raw=True,
    )
