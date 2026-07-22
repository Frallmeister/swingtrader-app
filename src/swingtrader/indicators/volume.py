"""Volume-based indicators derived from traded volume and price.

The module provides turnover, turnover normalization, and the Money Flow
Index. Turnover is calculated from adjusted close and volume and may
optionally be transformed with ``log1p``. The turnover z-score compares each
current observation with the median and population standard deviation of a
preceding reference window.

The public indicators accept either one chronologically ordered instrument or
a multi-instrument input carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. Stateful calculations are isolated within each
provider/ticker group so one instrument's history cannot leak into another's.
The returned series preserve the input index and row order, and the input data
is not mutated.
"""

import numpy as np
import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import validate_length


def turnover(data: pd.DataFrame, *, log: bool = False) -> pd.Series:
    """Calculate traded turnover for each observation.

    Turnover is defined as ``adjusted_close * volume``. When ``log`` is true, the
    result is transformed with ``log1p``, equivalent to
    ``log(1 + turnover)``. The transformation reduces the right skew commonly
    present in turnover while keeping zero-turnover observations finite.

    ``data`` must contain ``adjusted_close`` and ``volume`` columns. The
    calculation is row-local, so the function accepts either one instrument or a
    multi-instrument dataframe without requiring grouping.

    Parameters
    ----------
    data
        Market data containing ``adjusted_close`` and ``volume`` columns.
    log
        Whether to apply ``log1p`` to the calculated turnover.

    Returns
    -------
    pandas.Series
        An index-aligned series named ``turnover``.

    Raises
    ------
    ValueError
        If ``log`` is not a boolean.
    KeyError
        If a required input column is missing.
    """
    validate_required_columns(data, required_columns={"adjusted_close", "volume"})
    if not isinstance(log, bool):
        raise ValueError(f"The log parameter must be a boolean; got {log!r}")
    return _turnover(data, log=log)


def turnover_zscore(data: pd.DataFrame, *, length: int = 252, log: bool = False) -> pd.Series:
    """Calculate a point-in-time-safe turnover z-score.

    The score compares the current turnover with the distribution of the
    preceding ``length - 1`` observations. The current observation is excluded
    from the reference window. The calculation is

    ``(current turnover - prior median) / prior population standard deviation``.

    Turnover is ``adjusted_close * volume``. When ``log`` is true, ``log1p`` is
    applied before calculating the rolling statistics. Median centering reduces
    the influence of unusually large historical turnover observations, while the
    standard deviation expresses the deviation relative to historical
    dispersion.

    ``length`` describes the complete span including the current observation, so
    the historical reference window contains ``length - 1`` rows. The first
    ``length - 1`` observations for each instrument remain missing until that
    reference window is full. A zero reference standard deviation also produces
    a missing result through safe division.

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels, the calculation is isolated within each
    provider/ticker group. The original index and row order are preserved.

    Parameters
    ----------
    data
        Chronologically ordered market data containing ``adjusted_close`` and
        ``volume`` columns.
    length
        Total observation span including the current row. Must be at least 2.
    log
        Whether to apply ``log1p`` to turnover before normalization.

    Returns
    -------
    pandas.Series
        An index-aligned series named ``turnover_zscore``.

    Raises
    ------
    ValueError
        If ``length`` is invalid, is less than 2, or if ``log`` is not a boolean.
    KeyError
        If a required input column is missing.
    """
    validate_length(length)
    if length < 2:
        raise ValueError(f"The length parameter must be at least 2; got {length!r}")
    if not isinstance(log, bool):
        raise ValueError(f"The log parameter must be a boolean; got {log!r}")
    validate_required_columns(data, required_columns={"adjusted_close", "volume"})

    return apply_by_ticker(
        data,
        lambda group: _turnover_zscore(group, length=length, log=log),
    )


def mfi(
    data: pd.DataFrame,
    *,
    length: int = 14,
) -> pd.Series:
    """Calculate the Money Flow Index for one or many tickers.

    MFI is a bounded ``[0, 100]`` volume-weighted momentum oscillator, often
    described as a volume-weighted RSI. Each row's typical price is
    ``(high + low + close) / 3`` and its raw money flow is the typical price times
    ``volume``. A row's money flow is positive when its typical price rose from the
    prior row and negative when it fell; a row whose typical price is unchanged
    contributes to neither. MFI is
    ``100 * positive_flow / (positive_flow + negative_flow)`` over the trailing
    ``length`` rows, so a window with no negative flow returns 100 and a window
    with no positive flow returns 0. A window whose typical price never changes has
    neither positive nor negative flow and is left missing.

    ``data`` must contain ``high``, ``low``, ``close``, and ``volume`` columns in
    chronological order, because the oscillator needs the intraday extremes, the
    close, and the traded volume together. When ``data`` carries the canonical
    ``provider``, ``ticker``, and ``trading_date`` index levels the calculation is
    isolated within each group, so one ticker's history cannot leak into
    another's, and the original index and row order are preserved. The first
    ``length`` rows of each series remain missing until the trailing window is
    full, because the first row has no prior typical price to compare against.
    """
    validate_length(length)
    validate_required_columns(data, required_columns={"high", "low", "close", "volume"})
    return apply_by_ticker(data, lambda group: _mfi(group, length=length))


def _turnover(data: pd.DataFrame, *, log: bool) -> pd.Series:
    turnover_ = (data["adjusted_close"] * data["volume"]).rename("turnover")
    if log:
        return np.log1p(turnover_)
    return turnover_


def _turnover_zscore(data: pd.DataFrame, *, length: int = 252, log: bool = False) -> pd.Series:
    """Calculate the turnover z-score for one validated instrument.

    The caller must provide chronologically ordered single-instrument data and
    validated parameters. The current observation is excluded from the
    ``length - 1`` row reference window.
    """
    lookback_length = length - 1
    turnover_ = _turnover(data, log=log)
    rolling_turnover = turnover_.shift(1).rolling(
        window=lookback_length, min_periods=lookback_length
    )
    return safe_divide(turnover_ - rolling_turnover.median(), rolling_turnover.std(ddof=0)).rename(
        "turnover_zscore"
    )


def _mfi(data: pd.DataFrame, *, length: int) -> pd.Series:
    typical_price = (data.loc[:, "high"] + data.loc[:, "low"] + data.loc[:, "close"]) / 3
    raw_money_flow = typical_price * data.loc[:, "volume"]
    price_change = typical_price.diff()

    positive_flow = raw_money_flow.where(price_change > 0, 0.0).where(price_change.notna())
    negative_flow = raw_money_flow.where(price_change < 0, 0.0).where(price_change.notna())
    positive_sum = positive_flow.rolling(window=length, min_periods=length).sum()
    negative_sum = negative_flow.rolling(window=length, min_periods=length).sum()

    return (100 * safe_divide(positive_sum, positive_sum + negative_sum)).rename("mfi")
