"""Volume-based indicators: the Money Flow Index.

MFI needs the intraday extremes, the close, and the traded volume together, so it
consumes a dataframe with ``high``, ``low``, ``close``, and ``volume`` columns. It
accepts either one instrument or a multi-instrument input carrying the canonical
``provider``, ``ticker``, and ``trading_date`` index levels, in which case the
calculation is isolated within each provider/ticker group and the input row order
is preserved. The function returns a new index-aligned series and does not mutate
its input.

MFI lives in the indicator volume module even though the resulting model feature
currently belongs to the momentum feature family.
"""

import numpy as np
import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import validate_length


def turnover(data: pd.DataFrame, *, log: bool = False) -> pd.Series:
    """ADD DOCSTRING HERE."""
    validate_required_columns(data, required_columns={"adjusted_close", "volume"})
    if not isinstance(log, bool):
        raise ValueError(f"The log parameter must be a boolean; got {log!r}")
    return _turnover(data, log=log)


def turnover_zscore(data: pd.DataFrame, *, length: int = 252, log: bool = False) -> pd.Series:
    """ADD DOCSTRING HERE."""
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
    """ADD DOCSTRING HERE."""
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
