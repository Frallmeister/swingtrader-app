"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time technical features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each rolling or exponential
window has enough prior observations.

Numerical indicators accept either one ordered series for a single ticker or a
multi-ticker series carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. In the latter case the calculation is applied
independently within each provider/ticker group and the input row order is
preserved. Indicators return either one series or, for naturally multi-output
indicators, one dataframe. The family orchestrator returns a copy of the input
dataframe with final model feature columns appended.
"""

from collections.abc import Callable

import pandas as pd

from swingtrader.data.features._numerical import safe_divide
from swingtrader.data.features._validation import (
    validate_market_price_index,
    validate_required_columns,
)


def add_trend_features(
    data: pd.DataFrame,
    *,
    fast_slow_lengths: tuple[int, int] = (20, 50),
    ppo_lengths: tuple[int, int, int] = (12, 26, 9),
    ppo_percentile_min_history: int = 100,
) -> pd.DataFrame:
    """Return a copy of data with the default trend feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus an
    ``adjusted_close`` column. The index must be unique and sorted. The returned
    dataframe preserves the input rows and appends the final moving-average
    ratio, PPO, PPO signal, PPO histogram, and PPO percentile feature columns.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"adjusted_close"})

    fast, slow = fast_slow_lengths
    _validate_length(fast)
    _validate_length(slow)
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )
    _validate_ppo_lengths(ppo_lengths)
    _validate_min_history(ppo_percentile_min_history)

    data = data.copy()
    adjusted_close_by_ticker = data.loc[:, "adjusted_close"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )

    sma_fast = adjusted_close_by_ticker.transform(lambda values: _sma(values, length=fast))
    sma_slow = adjusted_close_by_ticker.transform(lambda values: _sma(values, length=slow))
    ema_fast = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=fast))
    ema_slow = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=slow))

    data["sma_fast_to_sma_slow"] = safe_divide(sma_fast, sma_slow).sub(1)
    data["ema_fast_to_ema_slow"] = safe_divide(ema_fast, ema_slow).sub(1)
    data["ema_fast_to_sma_fast"] = safe_divide(ema_fast, sma_fast).sub(1)

    ppo_block = _grouped_ppo(data, data.loc[:, "adjusted_close"], lengths=ppo_lengths)
    data[ppo_block.columns] = ppo_block

    ppo_by_ticker = data.loc[:, "ppo"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )
    data["ppo_percentile"] = ppo_by_ticker.transform(
        lambda values: _expanding_percentile(values, min_history=ppo_percentile_min_history)
    )
    return data


def sma(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate a simple moving average for one or many tickers.

    ``values`` must contain observations in chronological order. When ``values``
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the average is calculated independently within each group. The
    returned series preserves the input index, with the first ``length - 1``
    observations of each series left missing until the rolling window is full.
    """
    _validate_length(length)
    return _apply_by_ticker(values, lambda group: _sma(group, length=length))


def ema(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate an exponential moving average for one or many tickers.

    ``values`` must contain observations in chronological order. When ``values``
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the average is calculated independently within each group. The
    returned series preserves the input index. EMA uses pandas ``ewm`` with
    ``span=length``, ``adjust=False``, and ``min_periods=length``.
    """
    _validate_length(length)
    return _apply_by_ticker(values, lambda group: _ema(group, length=length))


def ppo(
    values: pd.Series,
    *,
    lengths: tuple[int, int, int] = (12, 26, 9),
    use_percent: bool = True,
) -> pd.DataFrame:
    """Calculate PPO, signal-line, and histogram values for one or many tickers.

    When ``values`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the indicator is calculated independently
    within each group. PPO is returned in percentage points by default. Pass
    ``use_percent=False`` to return the raw ratio. The signal and histogram use
    the same scaling as PPO.
    """
    fast_length, slow_length, signal_length = _validate_ppo_lengths(lengths)
    return _apply_by_ticker(
        values,
        lambda group: _ppo(
            group,
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
            use_percent=use_percent,
        ),
    )


def ppo_percentile(
    values: pd.Series,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Calculate point-in-time percentile ranks for one or many tickers.

    When ``values`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the ranks are calculated independently within
    each group.
    """
    _validate_min_history(min_history)
    return _apply_by_ticker(
        values, lambda group: _expanding_percentile(group, min_history=min_history)
    )


def _sma(values: pd.Series, *, length: int) -> pd.Series:
    return values.rolling(window=length, min_periods=length).mean()


def _ema(values: pd.Series, *, length: int) -> pd.Series:
    return values.ewm(span=length, adjust=False, min_periods=length).mean()


def _ppo(
    values: pd.Series,
    *,
    fast_length: int,
    slow_length: int,
    signal_length: int,
    use_percent: bool,
) -> pd.DataFrame:
    ema_fast = _ema(values, length=fast_length)
    ema_slow = _ema(values, length=slow_length)
    ppo_values = safe_divide(ema_fast - ema_slow, ema_slow)
    if use_percent:
        ppo_values = 100 * ppo_values
    signal_values = _ema(ppo_values, length=signal_length)
    histogram_values = ppo_values - signal_values

    return pd.DataFrame(
        {
            "ppo": ppo_values,
            "ppo_signal": signal_values,
            "ppo_histogram": histogram_values,
        },
        index=values.index,
    )


def _validate_length(length: int) -> None:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"Length must be a positive integer; got {length!r}")


def _validate_ppo_lengths(lengths: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(lengths) != 3:
        raise ValueError("PPO lengths must contain fast, slow, and signal lengths.")

    fast_length, slow_length, signal_length = lengths
    _validate_length(fast_length)
    _validate_length(slow_length)
    _validate_length(signal_length)
    if fast_length >= slow_length:
        raise ValueError(
            "The fast length must be lower than the slow length; "
            f"got fast={fast_length!r}, slow={slow_length!r}"
        )
    return fast_length, slow_length, signal_length


def _validate_min_history(min_history: int) -> None:
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ValueError(
            f"min_history must be a positive integer greater than 0; got {min_history!r}"
        )


def _validate_temporal_index(values: pd.Series) -> None:
    index = values.index
    if isinstance(index, pd.DatetimeIndex | pd.PeriodIndex) and not index.is_monotonic_increasing:
        raise ValueError("values must be chronologically ordered before calculating this indicator")


def _apply_by_ticker(
    values: pd.Series,
    func: Callable[[pd.Series], pd.Series | pd.DataFrame],
) -> pd.Series | pd.DataFrame:
    """Apply ``func`` per provider/ticker group for a multi-ticker series.

    When ``values`` carries a ``MultiIndex`` the canonical market-price contract
    is enforced, the calculation is isolated within each provider/ticker group,
    and the original index is preserved. Otherwise ``values`` is treated as a
    single ordered sequence and ``func`` is applied after a temporal-order check.
    """
    if isinstance(values.index, pd.MultiIndex):
        validate_market_price_index(values)
        if values.empty:
            return func(values)
        results = [
            func(group) for _, group in values.groupby(level=["provider", "ticker"], sort=False)
        ]
        return pd.concat(results).reindex(values.index)

    _validate_temporal_index(values)
    return func(values)


def _grouped_ppo(
    data: pd.DataFrame,
    values: pd.Series,
    *,
    lengths: tuple[int, int, int],
) -> pd.DataFrame:
    columns = ["ppo", "ppo_signal", "ppo_histogram"]
    blocks = [
        ppo(group_values, lengths=lengths, use_percent=False)
        for _, group_values in values.groupby(level=["provider", "ticker"], sort=False)
    ]
    if not blocks:
        return pd.DataFrame(index=data.index, columns=columns, dtype="float64")
    return pd.concat(blocks).reindex(data.index).loc[:, columns]


def _expanding_percentile(
    values: pd.Series,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Rank each value against valid observations preceding it."""
    expanding_rank = values.expanding().rank(method="max")
    valid_count = values.notna().cumsum()

    previous_count = valid_count - 1
    percentile = (expanding_rank - 1) / previous_count

    return percentile.where(previous_count >= min_history)
