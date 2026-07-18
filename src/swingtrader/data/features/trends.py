"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time technical features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each rolling or exponential
window has enough prior observations.

Numerical indicators operate on one ordered series and return either one series
or, for naturally multi-output indicators, one dataframe. The family
orchestrator returns a copy of the input dataframe with final model feature
columns appended.
"""

import pandas as pd

from swingtrader.data.features._numerical import safe_divide
from swingtrader.data.features._validation import validate_feature_input, validate_temporal_order


def add_trend_features(
    data: pd.DataFrame,
    *,
    fast_slow_lengths: tuple[int, int] = (20, 50),
    ppo_lengths: tuple[int, int, int] = (12, 26, 9),
    ppo_percentile_min_history: int = 100,
) -> pd.DataFrame:
    """Return a copy of data with the default trend feature set added.

    The input must contain provider, ticker, and trading_date identifiers either
    as columns or named index levels, plus an adjusted_close column. The returned
    dataframe preserves the input rows and appends the final moving-average
    ratio, PPO, PPO signal, PPO histogram, and PPO percentile feature columns.
    """
    required_columns = ["adjusted_close"]
    validate_feature_input(data, required_columns=required_columns)
    validate_temporal_order(data=data)

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
    adjusted_close_by_ticker = _grouped_series(data, data.loc[:, "adjusted_close"])

    sma_fast = adjusted_close_by_ticker.transform(lambda values: sma(values, length=fast))
    sma_slow = adjusted_close_by_ticker.transform(lambda values: sma(values, length=slow))
    ema_fast = adjusted_close_by_ticker.transform(lambda values: ema(values, length=fast))
    ema_slow = adjusted_close_by_ticker.transform(lambda values: ema(values, length=slow))

    data["sma_fast_to_sma_slow"] = safe_divide(sma_fast, sma_slow).sub(1)
    data["ema_fast_to_ema_slow"] = safe_divide(ema_fast, ema_slow).sub(1)
    data["ema_fast_to_sma_fast"] = safe_divide(ema_fast, sma_fast).sub(1)

    ppo_block = _grouped_ppo(data, data.loc[:, "adjusted_close"], lengths=ppo_lengths)
    data[ppo_block.columns] = ppo_block

    ppo_by_ticker = _grouped_series(data, data.loc[:, "ppo"])
    data["ppo_percentile"] = ppo_by_ticker.transform(
        lambda values: ppo_percentile(values, min_history=ppo_percentile_min_history)
    )
    return data


def sma(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate a simple moving average for one numerical sequence.

    ``values`` must contain observations in chronological order. The returned
    series preserves the input index, with the first ``length - 1`` observations
    left missing until the rolling window is full.
    """
    _validate_length(length)
    _validate_temporal_index(values)
    return values.rolling(window=length, min_periods=length).mean()


def ema(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate an exponential moving average for one numerical sequence.

    ``values`` must contain observations in chronological order. The returned
    series preserves the input index. EMA uses pandas ``ewm`` with
    ``span=length``, ``adjust=False``, and ``min_periods=length``.
    """
    _validate_length(length)
    _validate_temporal_index(values)
    return values.ewm(span=length, adjust=False, min_periods=length).mean()


def ppo(
    values: pd.Series,
    *,
    lengths: tuple[int, int, int] = (12, 26, 9),
) -> pd.DataFrame:
    """Calculate PPO, signal-line, and histogram values for one sequence.

    ``values`` must contain observations in chronological order. The returned
    dataframe preserves the input index and contains ``ppo``, ``ppo_signal``,
    and ``ppo_histogram`` columns. PPO values use the production ratio scaling.
    """
    fast_length, slow_length, signal_length = _validate_ppo_lengths(lengths)
    _validate_temporal_index(values)

    ema_fast = ema(values, length=fast_length)
    ema_slow = ema(values, length=slow_length)
    ppo_values = safe_divide(ema_fast - ema_slow, ema_slow)
    signal_values = ema(ppo_values, length=signal_length)
    histogram_values = ppo_values - signal_values

    return pd.DataFrame(
        {
            "ppo": ppo_values,
            "ppo_signal": signal_values,
            "ppo_histogram": histogram_values,
        },
        index=values.index,
    )


def ppo_percentile(
    values: pd.Series,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Calculate point-in-time percentile ranks for one PPO sequence."""
    _validate_min_history(min_history)
    _validate_temporal_index(values)
    return _expanding_percentile(values, min_history=min_history)


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
    if isinstance(index, pd.DatetimeIndex | pd.PeriodIndex):
        is_ordered = index.is_monotonic_increasing
    elif isinstance(index, pd.MultiIndex) and "trading_date" in index.names:
        is_ordered = index.get_level_values("trading_date").is_monotonic_increasing
    else:
        return

    if not is_ordered:
        raise ValueError("values must be chronologically ordered before calculating this indicator")


def _grouped_series(data: pd.DataFrame, values: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    identifiers = ("provider", "ticker")
    identifiers_set = set(identifiers)
    index_names = data.index.names
    columns = data.columns

    if identifiers_set.issubset(index_names):
        return values.groupby(
            [data.index.get_level_values(identifier) for identifier in identifiers],
            sort=False,
        )
    if identifiers_set.issubset(columns):
        return values.groupby(
            [data[identifier] for identifier in identifiers],
            sort=False,
        )
    raise ValueError("The identifiers 'provider' and 'ticker' must be in either index or columns")


def _grouped_ppo(
    data: pd.DataFrame,
    values: pd.Series,
    *,
    lengths: tuple[int, int, int],
) -> pd.DataFrame:
    columns = ["ppo", "ppo_signal", "ppo_histogram"]
    blocks = [
        ppo(group_values, lengths=lengths) for _, group_values in _grouped_series(data, values)
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
