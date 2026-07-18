"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time technical features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each rolling or exponential
window has enough prior observations.

Numerical helpers operate on one series and return one series. Feature
generators return dataframes containing only newly calculated feature columns.
The family orchestrator returns a copy of the input dataframe with those feature
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
    dataframe preserves the input rows and appends moving-average ratio features,
    the price percentage oscillator as a ratio, its signal line, and the PPO
    histogram, plus the PPO percentile rank against prior ticker history.
    """
    required_columns = ["adjusted_close"]
    validate_feature_input(data, required_columns=required_columns)
    validate_temporal_order(data=data)
    data = data.copy()

    fast, slow = fast_slow_lengths
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    ppo_block = ppo_features(
        data,
        lengths=ppo_lengths,
        source="adjusted_close",
        run_validation=False,
    )
    feature_blocks = [
        moving_average_features(
            data,
            fast_slow_lengths=fast_slow_lengths,
            source="adjusted_close",
            run_validation=False,
        ),
        ppo_block,
        ppo_percentile_features(
            data.assign(ppo=ppo_block["ppo"]),
            min_history=ppo_percentile_min_history,
            run_validation=False,
        ),
    ]
    features = pd.concat(feature_blocks, axis="columns")
    data[features.columns] = features
    return data


def moving_average_features(
    data: pd.DataFrame,
    *,
    fast_slow_lengths: tuple[int, int] = (20, 50),
    source: str = "adjusted_close",
    run_validation: bool = True,
) -> pd.DataFrame:
    """Calculate moving-average ratio feature columns.

    Returns only ``sma_fast_to_sma_slow``, ``ema_fast_to_ema_slow``, and
    ``ema_fast_to_sma_fast`` while preserving the input index and row order.
    """
    fast, slow = fast_slow_lengths
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    if run_validation:
        validate_feature_input(data, required_columns={source})
        validate_temporal_order(data)

    features = pd.DataFrame(index=data.index)
    source_by_ticker = _grouped_source(data, source)
    sma_fast = source_by_ticker.transform(lambda values: sma(values, length=fast))
    sma_slow = source_by_ticker.transform(lambda values: sma(values, length=slow))
    ema_fast = source_by_ticker.transform(lambda values: ema(values, length=fast))
    ema_slow = source_by_ticker.transform(lambda values: ema(values, length=slow))

    features["sma_fast_to_sma_slow"] = safe_divide(sma_fast, sma_slow).sub(1)
    features["ema_fast_to_ema_slow"] = safe_divide(ema_fast, ema_slow).sub(1)
    features["ema_fast_to_sma_fast"] = safe_divide(ema_fast, sma_fast).sub(1)
    return features


def ppo_features(
    data: pd.DataFrame,
    *,
    lengths: tuple[int, int, int] = (12, 26, 9),
    source: str = "adjusted_close",
    run_validation: bool = True,
) -> pd.DataFrame:
    """Calculate PPO, signal-line, and histogram feature columns.

    Returns only ``ppo``, ``ppo_signal``, and ``ppo_histogram`` while preserving
    the input index and row order. The production ``ppo`` column is stored as a
    ratio rather than percentage points.
    """
    fast, slow, signal_length = lengths
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    if run_validation:
        validate_feature_input(data, required_columns={source})
        validate_temporal_order(data)

    features = pd.DataFrame(index=data.index)
    source_by_ticker = _grouped_source(data, source)
    features["ppo"] = source_by_ticker.transform(
        lambda values: ppo(values, fast=fast, slow=slow, use_percent=False)
    )
    ppo_by_ticker = _grouped_series(data, features["ppo"])
    features["ppo_signal"] = ppo_by_ticker.transform(
        lambda values: ppo_signal(values, length=signal_length)
    )
    features["ppo_histogram"] = ppo_histogram(features["ppo"], features["ppo_signal"])
    return features


def ppo_percentile_features(
    data: pd.DataFrame,
    *,
    min_history: int = 1,
    source: str = "ppo",
    run_validation: bool = True,
) -> pd.DataFrame:
    """Calculate the ``ppo_percentile`` feature column.

    Each non-missing PPO value is ranked against valid observations in the same
    provider/ticker group up to that row, then scaled to a 0-1 percentile using
    only the count of preceding valid observations. Rows with fewer than
    ``min_history`` preceding valid PPO observations remain missing.
    """
    if run_validation:
        validate_feature_input(data, required_columns=[source])
        validate_temporal_order(data)

    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ValueError(
            f"min_history must be a positive integer greater than 0; got {min_history!r}"
        )

    grouped_ppo = _grouped_source(data, source)
    percentile = grouped_ppo.transform(
        lambda values: ppo_percentile(
            values,
            min_history=min_history,
        )
    )
    return percentile.rename("ppo_percentile").to_frame()


def ppo_percentile(
    values: pd.Series,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Calculate point-in-time percentile ranks for one PPO sequence."""
    return _expanding_percentile(values, min_history=min_history)


def ppo(
    values: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    use_percent: bool = True,
) -> pd.Series:
    """Calculate the price percentage oscillator for one numerical sequence.

    PPO is the difference between a fast and slow EMA divided by the slow EMA and
    is returned as percentage points by default. Pass ``use_percent=False`` to
    keep the raw ratio.
    """
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    ema_fast = ema(values, length=fast)
    ema_slow = ema(values, length=slow)
    result = safe_divide(ema_fast - ema_slow, ema_slow)
    if use_percent:
        result = result.mul(100)
    return result


def ppo_signal(
    values: pd.Series,
    *,
    length: int = 9,
) -> pd.Series:
    """Calculate an EMA signal line for one PPO sequence."""
    return ema(values, length=length)


def ppo_histogram(
    ppo_values: pd.Series,
    signal_values: pd.Series,
) -> pd.Series:
    """Calculate PPO histogram values as ``ppo - ppo_signal``."""
    return ppo_values - signal_values


def sma(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate a simple moving average for one numerical sequence."""
    _validate_length(length)
    return values.rolling(window=length, min_periods=length).mean()


def ema(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate an exponential moving average for one numerical sequence."""
    _validate_length(length)
    return values.ewm(span=length, adjust=False, min_periods=length).mean()


def _validate_length(length: int) -> None:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"Length must be a positive integer; got {length!r}")


def _grouped_source(data: pd.DataFrame, source: str) -> pd.core.groupby.SeriesGroupBy:
    return _grouped_series(data, data.loc[:, source])


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
