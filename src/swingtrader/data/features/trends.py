"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time technical features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each rolling or exponential
window has enough prior observations.
"""

from typing import Literal

import pandas as pd

from swingtrader.data.features._numerical import safe_divide
from swingtrader.data.features._validation import validate_feature_input, validate_temporal_order


def add_trend_features(
    data: pd.DataFrame,
    *,
    fast_slow_lengths: tuple[int, int] = (20, 50),
    ppo_lengths: tuple[int, int, int] = (12, 26, 9),
) -> pd.DataFrame:
    """Add the default trend feature set to a price dataframe.

    The input must contain provider, ticker, and trading_date identifiers either
    as columns or named index levels, plus an adjusted_close column. The returned
    dataframe preserves the input rows and appends moving-average ratio features,
    the price percentage oscillator as a ratio, its signal line, and the PPO
    histogram.
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

    # Moving average features
    sma_fast = sma(data=data, length=fast, source="adjusted_close", run_validation=False)
    sma_slow = sma(data=data, length=slow, source="adjusted_close", run_validation=False)
    ema_fast = ema(data=data, length=fast, source="adjusted_close", run_validation=False)
    ema_slow = ema(data=data, length=slow, source="adjusted_close", run_validation=False)
    data["sma_fast_to_sma_slow"] = safe_divide(sma_fast, sma_slow).sub(1)
    data["ema_fast_to_ema_slow"] = safe_divide(ema_fast, ema_slow).sub(1)
    data["ema_fast_to_sma_fast"] = safe_divide(ema_fast, sma_fast).sub(1)

    # PPO features
    ppo_fast, ppo_slow, ppo_signal_length = ppo_lengths
    data["ppo"] = ppo(
        data,
        fast=ppo_fast,
        slow=ppo_slow,
        source="adjusted_close",
        use_percent=False,
        run_validation=False,
    )
    data["ppo_signal"] = ppo_signal(data, length=ppo_signal_length, run_validation=False)
    data["ppo_histogram"] = ppo_histogram(data)
    return data


def ppo(
    data: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    source: str = "adjusted_close",
    use_percent: bool = True,
    run_validation: bool = True,
) -> pd.Series:
    """Calculate the price percentage oscillator for a source column.

    PPO is the difference between a fast and slow EMA divided by the slow EMA.
    Results are grouped by provider/ticker and are returned as percentage points
    by default; pass ``use_percent=False`` to keep the raw ratio.
    """
    if source not in data.columns:
        raise ValueError(f"Source column {source!r} does not exist.")

    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    if run_validation:
        validate_feature_input(data, required_columns={source})
        validate_temporal_order(data)
    ema_fast = ema(data=data, length=fast, source=source, run_validation=False)
    ema_slow = ema(data=data, length=slow, source=source, run_validation=False)
    result = safe_divide(ema_fast - ema_slow, ema_slow)
    if use_percent:
        result = result.mul(100)
    return result


def ppo_signal(
    data: pd.DataFrame,
    *,
    length: int = 9,
    run_validation: bool = True,
) -> pd.Series:
    """Calculate the EMA signal line for an existing ``ppo`` column."""
    if "ppo" not in data.columns:
        raise ValueError("The dataframe must have a column named 'ppo'.")
    return ema(data=data, length=length, source="ppo", run_validation=run_validation)


def ppo_histogram(
    data: pd.DataFrame,
) -> pd.Series:
    """Calculate PPO histogram values as ``ppo - ppo_signal``."""
    required_columns = {"ppo", "ppo_signal"}
    if not required_columns.issubset(data.columns):
        raise ValueError("The dataframe must have the columns 'ppo' and 'ppo_signal'.")
    return data["ppo"] - data["ppo_signal"]


def sma(
    *,
    data: pd.DataFrame,
    length: int,
    source: str,
    run_validation: bool = True,
) -> pd.Series:
    """Calculate a simple moving average per provider/ticker group."""
    return _moving_average(
        data=data,
        length=length,
        source=source,
        method="sma",
        run_validation=run_validation,
    )


def ema(
    *,
    data: pd.DataFrame,
    length: int,
    source: str,
    run_validation: bool = True,
) -> pd.Series:
    """Calculate an exponential moving average per provider/ticker group."""
    return _moving_average(
        data=data,
        length=length,
        source=source,
        method="ema",
        run_validation=run_validation,
    )


def _moving_average(
    *,
    data: pd.DataFrame,
    length: int,
    source: str,
    method: Literal["sma", "ema"],
    run_validation: bool,
) -> pd.Series:
    """Calculate a grouped moving average while preserving input row alignment."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"Length must be a positive integer greater than zero; got {length!r}")

    if source not in data.columns:
        raise ValueError(f"Source column {source!r} does not exist.")

    if run_validation:
        validate_feature_input(data, required_columns={source})
        validate_temporal_order(data)
    values = data.loc[:, source]

    identifiers = ("provider", "ticker")
    identifiers_set = set(identifiers)
    index_names = data.index.names
    columns = data.columns
    if identifiers_set.issubset(index_names):
        values_by_ticker = values.groupby(
            [data.index.get_level_values(identifier) for identifier in identifiers],
            sort=False,
        )
    elif identifiers_set.issubset(columns):
        values_by_ticker = values.groupby(
            [data[identifier] for identifier in identifiers],
            sort=False,
        )
    else:
        raise ValueError(
            "The identifiers 'provider' and 'ticker' must be in either index or columns"
        )

    if method == "sma":
        return values_by_ticker.transform(
            lambda series: series.rolling(window=length, min_periods=length).mean()
        )
    elif method == "ema":
        return values_by_ticker.transform(
            lambda series: series.ewm(span=length, adjust=False, min_periods=length).mean()
        )
    else:
        raise ValueError(f"Method must be either 'sma' or 'ema', got {method!r}")
