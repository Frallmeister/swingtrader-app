"""Momentum feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time momentum features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each exponential or
expanding-history window has enough prior observations.

Numerical indicators accept either one ordered series for a single ticker or a
multi-ticker series carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. In the latter case the calculation is applied
independently within each provider/ticker group and the input row order is
preserved. Indicators return either one series or, for naturally multi-output
indicators, one dataframe. The family orchestrator returns a copy of the input
dataframe with final model feature columns appended. The module currently
implements PPO-based and RSI-based features and a standalone MACD indicator.
"""

import pandas as pd

from swingtrader.data.features._numerical import (
    exponential_moving_average,
    safe_divide,
    wilder_moving_average,
)
from swingtrader.data.features._validation import (
    apply_by_ticker,
    validate_length,
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.data.features.volatility import bollinger_percent_b


def add_momentum_features(
    data: pd.DataFrame,
    *,
    ppo_lengths: tuple[int, int, int] = (12, 26, 9),
    ppo_percentile_min_history: int = 100,
    rsi_length: int = 14,
    rsi_bollinger_length: int = 20,
    rsi_bollinger_num_std: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of data with the default momentum feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus an
    ``adjusted_close`` column. The index must be unique and sorted. The returned
    dataframe preserves the input rows and appends the final PPO, PPO signal, PPO
    histogram, PPO percentile, RSI, and RSI %B feature columns.

    RSI is calculated from ``adjusted_close`` so its gains and losses are not
    distorted by split and dividend discontinuities in the raw close. ``rsi_percent_b``
    locates the RSI line within its own Bollinger Bands, giving a scale-invariant
    measure of how stretched momentum is relative to its recent range.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"adjusted_close"})
    _validate_fast_slow_signal_lengths(ppo_lengths)
    _validate_min_history(ppo_percentile_min_history)
    validate_length(rsi_length)

    data = data.copy()

    ppo_block = ppo(data.loc[:, "adjusted_close"], lengths=ppo_lengths, use_percent=False)
    data[ppo_block.columns] = ppo_block

    ppo_by_ticker = data.loc[:, "ppo"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )
    data["ppo_percentile"] = ppo_by_ticker.transform(
        lambda values: _expanding_percentile(values, min_history=ppo_percentile_min_history)
    )

    data["rsi"] = rsi(data.loc[:, "adjusted_close"], length=rsi_length)
    data["rsi_percent_b"] = bollinger_percent_b(
        data.loc[:, "rsi"],
        length=rsi_bollinger_length,
        num_std=rsi_bollinger_num_std,
    ).rename("rsi_percent_b")
    return data


def macd(
    values: pd.Series,
    *,
    lengths: tuple[int, int, int] = (12, 26, 9),
) -> pd.DataFrame:
    """Calculate MACD, signal-line, and histogram values for one or many tickers.

    Returns a dataframe with ``macd``, ``macd_signal``, and ``macd_histogram``
    columns. MACD is the difference between the fast and slow EMAs of ``values``
    and is expressed in the input price units, ``macd_signal`` is an EMA over
    ``macd``, and ``macd_histogram`` is ``macd`` minus ``macd_signal``. When
    ``values`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the indicator is calculated independently
    within each group.

    MACD is not part of :func:`add_momentum_features`; it is provided as a
    standalone indicator for future consumers such as the frontend application.
    """
    fast_length, slow_length, signal_length = _validate_fast_slow_signal_lengths(lengths)
    return apply_by_ticker(
        values,
        lambda group: _macd(
            group,
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
        ),
    )


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
    fast_length, slow_length, signal_length = _validate_fast_slow_signal_lengths(lengths)
    return apply_by_ticker(
        values,
        lambda group: _ppo(
            group,
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
            use_percent=use_percent,
        ),
    )


def rsi(
    values: pd.Series,
    *,
    length: int = 14,
) -> pd.Series:
    """Calculate Wilder's Relative Strength Index for one or many tickers.

    RSI is a bounded ``[0, 100]`` momentum oscillator built from the average gain
    and average loss over ``length`` rows, each smoothed with Wilder's recursive
    moving average. It is calculated as ``100 * avg_gain / (avg_gain + avg_loss)``,
    so a window with no losses returns 100 and a window with no gains returns 0. A
    fully flat window has neither gains nor losses and is left missing.

    ``values`` is a single ordered series, so the caller chooses the source, such
    as close, adjusted close, or an OHLC average. When it carries the canonical
    ``provider``, ``ticker``, and ``trading_date`` index levels the calculation is
    isolated within each group. The first ``length`` rows of each series remain
    missing until the smoothing window is full.

    The smoothing is the recursive form seeded from the first change rather than
    the canonical Wilder definition that seeds from the simple average of the
    first ``length`` changes, so early RSI values differ slightly from a canonical
    implementation before converging (see
    :func:`swingtrader.data.features._numerical.wilder_moving_average`).
    """
    validate_length(length)
    return apply_by_ticker(values, lambda group: _rsi(group, length=length))


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
    return apply_by_ticker(
        values, lambda group: _expanding_percentile(group, min_history=min_history)
    )


def _rsi(values: pd.Series, *, length: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = delta.mul(-1).clip(lower=0)
    avg_gain = wilder_moving_average(gain, length=length)
    avg_loss = wilder_moving_average(loss, length=length)
    return (100 * safe_divide(avg_gain, avg_gain + avg_loss)).rename("rsi")


def _macd(
    values: pd.Series,
    *,
    fast_length: int,
    slow_length: int,
    signal_length: int,
) -> pd.DataFrame:
    ema_fast = exponential_moving_average(values, length=fast_length)
    ema_slow = exponential_moving_average(values, length=slow_length)
    macd_values = ema_fast - ema_slow
    signal_values = exponential_moving_average(macd_values, length=signal_length)
    histogram_values = macd_values - signal_values

    return pd.DataFrame(
        {
            "macd": macd_values,
            "macd_signal": signal_values,
            "macd_histogram": histogram_values,
        },
        index=values.index,
    )


def _ppo(
    values: pd.Series,
    *,
    fast_length: int,
    slow_length: int,
    signal_length: int,
    use_percent: bool,
) -> pd.DataFrame:
    ema_fast = exponential_moving_average(values, length=fast_length)
    ema_slow = exponential_moving_average(values, length=slow_length)
    ppo_values = safe_divide(ema_fast - ema_slow, ema_slow)
    if use_percent:
        ppo_values = 100 * ppo_values
    signal_values = exponential_moving_average(ppo_values, length=signal_length)
    histogram_values = ppo_values - signal_values

    return pd.DataFrame(
        {
            "ppo": ppo_values,
            "ppo_signal": signal_values,
            "ppo_histogram": histogram_values,
        },
        index=values.index,
    )


def _validate_fast_slow_signal_lengths(lengths: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(lengths) != 3:
        raise ValueError("Lengths must contain fast, slow, and signal lengths.")

    fast_length, slow_length, signal_length = lengths
    validate_length(fast_length)
    validate_length(slow_length)
    validate_length(signal_length)
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
