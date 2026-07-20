"""MACD and PPO moving-average convergence/divergence indicators.

Both indicators are built from the fast and slow EMAs of a single ordered series.
They accept either one instrument or a multi-instrument series carrying the
canonical ``provider``, ``ticker``, and ``trading_date`` index levels, in which
case the calculation is applied independently within each provider/ticker group
and the input row order is preserved. Each function returns a new dataframe and
does not mutate its input.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker
from swingtrader.indicators._smoothing import exponential_moving_average
from swingtrader.indicators._validation import validate_fast_slow_signal_lengths


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
    """
    fast_length, slow_length, signal_length = validate_fast_slow_signal_lengths(lengths)
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
    fast_length, slow_length, signal_length = validate_fast_slow_signal_lengths(lengths)
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
