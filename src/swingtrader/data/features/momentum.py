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
implements PPO-based, RSI-based, and Money Flow Index features, the stochastic
oscillator, the LazyBear squeeze momentum indicator, and a standalone MACD
indicator.
"""

import pandas as pd

from swingtrader.data.features._numerical import (
    consecutive_true_count,
    exponential_moving_average,
    linreg,
    safe_divide,
    wilder_moving_average,
)
from swingtrader.data.features._validation import (
    apply_by_ticker,
    validate_length,
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.data.features.trends import sma
from swingtrader.data.features.volatility import (
    _atr,
    _true_range,
    bollinger_percent_b,
)


def add_momentum_features(
    data: pd.DataFrame,
    *,
    ppo_lengths: tuple[int, int, int] = (12, 26, 9),
    ppo_percentile_min_history: int = 100,
    rsi_length: int = 21,
    rsi_bollinger_length: int = 20,
    rsi_bollinger_num_std: float = 2.0,
    stochastic_k_length: int = 14,
    stochastic_k_smoothing: int = 3,
    stochastic_d_length: int = 3,
    mfi_length: int = 14,
    mfi_bollinger_length: int = 20,
    mfi_bollinger_num_std: float = 2.0,
    squeeze_bb_length: int = 20,
    squeeze_bb_mult: float = 2.0,
    squeeze_kc_length: int = 20,
    squeeze_kc_mult: float = 1.5,
    squeeze_atr_length: int = 14,
) -> pd.DataFrame:
    """Return a copy of data with the default momentum feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, ``close``, ``adjusted_close``, and ``volume`` columns. The
    index must be unique and sorted. The returned dataframe preserves the input
    rows and appends the final PPO, PPO signal, PPO histogram, PPO percentile,
    RSI, RSI %B, stochastic %K and %D, MFI, MFI %B, and LazyBear squeeze momentum
    feature columns.

    PPO, RSI, and ``rsi_percent_b`` are calculated from ``adjusted_close`` so they
    are not distorted by split and dividend discontinuities in the raw close.
    ``rsi_percent_b`` locates the RSI line within its own Bollinger Bands, giving
    a scale-invariant measure of how stretched momentum is relative to its recent
    range. The stochastic oscillator and the Money Flow Index are calculated from
    the raw ``high``, ``low``, ``close``, and (for MFI) ``volume`` because they
    need the intraday extremes and the traded volume together, matching the ADX
    and ATR calculations in the trend and volatility modules. ``mfi_percent_b``
    locates the MFI line within its own Bollinger Bands, mirroring
    ``rsi_percent_b``.

    The LazyBear squeeze momentum features (``squeeze_on``, ``squeeze_off``,
    ``squeeze_released``, ``squeeze_width_ratio``, ``squeeze_momentum_atr``,
    ``squeeze_momentum_atr_change``, ``squeeze_duration``, and
    ``squeeze_release_duration``) are calculated from the raw ``high``, ``low``,
    and ``close``, with True Range and ATR computed internally, because the
    squeeze compares Bollinger Bands against Keltner Channels and normalises the
    momentum histogram by ATR. The raw price-unit ``squeeze_momentum`` line is
    dropped so the persisted ``squeeze_momentum_atr`` feature stays comparable
    across tickers. See :func:`lazybear_squeeze_momentum` for the full definition.
    """
    validate_market_price_index(data)
    validate_required_columns(
        data, required_columns={"high", "low", "close", "adjusted_close", "volume"}
    )
    _validate_fast_slow_signal_lengths(ppo_lengths)
    _validate_min_history(ppo_percentile_min_history)
    validate_length(rsi_length)
    validate_length(mfi_length)

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

    stochastic_block = stochastic_oscillator(
        data.loc[:, ["high", "low", "close"]],
        k_length=stochastic_k_length,
        k_smoothing=stochastic_k_smoothing,
        d_length=stochastic_d_length,
    )
    data[stochastic_block.columns] = stochastic_block

    data["mfi"] = mfi(data.loc[:, ["high", "low", "close", "volume"]], length=mfi_length)
    data["mfi_percent_b"] = bollinger_percent_b(
        data.loc[:, "mfi"],
        length=mfi_bollinger_length,
        num_std=mfi_bollinger_num_std,
    ).rename("mfi_percent_b")

    squeeze_block = lazybear_squeeze_momentum(
        data.loc[:, ["high", "low", "close"]],
        bb_length=squeeze_bb_length,
        bb_mult=squeeze_bb_mult,
        kc_length=squeeze_kc_length,
        kc_mult=squeeze_kc_mult,
        atr_length=squeeze_atr_length,
    ).drop(columns=["squeeze_momentum"])
    data[squeeze_block.columns] = squeeze_block

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


def stochastic_oscillator(
    data: pd.DataFrame,
    *,
    k_length: int = 14,
    k_smoothing: int = 3,
    d_length: int = 3,
) -> pd.DataFrame:
    """Calculate the stochastic oscillator for one or many tickers.

    Returns a dataframe with ``stochastic_k`` and ``stochastic_d`` columns. The
    raw %K locates the close within its recent range as
    ``100 * (close - lowest_low) / (highest_high - lowest_low)`` over ``k_length``
    rows, where ``lowest_low`` and ``highest_high`` are the rolling minimum low
    and maximum high over the same window. ``stochastic_k`` is that raw %K
    smoothed with a simple moving average over ``k_smoothing`` rows, and
    ``stochastic_d`` is a further simple moving average of ``stochastic_k`` over
    ``d_length`` rows. Passing ``k_smoothing=1`` yields the fast stochastic; the
    conventional 14/3/3 defaults yield the slow stochastic. Both series are
    bounded to ``[0, 100]``.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order, because the oscillator needs the intraday extremes and
    the close together. When ``data`` carries the canonical ``provider``,
    ``ticker``, and ``trading_date`` index levels the calculation is isolated
    within each group, so one ticker's history cannot leak into another's, and
    the original index and row order are preserved. A window whose highest high
    equals its lowest low has no range and is left missing, and the warm-up rows
    of each series remain missing until every rolling window is full.
    """
    validate_length(k_length)
    validate_length(k_smoothing)
    validate_length(d_length)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _stochastic_oscillator(
            group,
            k_length=k_length,
            k_smoothing=k_smoothing,
            d_length=d_length,
        ),
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


def lazybear_squeeze_momentum(
    data: pd.DataFrame,
    *,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Calculate LazyBear's Squeeze Momentum indicator for one or many tickers.

    This is a pandas port of the open-source "Squeeze Momentum Indicator
    [LazyBear]" published on TradingView, itself a derivative of John Carter's TTM
    Squeeze. The squeeze compares Bollinger Bands against Keltner Channels: a
    squeeze is on while the Bollinger Bands sit inside the Keltner Channels (low
    volatility, a coiled market) and off once they expand back outside them. A
    separate linear-regression histogram measures the momentum building up while
    the market is squeezed.

    Returns a dataframe with the following columns:

    - ``squeeze_on``: nullable boolean, true while both Bollinger Bands sit inside
      the Keltner Channels;
    - ``squeeze_off``: nullable boolean, true while both Bollinger Bands sit
      outside the Keltner Channels;
    - ``squeeze_released``: nullable boolean, true on the first row after a
      ``squeeze_on`` row that is no longer squeezed, marking the bar the squeeze
      fires;
    - ``squeeze_width_ratio``: the Bollinger-band width divided by the
      Keltner-channel width, a scale-invariant measure of how compressed the bands
      are, where a value below one means the Bollinger Bands are inside the
      channels;
    - ``squeeze_momentum``: the raw linear-regression momentum histogram in the
      input price units;
    - ``squeeze_momentum_atr``: ``squeeze_momentum`` divided by ATR, a
      scale-invariant momentum measure comparable across tickers;
    - ``squeeze_momentum_atr_change``: the row-over-row change in
      ``squeeze_momentum_atr``, capturing whether momentum is building or fading;
    - ``squeeze_duration``: the number of consecutive rows the current squeeze has
      been on, resetting to zero while it is off;
    - ``squeeze_release_duration``: on each ``squeeze_released`` row, how many rows
      the squeeze that just fired had lasted.

    The Bollinger Bands are the ``bb_length``-row simple moving average of
    ``close`` plus and minus ``bb_mult`` population standard deviations. The
    Keltner Channels are the ``kc_length``-row simple moving average of ``close``
    plus and minus ``kc_mult`` times the ``kc_length``-row average True Range. The
    momentum histogram is a rolling linear regression of ``close`` detrended
    against the midpoint of its recent high/low range and moving average, and it
    is normalised by the ``atr_length``-row ATR. True Range and ATR are computed
    internally from ``high``, ``low``, and ``close``, so the caller does not supply
    them, matching the stochastic and Money Flow Index indicators.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order. When ``data`` carries the canonical ``provider``,
    ``ticker``, and ``trading_date`` index levels the calculation is isolated
    within each group, so one ticker's history cannot leak into another's, and the
    original index and row order are preserved. Warm-up rows remain missing until
    every rolling and smoothing window is full. The squeeze state becomes defined
    once the band windows fill, while the momentum histogram warms up later
    because its detrended input is itself a ``kc_length`` calculation that the
    linear regression then windows again.

    This indicator is included in :func:`add_momentum_features`, which appends all
    of these columns except the price-unit ``squeeze_momentum`` line.

    Notes
    -----
    Source: https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
    """
    validate_length(bb_length)
    validate_length(kc_length)
    validate_length(atr_length)
    _validate_multiplier(bb_mult)
    _validate_multiplier(kc_mult)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _lazybear_squeeze_momentum(
            group,
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult,
            atr_length=atr_length,
        ),
    )


def _lazybear_squeeze_momentum(
    data: pd.DataFrame,
    *,
    bb_length: int,
    bb_mult: float,
    kc_length: int,
    kc_mult: float,
    atr_length: int,
) -> pd.DataFrame:
    # Port of the open-source Squeeze Momentum Indicator [LazyBear]:
    # https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
    close = data["close"]
    high = data["high"]
    low = data["low"]
    true_range_ = _true_range(data)
    atr_ = _atr(data, length=atr_length)

    # Bollinger Bands
    bb_basis = sma(close, length=bb_length)
    bb_deviation = (
        bb_mult  # The original LazyBear script uses kc_mult here, most likely a mistake.
        * close.rolling(
            window=bb_length,
            min_periods=bb_length,
        ).std(ddof=0)
    )

    upper_bb = bb_basis + bb_deviation
    lower_bb = bb_basis - bb_deviation

    # Keltner Channels
    kc_basis = sma(close, length=kc_length)
    range_ma = sma(true_range_, length=kc_length)

    upper_kc = kc_basis + kc_mult * range_ma
    lower_kc = kc_basis - kc_mult * range_ma

    squeeze_ready = (
        pd.concat(
            [upper_bb, lower_bb, upper_kc, lower_kc],
            axis=1,
        )
        .notna()
        .all(axis=1)
    )

    squeeze_on = (
        ((lower_bb > lower_kc) & (upper_bb < upper_kc)).astype("boolean").where(squeeze_ready)
    )

    squeeze_off = (
        ((lower_bb < lower_kc) & (upper_bb > upper_kc)).astype("boolean").where(squeeze_ready)
    )

    squeeze_width_ratio = safe_divide(upper_bb - lower_bb, upper_kc - lower_kc)
    squeeze_released = squeeze_on.shift(1, fill_value=False) & squeeze_on.eq(False)
    squeeze_duration = consecutive_true_count(squeeze_on)
    squeeze_release_duration = squeeze_duration.shift(1).where(squeeze_released)

    # Calculate momentum
    highest_high = high.rolling(window=kc_length, min_periods=kc_length).max()
    lowest_low = low.rolling(window=kc_length, min_periods=kc_length).min()

    range_midpoint = (highest_high + lowest_low) / 2.0
    reference_level = (range_midpoint + kc_basis) / 2.0
    detrended_close = close - reference_level

    momentum = linreg(detrended_close, length=kc_length, offset=0)
    momentum_atr = safe_divide(momentum, atr_)
    momentum_atr_change = momentum_atr.diff()

    return pd.DataFrame(
        {
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "squeeze_released": squeeze_released,
            "squeeze_width_ratio": squeeze_width_ratio,
            "squeeze_momentum": momentum,
            "squeeze_momentum_atr": momentum_atr,
            "squeeze_momentum_atr_change": momentum_atr_change,
            "squeeze_duration": squeeze_duration,
            "squeeze_release_duration": squeeze_release_duration,
        },
        index=data.index,
    )


def _stochastic_oscillator(
    data: pd.DataFrame,
    *,
    k_length: int,
    k_smoothing: int,
    d_length: int,
) -> pd.DataFrame:
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]

    lowest_low = low.rolling(window=k_length, min_periods=k_length).min()
    highest_high = high.rolling(window=k_length, min_periods=k_length).max()
    raw_k = 100 * safe_divide(close - lowest_low, highest_high - lowest_low)
    stochastic_k = raw_k.rolling(window=k_smoothing, min_periods=k_smoothing).mean()
    stochastic_d = stochastic_k.rolling(window=d_length, min_periods=d_length).mean()

    return pd.DataFrame(
        {
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
        },
        index=data.index,
    )


def _rsi(values: pd.Series, *, length: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = delta.mul(-1).clip(lower=0)
    avg_gain = wilder_moving_average(gain, length=length)
    avg_loss = wilder_moving_average(loss, length=length)
    return (100 * safe_divide(avg_gain, avg_gain + avg_loss)).rename("rsi")


def _mfi(data: pd.DataFrame, *, length: int) -> pd.Series:
    typical_price = (data.loc[:, "high"] + data.loc[:, "low"] + data.loc[:, "close"]) / 3
    raw_money_flow = typical_price * data.loc[:, "volume"]
    price_change = typical_price.diff()

    positive_flow = raw_money_flow.where(price_change > 0, 0.0).where(price_change.notna())
    negative_flow = raw_money_flow.where(price_change < 0, 0.0).where(price_change.notna())
    positive_sum = positive_flow.rolling(window=length, min_periods=length).sum()
    negative_sum = negative_flow.rolling(window=length, min_periods=length).sum()

    return (100 * safe_divide(positive_sum, positive_sum + negative_sum)).rename("mfi")


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


def _validate_multiplier(multiplier: float) -> None:
    if isinstance(multiplier, bool) or not isinstance(multiplier, int | float) or multiplier <= 0:
        raise ValueError(f"Multiplier must be a positive number; got {multiplier!r}")


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
