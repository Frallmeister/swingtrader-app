"""Momentum feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time momentum features from ordered
daily price observations. It composes reusable indicators from
:mod:`swingtrader.indicators` into model-facing feature columns, deciding which
source columns each indicator uses, how quantities are normalized relative to one
another, how historical context is represented, and what the final feature
columns are named. Calculations are isolated by provider/ticker groups and leave
warm-up periods as missing values until each underlying window has enough prior
observations. The family orchestrator returns a copy of the input dataframe with
the final model feature columns appended and does not mutate its input.

Alongside the family orchestrator this module also exposes ``ppo_percentile``, a
model-oriented historical percentile transform. It is a feature construction
rather than a reusable technical indicator, so it lives here rather than in
:mod:`swingtrader.indicators`.
"""

import pandas as pd

from swingtrader.data.features._price_adjustment import (
    adjustment_consistent_price_frame,
)
from swingtrader.data.market_frame import (
    apply_by_ticker,
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.indicators.macd import ppo
from swingtrader.indicators.oscillators import rsi, stochastic_oscillator
from swingtrader.indicators.squeeze_momentum import lazybear_squeeze_momentum
from swingtrader.indicators.volatility import bollinger_percent_b
from swingtrader.indicators.volume import mfi

_MOMENTUM_FEATURE_COLUMNS = (
    "ppo",
    "ppo_signal",
    "ppo_histogram",
    "ppo_percentile",
    "rsi",
    "rsi_percent_b",
    "stochastic_k",
    "stochastic_d",
    "mfi",
    "mfi_percent_b",
    "squeeze_on",
    "squeeze_off",
    "squeeze_released",
    "squeeze_width_ratio",
    "squeeze_momentum_atr",
    "squeeze_momentum_atr_change",
    "squeeze_duration",
    "squeeze_release_duration",
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

    PPO, RSI, and ``rsi_percent_b`` are calculated from ``adjusted_close``. The
    stochastic oscillator, Money Flow Index, and LazyBear squeeze receive
    ``high``, ``low``, and ``close`` transformed onto the same adjusted-close
    scale. Source ``volume`` is retained for MFI. This preserves intraday
    geometry while preventing corporate-action adjustments from appearing as
    cross-session momentum or volatility. ``rsi_percent_b`` and
    ``mfi_percent_b`` locate their oscillator lines within their own Bollinger
    Bands.

    The LazyBear squeeze momentum features (``squeeze_on``, ``squeeze_off``,
    ``squeeze_released``, ``squeeze_width_ratio``, ``squeeze_momentum_atr``,
    ``squeeze_momentum_atr_change``, ``squeeze_duration``, and
    ``squeeze_release_duration``) calculate True Range and ATR internally from
    the adjustment-consistent price frame because the squeeze compares Bollinger
    Bands against Keltner Channels and normalises the
    momentum histogram by ATR. The raw price-unit ``squeeze_momentum`` line is
    dropped so the persisted ``squeeze_momentum_atr`` feature stays comparable
    across tickers. See
    :func:`swingtrader.indicators.squeeze_momentum.lazybear_squeeze_momentum` for
    the full definition.
    """
    validate_market_price_index(data)
    validate_required_columns(
        data, required_columns={"high", "low", "close", "adjusted_close", "volume"}
    )
    validate_new_columns(data, new_columns=_MOMENTUM_FEATURE_COLUMNS)

    data = data.copy()
    adjusted_hlc = adjustment_consistent_price_frame(
        data,
        price_columns=("high", "low", "close"),
    )

    ppo_block = ppo(data.loc[:, "adjusted_close"], lengths=ppo_lengths, use_percent=False)
    data[ppo_block.columns] = ppo_block

    data["ppo_percentile"] = ppo_percentile(
        data.loc[:, "ppo"], min_history=ppo_percentile_min_history
    ).rename("ppo_percentile")

    data["rsi"] = rsi(data.loc[:, "adjusted_close"], length=rsi_length)
    data["rsi_percent_b"] = bollinger_percent_b(
        data.loc[:, "rsi"],
        length=rsi_bollinger_length,
        num_std=rsi_bollinger_num_std,
    ).rename("rsi_percent_b")

    stochastic_block = stochastic_oscillator(
        adjusted_hlc,
        k_length=stochastic_k_length,
        k_smoothing=stochastic_k_smoothing,
        d_length=stochastic_d_length,
    )
    data[stochastic_block.columns] = stochastic_block

    adjusted_hlcv = adjusted_hlc.assign(volume=data.loc[:, "volume"])
    data["mfi"] = mfi(adjusted_hlcv, length=mfi_length)
    data["mfi_percent_b"] = bollinger_percent_b(
        data.loc[:, "mfi"],
        length=mfi_bollinger_length,
        num_std=mfi_bollinger_num_std,
    ).rename("mfi_percent_b")

    squeeze_block = lazybear_squeeze_momentum(
        adjusted_hlc,
        bb_length=squeeze_bb_length,
        bb_mult=squeeze_bb_mult,
        kc_length=squeeze_kc_length,
        kc_mult=squeeze_kc_mult,
        atr_length=squeeze_atr_length,
    ).drop(columns=["squeeze_momentum"])
    data[squeeze_block.columns] = squeeze_block

    return data


def ppo_percentile(
    values: pd.Series,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Calculate point-in-time percentile ranks for one or many tickers.

    Each observation is ranked against the valid observations that precede it,
    producing a model-oriented measure of where the current value sits within its
    own history. When ``values`` carries the canonical ``provider``, ``ticker``,
    and ``trading_date`` index levels the ranks are calculated independently
    within each group. Observations with fewer than ``min_history`` prior valid
    observations are left missing.
    """
    _validate_min_history(min_history)
    return apply_by_ticker(
        values, lambda group: _expanding_percentile(group, min_history=min_history)
    )


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
