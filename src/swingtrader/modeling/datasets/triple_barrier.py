"""Build next-open ATR-scaled triple-barrier targets from daily OHLC data."""

from typing import Literal

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    MARKET_PRICE_INDEX_NAMES,
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.indicators.volatility import atr

IntrabarPolicy = Literal[
    "stop_loss_first",
    "take_profit_first",
    "candle_path",
    "exclude",
]

TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS = (
    *MARKET_PRICE_INDEX_NAMES,
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
)
SUPPORTED_INTRABAR_POLICIES = frozenset(
    {"stop_loss_first", "take_profit_first", "candle_path", "exclude"}
)


def triple_barrier_output_columns(horizons: tuple[int, ...]) -> tuple[str, ...]:
    """Return the declared V3 output columns in deterministic horizon order."""
    return tuple(
        column
        for horizon in horizons
        for column in (
            f"triple_barrier_label_{horizon}d",
            f"time_to_event_{horizon}d",
            f"target_end_date_{horizon}d",
        )
    )


def add_triple_barrier_targets(
    prices: pd.DataFrame,
    *,
    atr_length: int,
    stop_atr_multiple: float,
    reward_risk_ratio: float,
    horizons: tuple[int, ...],
    intrabar_policy: IntrabarPolicy,
) -> pd.DataFrame:
    """Append next-open ATR-scaled triple-barrier targets.

    For each horizon, the label is ``1`` when take-profit is reached first,
    ``-1`` when stop-loss is reached first, and ``0`` on timeout. The event time
    is the 1-based barrier-hit session or the full horizon for a timeout.

    Parameters
    ----------
    prices
        Canonical daily market frame with OHLC and adjusted-close columns.
    atr_length
        Signal-row ATR lookback in observed sessions.
    stop_atr_multiple
        Stop-loss distance as a positive multiple of signal-row ATR.
    reward_risk_ratio
        Take-profit distance relative to the stop-loss distance.
    horizons
        Unique, strictly increasing positive session horizons.
    intrabar_policy
        Resolution rule when one daily bar touches both price barriers.

    Returns
    -------
    pandas.DataFrame
        An independent copy with label, event-time, and target-end-date columns
        for every horizon. Invalid, unresolved, or excluded outcomes are missing.
    """
    _validate_parameters(
        atr_length=atr_length,
        stop_atr_multiple=stop_atr_multiple,
        reward_risk_ratio=reward_risk_ratio,
        horizons=horizons,
        intrabar_policy=intrabar_policy,
    )
    validate_market_price_index(prices)
    _validate_required_columns(prices)

    result = prices.copy()
    outputs = _empty_outputs(len(result), horizons=horizons)
    if result.empty:
        return _append_outputs(result, outputs, horizons=horizons)

    calculation_frame = result.reset_index()
    calculation_frame["__position"] = np.arange(len(result))
    calculation_frame["trading_date"] = pd.to_datetime(calculation_frame["trading_date"])
    for column in ("open", "high", "low", "close", "adjusted_close"):
        calculation_frame[column] = pd.to_numeric(calculation_frame[column], errors="coerce")
    calculation_frame = _adjustment_consistent_ohlc(calculation_frame)

    for _, group in calculation_frame.groupby(["provider", "ticker"], sort=False):
        _label_group(
            group.reset_index(drop=True),
            outputs=outputs,
            atr_length=atr_length,
            stop_atr_multiple=stop_atr_multiple,
            reward_risk_ratio=reward_risk_ratio,
            horizons=horizons,
            intrabar_policy=intrabar_policy,
        )

    return _append_outputs(result, outputs, horizons=horizons)


def _adjustment_consistent_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    raw_close = frame["close"]
    adjusted_close = frame["adjusted_close"]
    factor = adjusted_close.div(raw_close.where(raw_close.ne(0)))
    factor = factor.where(np.isfinite(factor) & factor.gt(0))
    for column in ("open", "high", "low"):
        frame[column] = frame[column].mul(factor)
    frame["close"] = adjusted_close.where(factor.notna())
    return frame


def _label_group(
    group: pd.DataFrame,
    *,
    outputs: dict[str, np.ndarray],
    atr_length: int,
    stop_atr_multiple: float,
    reward_risk_ratio: float,
    horizons: tuple[int, ...],
    intrabar_policy: IntrabarPolicy,
) -> None:
    ohlc = group.loc[:, ["open", "high", "low", "close"]].astype("float64")
    valid_ohlc = _valid_ohlc_rows(ohlc)
    atr_input = ohlc.loc[:, ["high", "low", "close"]].copy()
    atr_input.loc[~valid_ohlc, :] = np.nan
    atr_input.index = pd.DatetimeIndex(group["trading_date"])
    atr_values = atr(atr_input, length=atr_length).to_numpy(dtype="float64", copy=True)
    atr_values[~valid_ohlc] = np.nan

    opens = ohlc["open"].to_numpy()
    highs = ohlc["high"].to_numpy()
    lows = ohlc["low"].to_numpy()
    closes = ohlc["close"].to_numpy()
    positions = group["__position"].to_numpy(dtype="int64")
    dates = group["trading_date"].to_numpy(dtype="datetime64[ns]")

    for horizon in horizons:
        for signal_row in range(len(group) - 1):
            signal_atr = atr_values[signal_row]
            entry_price = opens[signal_row + 1]
            if (
                not np.isfinite(signal_atr)
                or signal_atr <= 0
                or not np.isfinite(entry_price)
                or entry_price <= 0
            ):
                continue

            initial_risk = stop_atr_multiple * signal_atr
            stop_price = entry_price - initial_risk
            take_profit_price = entry_price + reward_risk_ratio * initial_risk
            if stop_price <= 0:
                continue

            available_sessions = min(horizon, len(group) - signal_row - 1)
            label, time_to_event = _first_barrier_label(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                valid_ohlc=valid_ohlc,
                start=signal_row + 1,
                sessions=available_sessions,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                intrabar_policy=intrabar_policy,
            )
            if label is None or time_to_event is None:
                continue
            if label == 0 and available_sessions < horizon:
                continue

            position = positions[signal_row]
            outputs[f"triple_barrier_label_{horizon}d"][position] = label
            outputs[f"time_to_event_{horizon}d"][position] = time_to_event
            outputs[f"target_end_date_{horizon}d"][position] = dates[signal_row + time_to_event]


def _first_barrier_label(
    *,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    valid_ohlc: np.ndarray,
    start: int,
    sessions: int,
    stop_price: float,
    take_profit_price: float,
    intrabar_policy: IntrabarPolicy,
) -> tuple[int | None, int | None]:
    for session in range(1, sessions + 1):
        row = start + session - 1
        if not valid_ohlc[row]:
            return None, None
        if opens[row] <= stop_price:
            return -1, session
        if opens[row] >= take_profit_price:
            return 1, session

        stop_hit = lows[row] <= stop_price
        take_profit_hit = highs[row] >= take_profit_price
        if not stop_hit and not take_profit_hit:
            continue
        if stop_hit and not take_profit_hit:
            return -1, session
        if take_profit_hit and not stop_hit:
            return 1, session

        if intrabar_policy == "exclude":
            return None, None
        if intrabar_policy == "take_profit_first":
            return 1, session
        if intrabar_policy == "stop_loss_first":
            return -1, session
        if closes[row] < opens[row]:
            return 1, session
        return -1, session

    return 0, sessions


def _valid_ohlc_rows(ohlc: pd.DataFrame) -> np.ndarray:
    values = ohlc.to_numpy(dtype="float64")
    open_values, high_values, low_values, close_values = values.T
    return (
        np.isfinite(values).all(axis=1)
        & (values > 0).all(axis=1)
        & (high_values >= np.maximum(open_values, close_values))
        & (low_values <= np.minimum(open_values, close_values))
    )


def _empty_outputs(length: int, *, horizons: tuple[int, ...]) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for horizon in horizons:
        for prefix in ("triple_barrier_label", "time_to_event"):
            values = np.empty(length, dtype=object)
            values[:] = pd.NA
            outputs[f"{prefix}_{horizon}d"] = values
        outputs[f"target_end_date_{horizon}d"] = np.full(
            length,
            np.datetime64("NaT", "ns"),
            dtype="datetime64[ns]",
        )
    return outputs


def _append_outputs(
    result: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    for horizon in horizons:
        result[f"triple_barrier_label_{horizon}d"] = pd.array(
            outputs[f"triple_barrier_label_{horizon}d"],
            dtype="Int8",
        )
        result[f"time_to_event_{horizon}d"] = pd.array(
            outputs[f"time_to_event_{horizon}d"],
            dtype="Int64",
        )
        result[f"target_end_date_{horizon}d"] = pd.array(
            outputs[f"target_end_date_{horizon}d"],
            dtype="datetime64[ns]",
        )
    return result


def _validate_parameters(
    *,
    atr_length: int,
    stop_atr_multiple: float,
    reward_risk_ratio: float,
    horizons: tuple[int, ...],
    intrabar_policy: str,
) -> None:
    if isinstance(atr_length, bool) or not isinstance(atr_length, int) or atr_length <= 0:
        raise ValueError("atr_length must be a positive integer")
    _validate_positive_number(stop_atr_multiple, name="stop_atr_multiple")
    _validate_positive_number(reward_risk_ratio, name="reward_risk_ratio")
    if not horizons or any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("horizons must contain positive integers")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and strictly increasing")
    if intrabar_policy not in SUPPORTED_INTRABAR_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_INTRABAR_POLICIES))
        raise ValueError(f"Unsupported intrabar_policy {intrabar_policy!r}; expected {supported}")


def _validate_positive_number(value: float, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _validate_required_columns(prices: pd.DataFrame) -> None:
    required_value_columns = set(TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS).difference(
        MARKET_PRICE_INDEX_NAMES
    )
    validate_required_columns(prices, required_columns=required_value_columns)
