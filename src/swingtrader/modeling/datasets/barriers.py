"""Build ATR-scaled stop-loss and take-profit targets from daily OHLCV data.

Signals use information through the completed signal session and enter at the
next observed open. The module handles adjustment-consistent prices, opening
gaps, same-bar ambiguity, incomplete terminal horizons, and deterministic
versioned output schemas.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from swingtrader.indicators.volatility import atr

EntryPriceRule = Literal["next_open"]
IntrabarPolicy = Literal[
    "stop_first",
    "target_first",
    "candle_path",
    "exclude_ambiguous",
]

BARRIER_REQUIRED_PRICE_COLUMNS = (
    "provider",
    "ticker",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
)
SUPPORTED_ENTRY_PRICE_RULES = frozenset({"next_open"})
SUPPORTED_INTRABAR_POLICIES = frozenset(
    {"stop_first", "target_first", "candle_path", "exclude_ambiguous"}
)


def barrier_output_columns(horizons: tuple[int, ...]) -> tuple[str, ...]:
    """Return the declared output columns for ATR barrier horizons.

    Parameters
    ----------
    horizons
        Positive observed-session horizons used by the barrier target family.

    Returns
    -------
    tuple[str, ...]
        Output column names in deterministic horizon-major order.
    """
    return tuple(
        column
        for horizon in horizons
        for column in (
            f"barrier_event_{horizon}d",
            f"target_tp_before_sl_{horizon}d",
            f"event_session_{horizon}d",
            f"time_to_event_{horizon}d",
            f"ambiguous_intrabar_{horizon}d",
            f"target_end_date_{horizon}d",
        )
    )


def add_atr_barrier_targets(
    prices: pd.DataFrame,
    *,
    atr_length: int,
    stop_atr_multiple: float,
    reward_risk_ratio: float,
    horizons: tuple[int, ...],
    entry_price_rule: EntryPriceRule,
    intrabar_policy: IntrabarPolicy,
) -> pd.DataFrame:
    """Append next-open ATR barrier outcomes over future observed sessions.

    The signal row may use information through its completed daily bar. Entry is
    the next observed session's opening price. Stop and take-profit barriers are
    fixed from the signal row's point-in-time ATR. Raw OHLC values are first
    expressed on the adjusted-close scale so corporate actions do not create
    artificial gaps or volatility. Opening gaps are evaluated before intrabar
    highs and lows.

    A bar touching both barriers is marked in ``ambiguous_intrabar_{horizon}d``.
    ``stop_first`` and ``target_first`` resolve it directly. ``candle_path`` uses
    open-low-high-close for green candles and open-high-low-close for red candles;
    doji bars resolve conservatively to the stop. ``exclude_ambiguous`` emits the
    categorical ``ambiguous`` event and leaves the binary target missing.

    Rows remain unlabeled when the available future data cannot resolve an event
    or a complete timeout horizon, or when required ATR/OHLC values are invalid.

    Parameters
    ----------
    prices
        Daily price rows containing provider, ticker, trading date, raw OHLC,
        and adjusted close columns.
    atr_length
        Number of completed sessions used by Wilder ATR.
    stop_atr_multiple
        Positive multiple of signal-row ATR placed below the entry price.
    reward_risk_ratio
        Positive take-profit distance relative to the initial stop distance.
    horizons
        Unique, strictly increasing observed-session horizons.
    entry_price_rule
        Entry convention. The current implementation supports ``next_open``.
    intrabar_policy
        Deterministic policy for sessions whose high and low touch both barriers.

    Returns
    -------
    pd.DataFrame
        A copy of ``prices`` with nullable barrier-event outputs appended for
        every configured horizon.

    Raises
    ------
    ValueError
        If configuration values are invalid, required columns are missing, or
        provider/ticker/trading-date observations are duplicated.
    """
    _validate_parameters(
        atr_length=atr_length,
        stop_atr_multiple=stop_atr_multiple,
        reward_risk_ratio=reward_risk_ratio,
        horizons=horizons,
        entry_price_rule=entry_price_rule,
        intrabar_policy=intrabar_policy,
    )
    _validate_required_columns(prices)

    result = prices.copy()
    outputs = _empty_outputs(len(result), horizons=horizons)
    if result.empty:
        return _append_outputs(result, outputs, horizons=horizons)

    normalized_dates = pd.to_datetime(result["trading_date"])
    _validate_unique_observations(result, normalized_dates)
    calculation_frame = pd.DataFrame(
        {
            "__position": np.arange(len(result)),
            "provider": result["provider"].to_numpy(),
            "ticker": result["ticker"].to_numpy(),
            "trading_date": normalized_dates.to_numpy(),
            "open": pd.to_numeric(result["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(result["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(result["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(result["close"], errors="coerce").to_numpy(),
            "adjusted_close": pd.to_numeric(result["adjusted_close"], errors="coerce").to_numpy(),
        }
    )
    calculation_frame = _adjustment_consistent_ohlc(calculation_frame).sort_values(
        ["provider", "ticker", "trading_date", "__position"],
        kind="mergesort",
    )

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
    atr_values = atr(atr_input, length=atr_length).to_numpy(
        dtype="float64",
        copy=True,
    )
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
            if not np.isfinite(signal_atr) or signal_atr <= 0:
                continue

            position = positions[signal_row]
            entry_price = opens[signal_row + 1]
            initial_risk = stop_atr_multiple * signal_atr
            stop_price = entry_price - initial_risk
            take_profit_price = entry_price + reward_risk_ratio * initial_risk
            if stop_price <= 0:
                continue

            available_sessions = min(horizon, len(group) - signal_row - 1)
            event, event_session, ambiguous = _first_barrier_event(
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
            if event is None or (event == "timeout" and available_sessions < horizon):
                continue

            if event == "timeout":
                resolution_session = horizon
            else:
                assert event_session is not None
                resolution_session = event_session
            outputs[f"target_end_date_{horizon}d"][position] = dates[
                signal_row + resolution_session
            ]
            outputs[f"barrier_event_{horizon}d"][position] = event
            outputs[f"ambiguous_intrabar_{horizon}d"][position] = ambiguous

            if event == "timeout":
                outputs[f"target_tp_before_sl_{horizon}d"][position] = False
                outputs[f"time_to_event_{horizon}d"][position] = horizon
            else:
                outputs[f"event_session_{horizon}d"][position] = event_session
                outputs[f"time_to_event_{horizon}d"][position] = event_session
                if event == "take_profit":
                    outputs[f"target_tp_before_sl_{horizon}d"][position] = True
                elif event == "stop_loss":
                    outputs[f"target_tp_before_sl_{horizon}d"][position] = False


def _first_barrier_event(
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
) -> tuple[str | None, int | None, bool | None]:
    for session in range(1, sessions + 1):
        row = start + session - 1
        if not valid_ohlc[row]:
            return None, None, None
        if opens[row] <= stop_price:
            return "stop_loss", session, False
        if opens[row] >= take_profit_price:
            return "take_profit", session, False

        stop_hit = lows[row] <= stop_price
        target_hit = highs[row] >= take_profit_price
        if not stop_hit and not target_hit:
            continue
        if stop_hit and not target_hit:
            return "stop_loss", session, False
        if target_hit and not stop_hit:
            return "take_profit", session, False

        if intrabar_policy == "exclude_ambiguous":
            return "ambiguous", session, True
        if intrabar_policy == "target_first":
            return "take_profit", session, True
        if intrabar_policy == "stop_first":
            return "stop_loss", session, True
        if closes[row] > opens[row]:
            return "stop_loss", session, True
        if closes[row] < opens[row]:
            return "take_profit", session, True
        return "stop_loss", session, True

    return "timeout", None, False


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
        for prefix in (
            "barrier_event",
            "target_tp_before_sl",
            "event_session",
            "time_to_event",
            "ambiguous_intrabar",
        ):
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
        result[f"barrier_event_{horizon}d"] = pd.array(
            outputs[f"barrier_event_{horizon}d"],
            dtype="string",
        )
        result[f"target_tp_before_sl_{horizon}d"] = pd.array(
            outputs[f"target_tp_before_sl_{horizon}d"],
            dtype="boolean",
        )
        for prefix in ("event_session", "time_to_event"):
            result[f"{prefix}_{horizon}d"] = pd.array(
                outputs[f"{prefix}_{horizon}d"],
                dtype="Int64",
            )
        result[f"ambiguous_intrabar_{horizon}d"] = pd.array(
            outputs[f"ambiguous_intrabar_{horizon}d"],
            dtype="boolean",
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
    entry_price_rule: str,
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
    if entry_price_rule not in SUPPORTED_ENTRY_PRICE_RULES:
        raise ValueError(f"Unsupported entry_price_rule {entry_price_rule!r}; expected 'next_open'")
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
    missing = _missing_columns(BARRIER_REQUIRED_PRICE_COLUMNS, prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")


def _validate_unique_observations(prices: pd.DataFrame, normalized_dates: pd.Series) -> None:
    keys = pd.DataFrame(
        {
            "provider": prices["provider"],
            "ticker": prices["ticker"],
            "trading_date": normalized_dates,
        }
    )
    if keys.duplicated().any():
        raise ValueError("Duplicate provider/ticker/trading_date observations are not allowed")


def _missing_columns(required: Sequence[str], available: pd.Index) -> list[str]:
    return [column for column in required if column not in available]
