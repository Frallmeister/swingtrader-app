"""Minimal daily-bar backtesting pilot for ranked long-entry signals."""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import validate_market_price_index, validate_required_columns

_PRICE_COLUMNS = ("open", "high", "low", "close")
_TRADE_COLUMNS = (
    "provider",
    "ticker",
    "signal_date",
    "entry_date",
    "exit_date",
    "score",
    "quantity",
    "entry_price",
    "exit_price",
    "stop_price",
    "take_profit_price",
    "initial_risk",
    "net_pnl",
    "entry_commission",
    "exit_commission",
    "reward_risk",
    "holding_sessions",
    "exit_reason",
    "ambiguous_intrabar",
)


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_cash: float,
    risk_fraction: float,
    max_positions: int,
    max_holding_sessions: int,
    minimum_score: float | None = None,
    stop_atr_multiple: float = 1.0,
    reward_risk_ratio: float = 2.0,
    commission_rate: float = 0.0025,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Simulate ranked long entries using raw daily OHLC prices.

    ``signals`` must contain ``score`` and ``atr`` values known after each signal
    session's close. ``atr`` must be calculated from raw OHLC prices. Entries and
    timeout exits execute at the next session open. Stops and targets are fixed
    from the entry open and signal-session ATR. Same-bar stop/target touches are
    resolved conservatively as stop losses.

    The result contains a transaction table, a compact daily equity table, and
    the requested performance summary. Remaining positions are closed at the
    final available close for reporting.
    """
    _validate_inputs(
        prices,
        signals,
        initial_cash=initial_cash,
        risk_fraction=risk_fraction,
        max_positions=max_positions,
        max_holding_sessions=max_holding_sessions,
        minimum_score=minimum_score,
        stop_atr_multiple=stop_atr_multiple,
        reward_risk_ratio=reward_risk_ratio,
        commission_rate=commission_rate,
    )
    prices = prices.loc[:, _PRICE_COLUMNS].astype("float64")
    signals = signals.loc[:, ["score", "atr"]].astype("float64")
    trading_dates = pd.DatetimeIndex(
        prices.index.get_level_values("trading_date").unique()
    ).sort_values()
    next_date = dict(zip(trading_dates[:-1], trading_dates[1:], strict=True))

    cash = float(initial_cash)
    positions: dict[tuple[str, str], dict[str, object]] = {}
    entries_by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    timeouts_by_date: dict[pd.Timestamp, list[tuple[str, str]]] = {}
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    def close_position(
        key: tuple[str, str],
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        ambiguous_intrabar: bool = False,
    ) -> None:
        nonlocal cash
        position = positions.pop(key)
        quantity = int(position["quantity"])
        entry_price = float(position["entry_price"])
        exit_value = quantity * exit_price
        exit_commission = exit_value * commission_rate
        net_pnl = (
            exit_value
            - quantity * entry_price
            - float(position["entry_commission"])
            - exit_commission
        )
        initial_risk = quantity * float(position["atr_distance"])
        trades.append(
            {
                "provider": position["provider"],
                "ticker": position["ticker"],
                "signal_date": position["signal_date"],
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "score": position["score"],
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": position["stop_price"],
                "take_profit_price": position["take_profit_price"],
                "initial_risk": initial_risk,
                "net_pnl": net_pnl,
                "entry_commission": position["entry_commission"],
                "exit_commission": exit_commission,
                "reward_risk": net_pnl / initial_risk,
                "holding_sessions": position["holding_sessions"],
                "exit_reason": exit_reason,
                "ambiguous_intrabar": ambiguous_intrabar,
            }
        )
        cash += exit_value - exit_commission

    for trading_date in trading_dates:
        bars = _rows_for_date(prices, trading_date)

        # Manual timeout exits and opening-gap exits happen before new entries.
        for key in timeouts_by_date.pop(trading_date, []):
            if key not in positions:
                continue
            if key not in bars:
                following_date = next_date.get(trading_date)
                if following_date is not None:
                    timeouts_by_date.setdefault(following_date, []).append(key)
                continue
            close_position(key, trading_date, float(bars[key]["open"]), "timeout")

        for key, position in list(positions.items()):
            if key not in bars:
                continue
            opening_price = float(bars[key]["open"])
            if opening_price <= float(position["stop_price"]):
                reason = "stop_loss"
            elif opening_price >= float(position["take_profit_price"]):
                reason = "take_profit"
            else:
                continue
            close_position(key, trading_date, opening_price, reason)

        for order in entries_by_date.pop(trading_date, []):
            key = (str(order["provider"]), str(order["ticker"]))
            if key in positions or key not in bars or len(positions) >= max_positions:
                continue
            entry_price = float(bars[key]["open"])
            atr_distance = float(order["atr_distance"])
            stop_price = entry_price - atr_distance
            affordable = floor(cash / (entry_price * (1.0 + commission_rate)))
            quantity = min(int(order["planned_quantity"]), affordable)
            if stop_price <= 0 or quantity <= 0:
                continue
            entry_commission = quantity * entry_price * commission_rate
            cash -= quantity * entry_price + entry_commission
            positions[key] = {
                **order,
                "entry_date": trading_date,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_price": stop_price,
                "take_profit_price": entry_price + reward_risk_ratio * atr_distance,
                "entry_commission": entry_commission,
                "holding_sessions": 0,
                "last_close": entry_price,
            }

        # Evaluate the full daily range, including the entry session.
        for key, position in list(positions.items()):
            if key not in bars:
                continue
            bar = bars[key]
            position["holding_sessions"] = int(position["holding_sessions"]) + 1
            position["last_close"] = float(bar["close"])
            stop_hit = float(bar["low"]) <= float(position["stop_price"])
            target_hit = float(bar["high"]) >= float(position["take_profit_price"])
            if stop_hit or target_hit:
                ambiguous = stop_hit and target_hit
                reason = "stop_loss" if stop_hit else "take_profit"
                exit_price = (
                    float(position["stop_price"])
                    if stop_hit
                    else float(position["take_profit_price"])
                )
                close_position(key, trading_date, exit_price, reason, ambiguous)
            elif int(position["holding_sessions"]) >= max_holding_sessions:
                following_date = next_date.get(trading_date)
                if following_date is not None:
                    timeouts_by_date.setdefault(following_date, []).append(key)

        market_value = sum(
            int(position["quantity"]) * float(position["last_close"])
            for position in positions.values()
        )
        equity = cash + market_value

        following_date = next_date.get(trading_date)
        if following_date is not None:
            timeout_keys = set(timeouts_by_date.get(following_date, []))
            occupied_next_open = (
                len(positions) - len(timeout_keys) + len(entries_by_date.get(following_date, []))
            )
            available_slots = max_positions - occupied_next_open
            pending_tickers = {
                (str(order["provider"]), str(order["ticker"]))
                for orders in entries_by_date.values()
                for order in orders
            }
            for provider, ticker, score, atr_value in _signals_for_date(
                signals,
                trading_date,
                minimum_score=minimum_score,
            ):
                key = (provider, ticker)
                if available_slots <= 0:
                    break
                if key in positions or key in pending_tickers:
                    continue
                if (provider, ticker, following_date) not in prices.index:
                    continue
                atr_distance = stop_atr_multiple * atr_value
                planned_quantity = floor(equity * risk_fraction / atr_distance)
                if planned_quantity <= 0:
                    continue
                entries_by_date.setdefault(following_date, []).append(
                    {
                        "provider": provider,
                        "ticker": ticker,
                        "signal_date": trading_date,
                        "score": score,
                        "atr_distance": atr_distance,
                        "planned_quantity": planned_quantity,
                    }
                )
                pending_tickers.add(key)
                available_slots -= 1

        equity_rows.append(
            {
                "trading_date": trading_date,
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "open_positions": len(positions),
            }
        )

    final_date = trading_dates[-1]
    final_bars = _rows_for_date(prices, final_date)
    for key, position in list(positions.items()):
        exit_price = (
            float(final_bars[key]["close"]) if key in final_bars else float(position["last_close"])
        )
        close_position(key, final_date, exit_price, "end_of_data")
    equity_rows[-1].update(cash=cash, market_value=0.0, equity=cash, open_positions=0)

    trades_frame = pd.DataFrame(trades, columns=_TRADE_COLUMNS)
    equity_frame = pd.DataFrame(equity_rows).set_index("trading_date")
    return {
        "trades": trades_frame,
        "equity": equity_frame,
        "summary": summarize_backtest(
            trades_frame,
            initial_cash=initial_cash,
            final_equity=float(equity_frame.iloc[-1]["equity"]),
        ),
    }


def summarize_backtest(
    trades: pd.DataFrame,
    *,
    initial_cash: float,
    final_equity: float,
) -> pd.Series:
    """Return the small metric set selected for the backtesting pilot."""
    if trades.empty:
        return pd.Series(
            {
                "win_rate": np.nan,
                "average_reward_risk_win": np.nan,
                "average_reward_risk_loss": np.nan,
                "total_reward_risk": 0.0,
                "total_return": final_equity / initial_cash - 1.0,
                "expectancy": np.nan,
                "average_holding_sessions": np.nan,
                "commissions_paid": 0.0,
            },
            dtype="float64",
        )

    winners = trades.loc[trades["net_pnl"] > 0]
    losers = trades.loc[trades["net_pnl"] < 0]
    return pd.Series(
        {
            "win_rate": len(winners) / len(trades),
            "average_reward_risk_win": winners["reward_risk"].mean(),
            "average_reward_risk_loss": losers["reward_risk"].mean(),
            "total_reward_risk": trades["reward_risk"].sum(),
            "total_return": final_equity / initial_cash - 1.0,
            "expectancy": trades["reward_risk"].mean(),
            "average_holding_sessions": trades["holding_sessions"].mean(),
            "commissions_paid": (
                trades["entry_commission"].sum() + trades["exit_commission"].sum()
            ),
        },
        dtype="float64",
    )


def _rows_for_date(
    prices: pd.DataFrame,
    trading_date: pd.Timestamp,
) -> dict[tuple[str, str], pd.Series]:
    rows = prices.xs(trading_date, level="trading_date")
    return {(str(provider), str(ticker)): row for (provider, ticker), row in rows.iterrows()}


def _signals_for_date(
    signals: pd.DataFrame,
    trading_date: pd.Timestamp,
    *,
    minimum_score: float | None = None,
) -> list[tuple[str, str, float, float]]:
    try:
        rows = signals.xs(trading_date, level="trading_date", drop_level=False)
    except KeyError:
        return []

    rows = rows.reset_index()
    valid = np.isfinite(rows["score"]) & np.isfinite(rows["atr"]) & rows["atr"].gt(0)

    if minimum_score is not None:
        valid = valid & rows["score"].ge(minimum_score)

    rows = rows.loc[valid]
    rows = rows.sort_values(
        ["score", "provider", "ticker"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    return [
        (str(row.provider), str(row.ticker), float(row.score), float(row.atr))
        for row in rows.itertuples(index=False)
    ]


def _validate_inputs(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_cash: float,
    risk_fraction: float,
    max_positions: int,
    max_holding_sessions: int,
    minimum_score: float | None,
    stop_atr_multiple: float,
    reward_risk_ratio: float,
    commission_rate: float,
) -> None:
    validate_market_price_index(prices)
    validate_market_price_index(signals)
    validate_required_columns(prices, required_columns=_PRICE_COLUMNS)
    validate_required_columns(signals, required_columns=("score", "atr"))
    if prices.empty:
        raise ValueError("prices must not be empty")
    if not signals.index.difference(prices.index).empty:
        raise ValueError("Every signal row must match a raw price row.")

    if minimum_score is not None and (
        isinstance(minimum_score, bool) or not np.isfinite(minimum_score)
    ):
        raise ValueError("minimum_score must be None or a finite number")

    numeric_ohlc = prices.loc[:, _PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric_ohlc.to_numpy(dtype="float64")
    if len(values):
        opens, highs, lows, closes = values.T
        valid = (
            np.isfinite(values).all(axis=1)
            & (values > 0).all(axis=1)
            & (highs >= np.maximum(opens, closes))
            & (lows <= np.minimum(opens, closes))
        )
        if not valid.all():
            index = prices.index[int(np.flatnonzero(~valid)[0])]
            raise ValueError(f"Invalid raw OHLC row at {index!r}.")

    positive_numbers = {
        "initial_cash": initial_cash,
        "risk_fraction": risk_fraction,
        "stop_atr_multiple": stop_atr_multiple,
    }
    for name, value in positive_numbers.items():
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if risk_fraction > 1:
        raise ValueError("risk_fraction must not exceed 1")
    for name, value in {
        "max_positions": max_positions,
        "max_holding_sessions": max_holding_sessions,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(reward_risk_ratio) or reward_risk_ratio < 2:
        raise ValueError("reward_risk_ratio must be at least 2")
    if not np.isfinite(commission_rate) or commission_rate < 0:
        raise ValueError("commission_rate must be a finite non-negative number")
