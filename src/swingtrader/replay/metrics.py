"""Portfolio metrics derived from replay positions and daily equity snapshots."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

import numpy as np


def calculate_performance_metrics(
    *,
    initial_cash: float,
    cash: float,
    positions: list[dict[str, Any]],
    close_prices: dict[str, float],
    previous_snapshots: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    open_positions = [position for position in positions if position["status"] == "open"]
    closed_positions = [position for position in positions if position["status"] == "closed"]
    market_value = sum(
        position["quantity"] * close_prices.get(position["ticker"], position["entry_price"])
        for position in open_positions
    )
    equity = cash + market_value
    realized_rs = [float(position["realized_r"]) for position in closed_positions]
    realized_pnls = [float(position["realized_pnl"]) for position in closed_positions]

    expectancy_r = fmean(realized_rs) if realized_rs else None
    win_rate = (
        sum(value > 0 for value in realized_pnls) / len(realized_pnls) if realized_pnls else None
    )
    cumulative_r = sum(realized_rs)
    total_return = equity / initial_cash - 1.0

    equities = [float(snapshot["equity"]) for snapshot in previous_snapshots]
    if not equities or not math.isclose(equities[-1], equity):
        equities.append(equity)
    returns = np.asarray(
        [
            current / previous - 1.0
            for previous, current in zip(equities, equities[1:], strict=False)
        ],
        dtype=float,
    )
    sharpe_ratio = None
    sortino_ratio = None
    if returns.size >= 2:
        volatility = float(np.std(returns, ddof=1))
        if volatility > 0:
            sharpe_ratio = float(np.mean(returns) / volatility * np.sqrt(252.0))
        downside_returns = np.minimum(returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))
        if downside_deviation > 0:
            sortino_ratio = float(np.mean(returns) / downside_deviation * np.sqrt(252.0))

    return {
        "cash": cash,
        "market_value": market_value,
        "equity": equity,
        "total_return": total_return,
        "expectancy_r": expectancy_r,
        "win_rate": win_rate,
        "cumulative_r": cumulative_r,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "closed_positions": len(closed_positions),
    }
