"""Resolve daily stop-loss and take-profit hits without intraday prices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from swingtrader.replay.domain import BarrierPolicy, DailyBar


class BarrierExit(StrEnum):
    STOP = "stop"
    TARGET = "target"


@dataclass(frozen=True)
class BarrierResolution:
    exit: BarrierExit
    execution_price: float
    reason: str


def resolve_long_barriers(
    bar: DailyBar,
    *,
    stop_price: float | None,
    target_price: float | None,
    policy: BarrierPolicy,
) -> BarrierResolution | None:
    """Resolve long-position barriers, checking observable opening gaps first.

    The candle-path policy intentionally matches the existing triple-barrier target:
    a bearish candle follows open -> high -> low -> close, while a non-bearish candle
    follows open -> low -> high -> close.
    """
    if stop_price is not None and bar.open <= stop_price:
        return BarrierResolution(BarrierExit.STOP, bar.open, "gap_through_stop")
    if target_price is not None and bar.open >= target_price:
        return BarrierResolution(BarrierExit.TARGET, bar.open, "gap_through_target")

    stop_hit = stop_price is not None and bar.low <= stop_price
    target_hit = target_price is not None and bar.high >= target_price
    if stop_hit and not target_hit:
        return BarrierResolution(BarrierExit.STOP, float(stop_price), "intraday_stop")
    if target_hit and not stop_hit:
        return BarrierResolution(BarrierExit.TARGET, float(target_price), "intraday_target")
    if not stop_hit and not target_hit:
        return None

    if policy is BarrierPolicy.TARGET_FIRST:
        return BarrierResolution(BarrierExit.TARGET, float(target_price), "ambiguous_target_first")
    if policy is BarrierPolicy.STOP_FIRST:
        return BarrierResolution(BarrierExit.STOP, float(stop_price), "ambiguous_stop_first")
    if bar.close < bar.open:
        return BarrierResolution(BarrierExit.TARGET, float(target_price), "ambiguous_candle_path")
    return BarrierResolution(BarrierExit.STOP, float(stop_price), "ambiguous_candle_path")
