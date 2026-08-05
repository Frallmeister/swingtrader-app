"""Small domain types shared by the discretionary replay services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ReplayPhase(StrEnum):
    """Information phase currently visible to the user."""

    EVENING = "evening"
    MORNING = "morning"
    COMPLETED = "completed"


class ReplayStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class PositionDecisionAction(StrEnum):
    KEEP = "keep"
    SELL = "sell"
    REDUCE = "reduce"
    BUY = "buy"


class DecisionStatus(StrEnum):
    DRAFT = "draft"
    FINAL = "final"
    CANCELLED = "cancelled"
    EXECUTED = "executed"


class BarrierPolicy(StrEnum):
    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"
    CANDLE_PATH = "candle_path"


class CourtageProfileName(StrEnum):
    MINI = "mini"
    SMALL = "small"
    MEDIUM = "medium"
    FIXED = "fixed"


@dataclass(frozen=True)
class CourtageProfile:
    """Immutable fee rules copied into a replay at creation."""

    name: CourtageProfileName
    variable_rate: float | None
    minimum_fee_sek: float
    fixed_fee_sek: float | None = None

    def calculate(self, gross_value: float) -> float:
        if gross_value < 0:
            raise ValueError("gross_value must be non-negative")
        if self.fixed_fee_sek is not None:
            return self.fixed_fee_sek if gross_value > 0 else 0.0
        if self.variable_rate is None:
            raise ValueError("A variable courtage profile must define variable_rate")
        return max(gross_value * self.variable_rate, self.minimum_fee_sek) if gross_value else 0.0


AVANZA_COURTAGE_PROFILES: dict[CourtageProfileName, CourtageProfile] = {
    CourtageProfileName.MINI: CourtageProfile(
        CourtageProfileName.MINI, variable_rate=0.0025, minimum_fee_sek=1.0
    ),
    CourtageProfileName.SMALL: CourtageProfile(
        CourtageProfileName.SMALL, variable_rate=0.0015, minimum_fee_sek=39.0
    ),
    CourtageProfileName.MEDIUM: CourtageProfile(
        CourtageProfileName.MEDIUM, variable_rate=0.00069, minimum_fee_sek=69.0
    ),
    CourtageProfileName.FIXED: CourtageProfile(
        CourtageProfileName.FIXED,
        variable_rate=None,
        minimum_fee_sek=99.0,
        fixed_fee_sek=99.0,
    ),
}


@dataclass(frozen=True)
class DailyBar:
    trading_date: date
    open: float
    high: float
    low: float
    close: float
