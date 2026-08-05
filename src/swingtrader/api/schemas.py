"""Pydantic request models for the replay HTTP API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from swingtrader.replay.domain import BarrierPolicy, CourtageProfileName, PositionDecisionAction


class CreateReplayRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    provider: str = "yfinance"
    tickers: list[str] = Field(min_length=1)
    start_date: date
    end_date: date
    initial_cash: float = Field(gt=0)
    courtage_profile: CourtageProfileName
    barrier_policy: BarrierPolicy


class EveningDecisionRequest(BaseModel):
    action: PositionDecisionAction
    ticker: str
    position_id: str | None = None
    quantity: int | None = Field(default=None, gt=0)
    allocation_sek: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    target_price: float | None = Field(default=None, gt=0)
    risk_label: str | None = Field(default=None, max_length=80)
    priority: int = 0
    note: str | None = None


class MorningRevisionRequest(BaseModel):
    cancelled: bool = False
    action: PositionDecisionAction | None = None
    quantity: int | None = Field(default=None, gt=0)
    allocation_sek: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    target_price: float | None = Field(default=None, gt=0)
    risk_label: str | None = Field(default=None, max_length=80)
    priority: int | None = None
    note: str | None = None

    def changes(self) -> dict[str, Any]:
        values = self.model_dump(exclude={"cancelled"}, exclude_none=True)
        if "action" in values:
            values["action"] = values["action"].value
        return values


class IndicatorInstance(BaseModel):
    indicator_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class ChartRequest(BaseModel):
    indicators: list[IndicatorInstance] = Field(default_factory=list)
    lookback_sessions: int = Field(default=180, ge=20, le=2000)


class Operand(BaseModel):
    kind: Literal["column", "indicator"]
    column: str | None = None
    indicator_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    output: str | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> Operand:
        if self.kind == "column" and not self.column:
            raise ValueError("A column operand requires column")
        if self.kind == "indicator" and (not self.indicator_id or not self.output):
            raise ValueError("An indicator operand requires indicator_id and output")
        return self


class ScreeningExpression(BaseModel):
    left: Operand
    operation: Literal["identity", "divide", "subtract", "add", "multiply"] = "identity"
    right: Operand | None = None
    lookback_sessions: int = Field(default=1, ge=1, le=252)
    aggregation: Literal["latest", "maximum", "minimum", "mean"] = "latest"


class ScreeningRule(BaseModel):
    expression: ScreeningExpression
    comparison: Literal["gt", "gte", "lt", "lte", "between", "eq"]
    value: float | bool | None = None
    minimum: float | None = None
    maximum: float | None = None


class ScreeningSort(BaseModel):
    expression: ScreeningExpression
    direction: Literal["asc", "desc"] = "desc"


class ScreeningConfiguration(BaseModel):
    name: str | None = None
    preset_id: str | None = None
    rules: list[ScreeningRule] = Field(default_factory=list)
    sort: list[ScreeningSort] = Field(default_factory=list)
    exclude_owned: bool = True
    exclude_pending_buys: bool = True


class WatchlistRequest(BaseModel):
    ticker: str
    note: str | None = None


class CreatePresetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    configuration: ScreeningConfiguration


class UpdatePresetRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    configuration: ScreeningConfiguration | None = None
