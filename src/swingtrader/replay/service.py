"""Application service implementing the evening/morning discretionary replay."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from swingtrader.replay.barriers import resolve_long_barriers
from swingtrader.replay.db import resolve_replay_database_engine
from swingtrader.replay.domain import (
    AVANZA_COURTAGE_PROFILES,
    BarrierPolicy,
    CourtageProfile,
    CourtageProfileName,
    DecisionStatus,
    PositionDecisionAction,
    ReplayPhase,
    ReplayStatus,
)
from swingtrader.replay.indicators import calculate_indicator, list_indicator_definitions
from swingtrader.replay.market_data import ReplayMarketData
from swingtrader.replay.metrics import calculate_performance_metrics
from swingtrader.replay.repository import ReplayRepository
from swingtrader.replay.screening import ScreeningService


class ReplayValidationError(ValueError):
    """Raised when a requested replay transition would violate its rules."""


class InsufficientCashError(ReplayValidationError):
    """Raised when a confirmed purchase cannot be paid in full."""


class ReplayService:
    def __init__(self, *, engine: Engine | None = None, database_url: str | None = None):
        self.engine = resolve_replay_database_engine(engine=engine, database_url=database_url)
        self.repository = ReplayRepository(self.engine)
        self.market = ReplayMarketData(self.engine)
        self.screening = ScreeningService()

    def create_replay(
        self,
        *,
        name: str,
        provider: str,
        tickers: list[str],
        start_date: date,
        end_date: date,
        initial_cash: float,
        courtage_profile: CourtageProfileName,
        barrier_policy: BarrierPolicy,
    ) -> dict[str, Any]:
        selected_tickers = sorted(set(tickers))
        if not selected_tickers:
            raise ReplayValidationError("At least one ticker is required")
        if initial_cash <= 0:
            raise ReplayValidationError("initial_cash must be positive")
        if start_date > end_date:
            raise ReplayValidationError("start_date must not be after end_date")
        dates = self.market.trading_dates(
            provider=provider,
            tickers=selected_tickers,
            start_date=start_date,
            end_date=end_date,
        )
        if not dates:
            raise ReplayValidationError("No market data exists in the requested replay interval")
        profile = AVANZA_COURTAGE_PROFILES[courtage_profile]
        session = self.repository.insert_session(
            {
                "name": name,
                "provider": provider,
                "tickers": selected_tickers,
                "start_date": start_date,
                "end_date": end_date,
                "current_date": dates[0],
                "phase": ReplayPhase.EVENING.value,
                "status": ReplayStatus.ACTIVE.value,
                "initial_cash": initial_cash,
                "cash": initial_cash,
                "courtage_profile": self._profile_dict(profile),
                "barrier_policy": barrier_policy.value,
            }
        )
        self.repository.add_event(
            replay_id=session["id"],
            event_type="replay_created",
            trading_date=session["current_date"],
            phase=session["phase"],
            payload={"name": name, "initial_cash": initial_cash},
        )
        self._record_metrics(session["id"])
        return self.get_state(session["id"])

    def list_replays(self) -> list[dict[str, Any]]:
        return self.repository.list_sessions()

    def get_state(self, replay_id: str) -> dict[str, Any]:
        session = self.repository.get_session(replay_id)
        positions = self.repository.positions(replay_id)
        open_positions = [position for position in positions if position["status"] == "open"]
        phase = ReplayPhase(session["phase"])
        if phase is ReplayPhase.EVENING:
            decisions = self.repository.decisions(
                replay_id,
                statuses=(DecisionStatus.DRAFT.value, DecisionStatus.FINAL.value),
                decision_date=session["current_date"],
                phase=ReplayPhase.EVENING.value,
            )
        else:
            decisions = self.repository.decisions(
                replay_id, statuses=(DecisionStatus.FINAL.value,)
            )
        decided_position_ids = {
            decision["position_id"] for decision in decisions if decision["position_id"] is not None
        }
        metrics = self.repository.metric_snapshots(replay_id)
        return {
            "session": session,
            "positions": positions,
            "pending_decisions": decisions,
            "watchlist": [
                item for item in self.repository.watchlist(replay_id) if item["status"] == "active"
            ],
            "metrics": metrics[-1] if metrics else None,
            "metric_history": metrics,
            "outstanding_position_ids": [
                position["id"]
                for position in open_positions
                if position["id"] not in decided_position_ids and phase is ReplayPhase.EVENING
            ],
            "morning_open_prices": self._morning_open_prices(session, decisions),
            "position_scatter": self._position_scatter(session, open_positions),
            "fills": self.repository.fills(replay_id),
            "events": self.repository.events(replay_id, limit=100),
        }

    def save_evening_decision(
        self,
        replay_id: str,
        *,
        action: PositionDecisionAction,
        ticker: str,
        position_id: str | None = None,
        quantity: int | None = None,
        allocation_sek: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        risk_label: str | None = None,
        priority: int = 0,
        note: str | None = None,
    ) -> dict[str, Any]:
        session = self._active_session(replay_id, ReplayPhase.EVENING)
        if ticker not in session["tickers"]:
            raise ReplayValidationError(f"Ticker {ticker} is outside this replay universe")
        if action is PositionDecisionAction.BUY:
            if position_id is not None:
                raise ReplayValidationError(
                    "A buy decision must not reference an existing position"
                )
            if any(
                position["ticker"] == ticker
                for position in self.repository.positions(replay_id, status="open")
            ):
                raise ReplayValidationError(f"An open position already exists for {ticker}")
            if quantity is None and allocation_sek is None:
                raise ReplayValidationError("A buy requires quantity or allocation_sek")
        else:
            if position_id is None:
                raise ReplayValidationError(f"{action.value} requires position_id")
            position = self.repository.get_position(position_id)
            if position["replay_id"] != replay_id or position["status"] != "open":
                raise ReplayValidationError("The position is not open in this replay")
            if position["ticker"] != ticker:
                raise ReplayValidationError("Decision ticker does not match the position")
            if action is PositionDecisionAction.REDUCE:
                if quantity is None or quantity <= 0 or quantity >= position["quantity"]:
                    raise ReplayValidationError(
                        "A reduction quantity must be between 1 and quantity - 1"
                    )

        self._cancel_replaced_decisions(
            replay_id=replay_id,
            current_date=session["current_date"],
            position_id=position_id,
            ticker=ticker,
            is_buy=action is PositionDecisionAction.BUY,
        )
        decision = self.repository.insert_decision(
            {
                "replay_id": replay_id,
                "position_id": position_id,
                "ticker": ticker,
                "decision_date": session["current_date"],
                "phase": ReplayPhase.EVENING.value,
                "action": action.value,
                "quantity": quantity,
                "allocation_sek": allocation_sek,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_label": risk_label,
                "priority": priority,
                "status": DecisionStatus.DRAFT.value,
                "note": note,
            }
        )
        self.repository.add_event(
            replay_id=replay_id,
            event_type="evening_decision_saved",
            trading_date=session["current_date"],
            phase=session["phase"],
            ticker=ticker,
            position_id=position_id,
            payload={"decision_id": decision["id"], "action": action.value},
        )
        return self.get_state(replay_id)

    def finalize_evening(self, replay_id: str) -> dict[str, Any]:
        session = self._active_session(replay_id, ReplayPhase.EVENING)
        open_positions = self.repository.positions(replay_id, status="open")
        decisions = self.repository.decisions(
            replay_id,
            statuses=(DecisionStatus.DRAFT.value, DecisionStatus.FINAL.value),
            decision_date=session["current_date"],
            phase=ReplayPhase.EVENING.value,
        )
        position_ids = {decision["position_id"] for decision in decisions}
        missing = [
            position["ticker"]
            for position in open_positions
            if position["id"] not in position_ids
        ]
        if missing:
            raise ReplayValidationError(
                "Every open position requires an explicit decision: " + ", ".join(missing)
            )
        next_date = self.market.next_trading_date(
            provider=session["provider"],
            tickers=session["tickers"],
            after=session["current_date"],
            end_date=session["end_date"],
        )
        if next_date is None:
            self.repository.update_session(
                replay_id, phase=ReplayPhase.COMPLETED.value, status=ReplayStatus.COMPLETED.value
            )
            return self.get_state(replay_id)
        for decision in decisions:
            self.repository.update_decision(decision["id"], status=DecisionStatus.FINAL.value)
        self.repository.update_session(
            replay_id, current_date=next_date, phase=ReplayPhase.MORNING.value
        )
        self.repository.add_event(
            replay_id=replay_id,
            event_type="evening_finalized",
            trading_date=session["current_date"],
            phase=ReplayPhase.EVENING.value,
            payload={"next_morning": next_date.isoformat(), "decisions": len(decisions)},
        )
        return self.get_state(replay_id)

    def revise_morning_decision(
        self, replay_id: str, decision_id: str, *, cancelled: bool = False, **changes: Any
    ) -> dict[str, Any]:
        session = self._active_session(replay_id, ReplayPhase.MORNING)
        decisions = {decision["id"]: decision for decision in self.repository.decisions(replay_id)}
        decision = decisions.get(decision_id)
        if decision is None or decision["status"] != DecisionStatus.FINAL.value:
            raise ReplayValidationError("The decision is not pending morning confirmation")

        allowed = {
            "action",
            "quantity",
            "allocation_sek",
            "stop_price",
            "target_price",
            "risk_label",
            "priority",
            "note",
        }
        update_values = {key: value for key, value in changes.items() if key in allowed}
        if not cancelled:
            self._validate_pending_decision(replay_id, {**decision, **update_values})
        update_values["status"] = (
            DecisionStatus.CANCELLED.value if cancelled else DecisionStatus.FINAL.value
        )
        self.repository.update_decision(decision_id, **update_values)
        self.repository.add_event(
            replay_id=replay_id,
            event_type="morning_decision_cancelled" if cancelled else "morning_decision_revised",
            trading_date=session["current_date"],
            phase=session["phase"],
            ticker=decision["ticker"],
            position_id=decision["position_id"],
            payload={"decision_id": decision_id, "changes": update_values},
        )
        return self.get_state(replay_id)

    def complete_morning(self, replay_id: str) -> dict[str, Any]:
        session = self._active_session(replay_id, ReplayPhase.MORNING)
        decisions = self.repository.decisions(replay_id, statuses=(DecisionStatus.FINAL.value,))
        positions = self.repository.positions(replay_id, status="open")
        position_by_id = {position["id"]: position for position in positions}
        bars = self._required_bars(session, positions, decisions)
        profile = self._profile(session)
        policy = BarrierPolicy(session["barrier_policy"])

        gap_exits: dict[str, tuple[dict[str, Any], float, str]] = {}
        for position in positions:
            resolution = resolve_long_barriers(
                bars[position["ticker"]],
                stop_price=position["stop_price"],
                target_price=position["target_price"],
                policy=policy,
            )
            if resolution and resolution.reason.startswith("gap_"):
                gap_exits[position["id"]] = (
                    position,
                    resolution.execution_price,
                    resolution.reason,
                )

        manual_sales: list[tuple[dict[str, Any], dict[str, Any], int, float, str]] = []
        retained_updates: list[tuple[str, dict[str, float]]] = []
        for decision in decisions:
            self._validate_pending_decision(replay_id, decision)
            position_id = decision["position_id"]
            if position_id is None or position_id in gap_exits:
                continue
            position = position_by_id[position_id]
            action = PositionDecisionAction(decision["action"])
            if action is PositionDecisionAction.SELL:
                manual_sales.append(
                    (
                        position,
                        decision,
                        position["quantity"],
                        bars[position["ticker"]].open,
                        "sell",
                    )
                )
                continue
            if action is PositionDecisionAction.REDUCE:
                manual_sales.append(
                    (
                        position,
                        decision,
                        int(decision["quantity"]),
                        bars[position["ticker"]].open,
                        "reduce",
                    )
                )
            if action in {PositionDecisionAction.KEEP, PositionDecisionAction.REDUCE}:
                changes = {
                    key: decision[key]
                    for key in ("stop_price", "target_price")
                    if decision[key] is not None
                }
                self._validate_retained_levels(
                    ticker=position["ticker"],
                    open_price=bars[position["ticker"]].open,
                    stop_price=changes.get("stop_price", position["stop_price"]),
                    target_price=changes.get("target_price", position["target_price"]),
                )
                if changes:
                    retained_updates.append((position_id, changes))

        cash = float(session["cash"])
        for position, price, _reason in gap_exits.values():
            cash += self._sale_net_proceeds(position["quantity"], price, profile)
        for _position, _decision, quantity, price, _reason in manual_sales:
            cash += self._sale_net_proceeds(quantity, price, profile)

        planned_buys: list[tuple[dict[str, Any], int, float, float]] = []
        buy_decisions = sorted(
            (
                decision
                for decision in decisions
                if decision["action"] == PositionDecisionAction.BUY.value
            ),
            key=lambda decision: (decision["priority"], decision["created_at"]),
        )
        for decision in buy_decisions:
            price = bars[decision["ticker"]].open
            budget = float(decision["allocation_sek"] or cash)
            quantity = int(
                decision["quantity"]
                or self._maximum_quantity(price, min(budget, cash), profile)
            )
            gross = quantity * price
            courtage = profile.calculate(gross)
            required = gross + courtage
            if quantity <= 0 or required > cash + 1e-9 or required > budget + 1e-9:
                raise InsufficientCashError(
                    f"Buy for {decision['ticker']} requires {required:.2f} SEK; "
                    f"available cash is {cash:.2f} SEK and order budget is {budget:.2f} SEK."
                )
            self._validate_entry_levels(
                ticker=decision["ticker"],
                entry_price=price,
                stop_price=decision["stop_price"],
                target_price=decision["target_price"],
            )
            planned_buys.append((decision, quantity, price, courtage))
            cash -= required

        for position, price, reason in gap_exits.values():
            self._execute_sale(
                session=session,
                position=position,
                decision=None,
                quantity=position["quantity"],
                price=price,
                reason=reason,
                profile=profile,
            )
        for position, decision, quantity, price, reason in manual_sales:
            self._execute_sale(
                session=session,
                position=self.repository.get_position(position["id"]),
                decision=decision,
                quantity=quantity,
                price=price,
                reason=reason,
                profile=profile,
            )
        for position_id, changes in retained_updates:
            if self.repository.get_position(position_id)["status"] == "open":
                self.repository.update_position(position_id, **changes)

        for decision, quantity, price, courtage in planned_buys:
            initial_risk = (
                price - decision["stop_price"] if decision["stop_price"] is not None else None
            )
            position = self.repository.insert_position(
                {
                    "replay_id": replay_id,
                    "ticker": decision["ticker"],
                    "opened_date": session["current_date"],
                    "closed_date": None,
                    "entry_price": price,
                    "quantity": quantity,
                    "initial_quantity": quantity,
                    "initial_risk_per_share": initial_risk,
                    "entry_courtage": courtage,
                    "stop_price": decision["stop_price"],
                    "target_price": decision["target_price"],
                    "realized_pnl": 0.0,
                    "realized_r": 0.0,
                    "status": "open",
                }
            )
            self.repository.insert_fill(
                {
                    "replay_id": replay_id,
                    "decision_id": decision["id"],
                    "position_id": position["id"],
                    "ticker": decision["ticker"],
                    "trading_date": session["current_date"],
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "gross_value": quantity * price,
                    "courtage": courtage,
                    "realized_pnl": None,
                    "realized_r": None,
                    "reason": "buy",
                }
            )
            self._close_watchlist_for_purchase(
                replay_id, decision["ticker"], session["current_date"]
            )
            self.repository.add_event(
                replay_id=replay_id,
                event_type="position_opened",
                trading_date=session["current_date"],
                phase=ReplayPhase.MORNING.value,
                ticker=decision["ticker"],
                position_id=position["id"],
                payload={
                    "quantity": quantity,
                    "price": price,
                    "courtage": courtage,
                    "decision_id": decision["id"],
                },
            )

        for decision in decisions:
            self.repository.update_decision(decision["id"], status=DecisionStatus.EXECUTED.value)
        self.repository.update_session(
            replay_id,
            cash=cash,
            phase=ReplayPhase.EVENING.value,
        )
        self._process_intraday_barriers(replay_id)
        self._record_metrics(replay_id)
        self.repository.add_event(
            replay_id=replay_id,
            event_type="morning_completed",
            trading_date=session["current_date"],
            phase=ReplayPhase.MORNING.value,
            payload={
                "gap_exits": len(gap_exits),
                "manual_sales": len(manual_sales),
                "buys": len(planned_buys),
            },
        )
        return self.get_state(replay_id)

    def chart(
        self,
        replay_id: str,
        ticker: str,
        indicators: list[dict[str, Any]],
        *,
        lookback_sessions: int = 180,
    ) -> dict[str, Any]:
        session = self.repository.get_session(replay_id)
        if ticker not in session["tickers"]:
            raise ReplayValidationError(f"Ticker {ticker} is outside this replay universe")
        prices, current_open = self.market.visible_prices(
            provider=session["provider"],
            ticker=ticker,
            current_date=session["current_date"],
            phase=ReplayPhase(session["phase"]),
        )
        if prices.empty:
            return {
                "ticker": ticker,
                "bars": [],
                "indicator_groups": [],
                "current_open": current_open,
            }
        single = prices.droplevel(["provider", "ticker"]).tail(lookback_sessions)
        bars = [
            {
                "time": index.date().isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume) if pd.notna(row.volume) else None,
            }
            for index, row in single.iterrows()
        ]
        groups = []
        for config in indicators:
            result = calculate_indicator(
                single,
                indicator_id=config["indicator_id"],
                parameters=config.get("parameters"),
                source=config.get("source"),
            )
            groups.append(
                {
                    "indicator_id": config["indicator_id"],
                    "parameters": config.get("parameters", {}),
                    "source": config.get("source"),
                    "outputs": {
                        column: [
                            {"time": index.date().isoformat(), "value": self._json_value(value)}
                            for index, value in result[column].items()
                            if pd.notna(value)
                        ]
                        for column in result.columns
                    },
                }
            )
        return {
            "ticker": ticker,
            "phase": session["phase"],
            "current_date": session["current_date"],
            "bars": bars,
            "indicator_groups": groups,
            "current_open": current_open,
        }

    def indicator_catalogue(self) -> list[dict[str, Any]]:
        return list_indicator_definitions()

    def run_screen(self, replay_id: str, configuration: dict[str, Any]) -> dict[str, Any]:
        session = self._active_session(replay_id, ReplayPhase.EVENING)
        prices = self.market.prices(
            provider=session["provider"],
            tickers=session["tickers"],
            end_date=session["current_date"],
        )
        excluded: set[str] = set()
        if configuration.get("exclude_owned", True):
            excluded.update(
                position["ticker"]
                for position in self.repository.positions(replay_id, status="open")
            )
        if configuration.get("exclude_pending_buys", True):
            excluded.update(
                decision["ticker"]
                for decision in self.repository.decisions(
                    replay_id,
                    statuses=(DecisionStatus.DRAFT.value, DecisionStatus.FINAL.value),
                    decision_date=session["current_date"],
                )
                if decision["action"] == PositionDecisionAction.BUY.value
            )
        results = self.screening.run(prices, configuration, excluded_tickers=excluded)
        run = self.repository.insert_screening_run(
            {
                "replay_id": replay_id,
                "preset_id": configuration.get("preset_id"),
                "trading_date": session["current_date"],
                "configuration": configuration,
                "results": results,
            }
        )
        return {"run_id": run["id"], "results": results}

    def add_watchlist(self, replay_id: str, ticker: str, note: str | None = None) -> dict[str, Any]:
        session = self.repository.get_session(replay_id)
        if ticker not in session["tickers"]:
            raise ReplayValidationError(f"Ticker {ticker} is outside this replay universe")
        self.repository.add_watchlist_item(
            replay_id=replay_id, ticker=ticker, added_date=session["current_date"], note=note
        )
        return self.get_state(replay_id)

    def remove_watchlist(
        self, replay_id: str, item_id: str, reason: str = "removed"
    ) -> dict[str, Any]:
        session = self.repository.get_session(replay_id)
        self.repository.update_watchlist_item(
            item_id,
            status="removed",
            removed_date=session["current_date"],
            removal_reason=reason,
        )
        return self.get_state(replay_id)

    def create_preset(
        self, *, name: str, configuration: dict[str, Any], description: str | None = None
    ) -> dict[str, Any]:
        return self.repository.insert_preset(
            {
                "name": name,
                "description": description,
                "configuration": configuration,
            }
        )

    def list_presets(self) -> list[dict[str, Any]]:
        return self.repository.presets()

    def update_preset(self, preset_id: str, **values: Any) -> list[dict[str, Any]]:
        self.repository.update_preset(preset_id, **values)
        return self.repository.presets()

    def delete_preset(self, preset_id: str) -> list[dict[str, Any]]:
        self.repository.delete_preset(preset_id)
        return self.repository.presets()

    def _validate_pending_decision(self, replay_id: str, decision: dict[str, Any]) -> None:
        action = PositionDecisionAction(decision["action"])
        position_id = decision["position_id"]
        if action is PositionDecisionAction.BUY:
            if position_id is not None:
                raise ReplayValidationError("A buy decision cannot reference an existing position")
            if decision.get("quantity") is None and decision.get("allocation_sek") is None:
                raise ReplayValidationError("A buy requires quantity or allocation_sek")
            return
        if position_id is None:
            raise ReplayValidationError(f"{action.value} requires position_id")
        position = self.repository.get_position(position_id)
        if position["replay_id"] != replay_id or position["status"] != "open":
            raise ReplayValidationError("The position is not open in this replay")
        if action is PositionDecisionAction.REDUCE:
            quantity = decision.get("quantity")
            if quantity is None or int(quantity) <= 0 or int(quantity) >= position["quantity"]:
                raise ReplayValidationError(
                        "A reduction quantity must be between 1 and quantity - 1"
                    )

    @staticmethod
    def _validate_entry_levels(
        *, ticker: str, entry_price: float, stop_price: float | None, target_price: float | None
    ) -> None:
        if stop_price is not None and stop_price >= entry_price:
            raise ReplayValidationError(f"The stop for {ticker} must be below its entry price")
        if target_price is not None and target_price <= entry_price:
            raise ReplayValidationError(f"The target for {ticker} must be above its entry price")

    @staticmethod
    def _validate_retained_levels(
        *, ticker: str, open_price: float, stop_price: float | None, target_price: float | None
    ) -> None:
        if stop_price is not None and stop_price >= open_price:
            raise ReplayValidationError(
                f"The revised stop for {ticker} must be below the morning open"
            )
        if target_price is not None and target_price <= open_price:
            raise ReplayValidationError(
                f"The revised target for {ticker} must be above the morning open"
            )

    @staticmethod
    def _sale_net_proceeds(quantity: int, price: float, profile: CourtageProfile) -> float:
        gross = quantity * price
        return gross - profile.calculate(gross)

    def _active_session(self, replay_id: str, required_phase: ReplayPhase) -> dict[str, Any]:
        session = self.repository.get_session(replay_id)
        if session["status"] != ReplayStatus.ACTIVE.value:
            raise ReplayValidationError("The replay is not active")
        if session["phase"] != required_phase.value:
            raise ReplayValidationError(
                (
                    f"This action requires {required_phase.value} mode; "
                    f"current mode is {session['phase']}"
                )
            )
        return session

    def _cancel_replaced_decisions(
        self,
        *,
        replay_id: str,
        current_date: date,
        position_id: str | None,
        ticker: str,
        is_buy: bool,
    ) -> None:
        existing = self.repository.decisions(
            replay_id,
            statuses=(DecisionStatus.DRAFT.value, DecisionStatus.FINAL.value),
            decision_date=current_date,
            phase=ReplayPhase.EVENING.value,
        )
        for decision in existing:
            same_target = decision["position_id"] == position_id if position_id else (
                is_buy and decision["position_id"] is None and decision["ticker"] == ticker
            )
            if same_target:
                self.repository.update_decision(
                    decision["id"], status=DecisionStatus.CANCELLED.value
                )

    def _required_bars(
        self,
        session: dict[str, Any],
        positions: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tickers = {position["ticker"] for position in positions}
        tickers.update(decision["ticker"] for decision in decisions)
        bars = {
            ticker: self.market.daily_bar(
                provider=session["provider"], ticker=ticker, trading_date=session["current_date"]
            )
            for ticker in tickers
        }
        missing = sorted(ticker for ticker, bar in bars.items() if bar is None)
        if missing:
            raise ReplayValidationError("Missing opening bar for: " + ", ".join(missing))
        return bars

    def _execute_sale(
        self,
        *,
        session: dict[str, Any],
        position: dict[str, Any],
        decision: dict[str, Any] | None,
        quantity: int,
        price: float,
        reason: str,
        profile: CourtageProfile,
    ) -> float:
        gross = quantity * price
        courtage = profile.calculate(gross)
        entry_courtage_share = position["entry_courtage"] * quantity / position["initial_quantity"]
        realized_pnl = (
            (price - position["entry_price"]) * quantity
            - courtage
            - entry_courtage_share
        )
        initial_risk_total = (
            position["initial_risk_per_share"] * position["initial_quantity"]
            if position["initial_risk_per_share"] is not None
            else None
        )
        realized_r = realized_pnl / initial_risk_total if initial_risk_total else 0.0
        remaining = position["quantity"] - quantity
        self.repository.update_position(
            position["id"],
            quantity=remaining,
            realized_pnl=position["realized_pnl"] + realized_pnl,
            realized_r=position["realized_r"] + realized_r,
            status="closed" if remaining == 0 else "open",
            closed_date=session["current_date"] if remaining == 0 else None,
        )
        self.repository.insert_fill(
            {
                "replay_id": session["id"],
                "decision_id": decision["id"] if decision else None,
                "position_id": position["id"],
                "ticker": position["ticker"],
                "trading_date": session["current_date"],
                "side": "sell",
                "quantity": quantity,
                "price": price,
                "gross_value": gross,
                "courtage": courtage,
                "realized_pnl": realized_pnl,
                "realized_r": realized_r,
                "reason": reason,
            }
        )
        self.repository.add_event(
            replay_id=session["id"],
            event_type="position_closed" if remaining == 0 else "position_reduced",
            trading_date=session["current_date"],
            phase=session["phase"],
            ticker=position["ticker"],
            position_id=position["id"],
            payload={"quantity": quantity, "price": price, "courtage": courtage, "reason": reason},
        )
        return gross - courtage

    def _process_intraday_barriers(self, replay_id: str) -> None:
        session = self.repository.get_session(replay_id)
        profile = self._profile(session)
        cash = float(session["cash"])
        changed = False
        for position in self.repository.positions(replay_id, status="open"):
            bar = self.market.daily_bar(
                provider=session["provider"],
                ticker=position["ticker"],
                trading_date=session["current_date"],
            )
            if bar is None:
                continue
            resolution = resolve_long_barriers(
                bar,
                stop_price=position["stop_price"],
                target_price=position["target_price"],
                policy=BarrierPolicy(session["barrier_policy"]),
            )
            if resolution is None or resolution.reason.startswith("gap_"):
                continue
            cash += self._execute_sale(
                session=session,
                position=position,
                decision=None,
                quantity=position["quantity"],
                price=resolution.execution_price,
                reason=resolution.reason,
                profile=profile,
            )
            changed = True
        if changed:
            self.repository.update_session(replay_id, cash=cash)

    def _record_metrics(self, replay_id: str) -> None:
        session = self.repository.get_session(replay_id)
        positions = self.repository.positions(replay_id)
        open_tickers = [
            position["ticker"] for position in positions if position["status"] == "open"
        ]
        closes = self.market.close_prices(
            provider=session["provider"],
            tickers=open_tickers,
            trading_date=session["current_date"],
        ) if open_tickers else {}
        metrics = calculate_performance_metrics(
            initial_cash=session["initial_cash"],
            cash=session["cash"],
            positions=positions,
            close_prices=closes,
            previous_snapshots=self.repository.metric_snapshots(replay_id),
        )
        self.repository.upsert_metric_snapshot(
            {"replay_id": replay_id, "trading_date": session["current_date"], **metrics}
        )

    def _morning_open_prices(
        self, session: dict[str, Any], decisions: list[dict[str, Any]]
    ) -> dict[str, float]:
        if session["phase"] != ReplayPhase.MORNING.value:
            return {}
        tickers = {decision["ticker"] for decision in decisions}
        tickers.update(
            position["ticker"]
            for position in self.repository.positions(session["id"], status="open")
        )
        prices = {}
        for ticker in tickers:
            bar = self.market.daily_bar(
                provider=session["provider"], ticker=ticker, trading_date=session["current_date"]
            )
            if bar:
                prices[ticker] = bar.open
        return prices

    def _position_scatter(
        self, session: dict[str, Any], positions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        closes = self.market.close_prices(
            provider=session["provider"],
            tickers=[position["ticker"] for position in positions],
            trading_date=session["current_date"],
        ) if positions else {}
        rows = []
        for position in positions:
            sessions = self.market.trading_dates(
                provider=session["provider"],
                tickers=[position["ticker"]],
                start_date=position["opened_date"],
                end_date=session["current_date"],
            )
            days_owned = max(len(sessions), 1)
            close = closes.get(position["ticker"], position["entry_price"])
            simple_return = close / position["entry_price"] - 1.0
            rows.append(
                {
                    "ticker": position["ticker"],
                    "days_owned": days_owned,
                    "simple_return": simple_return,
                    "annualized_simple_return": simple_return * 252.0 / days_owned,
                }
            )
        return rows

    def _close_watchlist_for_purchase(
        self, replay_id: str, ticker: str, trading_date: date
    ) -> None:
        for item in self.repository.watchlist(replay_id):
            if item["ticker"] == ticker and item["status"] == "active":
                self.repository.update_watchlist_item(
                    item["id"],
                    status="bought",
                    removed_date=trading_date,
                    removal_reason="bought",
                )

    @staticmethod
    def _maximum_quantity(price: float, budget: float, profile: CourtageProfile) -> int:
        quantity = math.floor(budget / price)
        while quantity > 0:
            gross = quantity * price
            if gross + profile.calculate(gross) <= budget + 1e-9:
                return quantity
            quantity -= 1
        return 0

    @staticmethod
    def _profile(session: dict[str, Any]) -> CourtageProfile:
        value = session["courtage_profile"]
        return CourtageProfile(
            name=CourtageProfileName(value["name"]),
            variable_rate=value["variable_rate"],
            minimum_fee_sek=value["minimum_fee_sek"],
            fixed_fee_sek=value["fixed_fee_sek"],
        )

    @staticmethod
    def _profile_dict(profile: CourtageProfile) -> dict[str, Any]:
        return {
            "name": profile.name.value,
            "variable_rate": profile.variable_rate,
            "minimum_fee_sek": profile.minimum_fee_sek,
            "fixed_fee_sek": profile.fixed_fee_sek,
        }

    @staticmethod
    def _json_value(value: Any) -> float | int | bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        return float(value)
