"""Straightforward SQLAlchemy Core persistence for discretionary replays."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Engine

from swingtrader.replay import schema


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class ReplayRepository:
    """Persist current replay state plus an immutable audit trail."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def insert_session(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        row = {"id": new_id(), "created_at": now, "updated_at": now, **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_sessions).values(**row))
        return row

    def get_session(self, replay_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(schema.replay_sessions).where(schema.replay_sessions.c.id == replay_id)
            ).mappings().one_or_none()
        if row is None:
            raise KeyError(f"Unknown replay: {replay_id}")
        return dict(row)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.replay_sessions).order_by(schema.replay_sessions.c.updated_at.desc())
            ).mappings()
            return [dict(row) for row in rows]

    def update_session(self, replay_id: str, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(schema.replay_sessions)
                .where(schema.replay_sessions.c.id == replay_id)
                .values(updated_at=utc_now(), **values)
            )

    def insert_position(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        row = {"id": new_id(), "created_at": now, "updated_at": now, **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_positions).values(**row))
        return row

    def positions(self, replay_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        statement = select(schema.replay_positions).where(
            schema.replay_positions.c.replay_id == replay_id
        )
        if status is not None:
            statement = statement.where(schema.replay_positions.c.status == status)
        statement = statement.order_by(schema.replay_positions.c.created_at)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def get_position(self, position_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(schema.replay_positions).where(schema.replay_positions.c.id == position_id)
            ).mappings().one_or_none()
        if row is None:
            raise KeyError(f"Unknown position: {position_id}")
        return dict(row)

    def update_position(self, position_id: str, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(schema.replay_positions)
                .where(schema.replay_positions.c.id == position_id)
                .values(updated_at=utc_now(), **values)
            )

    def insert_decision(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        row = {"id": new_id(), "created_at": now, "updated_at": now, **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_decisions).values(**row))
        return row

    def decisions(
        self,
        replay_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
        decision_date: date | None = None,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        statement = select(schema.replay_decisions).where(
            schema.replay_decisions.c.replay_id == replay_id
        )
        if statuses:
            statement = statement.where(schema.replay_decisions.c.status.in_(statuses))
        if decision_date:
            statement = statement.where(schema.replay_decisions.c.decision_date == decision_date)
        if phase:
            statement = statement.where(schema.replay_decisions.c.phase == phase)
        statement = statement.order_by(
            schema.replay_decisions.c.priority, schema.replay_decisions.c.created_at
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def update_decision(self, decision_id: str, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(schema.replay_decisions)
                .where(schema.replay_decisions.c.id == decision_id)
                .values(updated_at=utc_now(), **values)
            )

    def insert_fill(self, values: dict[str, Any]) -> dict[str, Any]:
        row = {"id": new_id(), "created_at": utc_now(), **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_fills).values(**row))
        return row

    def fills(self, replay_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.replay_fills)
                .where(schema.replay_fills.c.replay_id == replay_id)
                .order_by(schema.replay_fills.c.trading_date, schema.replay_fills.c.created_at)
            ).mappings()
            return [dict(row) for row in rows]

    def add_event(
        self,
        *,
        replay_id: str,
        event_type: str,
        trading_date: date,
        phase: str,
        payload: dict[str, Any],
        ticker: str | None = None,
        position_id: str | None = None,
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            next_sequence = connection.execute(
                select(
                    func.coalesce(func.max(schema.replay_events.c.sequence_number), 0) + 1
                ).where(
                    schema.replay_events.c.replay_id == replay_id
                )
            ).scalar_one()
            row = {
                "id": new_id(),
                "replay_id": replay_id,
                "sequence_number": next_sequence,
                "event_type": event_type,
                "trading_date": trading_date,
                "phase": phase,
                "ticker": ticker,
                "position_id": position_id,
                "payload": payload,
                "created_at": utc_now(),
            }
            connection.execute(insert(schema.replay_events).values(**row))
        return row

    def events(self, replay_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.replay_events)
                .where(schema.replay_events.c.replay_id == replay_id)
                .order_by(schema.replay_events.c.sequence_number.desc())
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def add_watchlist_item(
        self, *, replay_id: str, ticker: str, added_date: date, note: str | None = None
    ) -> dict[str, Any]:
        active = next(
            (
                item
                for item in self.watchlist(replay_id)
                if item["ticker"] == ticker and item["status"] == "active"
            ),
            None,
        )
        if active:
            return active
        now = utc_now()
        row = {
            "id": new_id(),
            "replay_id": replay_id,
            "ticker": ticker,
            "added_date": added_date,
            "status": "active",
            "note": note,
            "removed_date": None,
            "removal_reason": None,
            "created_at": now,
            "updated_at": now,
        }
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_watchlist_items).values(**row))
        return row

    def watchlist(self, replay_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.replay_watchlist_items)
                .where(schema.replay_watchlist_items.c.replay_id == replay_id)
                .order_by(schema.replay_watchlist_items.c.added_date)
            ).mappings()
            return [dict(row) for row in rows]

    def update_watchlist_item(self, item_id: str, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(schema.replay_watchlist_items)
                .where(schema.replay_watchlist_items.c.id == item_id)
                .values(updated_at=utc_now(), **values)
            )

    def insert_preset(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        row = {"id": new_id(), "created_at": now, "updated_at": now, **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.screening_presets).values(**row))
        return row

    def presets(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.screening_presets).order_by(schema.screening_presets.c.name)
            ).mappings()
            return [dict(row) for row in rows]

    def update_preset(self, preset_id: str, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(schema.screening_presets)
                .where(schema.screening_presets.c.id == preset_id)
                .values(updated_at=utc_now(), **values)
            )

    def delete_preset(self, preset_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                delete(schema.screening_presets).where(schema.screening_presets.c.id == preset_id)
            )

    def insert_screening_run(self, values: dict[str, Any]) -> dict[str, Any]:
        row = {"id": new_id(), "created_at": utc_now(), **values}
        with self.engine.begin() as connection:
            connection.execute(insert(schema.replay_screening_runs).values(**row))
        return row

    def upsert_metric_snapshot(self, values: dict[str, Any]) -> dict[str, Any]:
        replay_id = values["replay_id"]
        trading_date = values["trading_date"]
        with self.engine.begin() as connection:
            existing_id = connection.execute(
                select(schema.replay_metric_snapshots.c.id).where(
                    schema.replay_metric_snapshots.c.replay_id == replay_id,
                    schema.replay_metric_snapshots.c.trading_date == trading_date,
                )
            ).scalar_one_or_none()
            if existing_id:
                connection.execute(
                    update(schema.replay_metric_snapshots)
                    .where(schema.replay_metric_snapshots.c.id == existing_id)
                    .values(**values, created_at=utc_now())
                )
                return {"id": existing_id, **values}
            row = {"id": new_id(), "created_at": utc_now(), **values}
            connection.execute(insert(schema.replay_metric_snapshots).values(**row))
            return row

    def metric_snapshots(self, replay_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.replay_metric_snapshots)
                .where(schema.replay_metric_snapshots.c.replay_id == replay_id)
                .order_by(schema.replay_metric_snapshots.c.trading_date)
            ).mappings()
            return [dict(row) for row in rows]
