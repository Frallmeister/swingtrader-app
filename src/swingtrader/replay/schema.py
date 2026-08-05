"""SQLAlchemy tables for resumable replay sessions and their audit history."""

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

replay_sessions = Table(
    "replay_sessions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("provider", String(64), nullable=False),
    Column("tickers", JSON, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("current_date", Date, nullable=False),
    Column("phase", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("initial_cash", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("courtage_profile", JSON, nullable=False),
    Column("barrier_policy", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

replay_positions = Table(
    "replay_positions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(32), nullable=False),
    Column("opened_date", Date, nullable=False),
    Column("closed_date", Date),
    Column("entry_price", Float, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("initial_quantity", Integer, nullable=False),
    Column("initial_risk_per_share", Float),
    Column("entry_courtage", Float, nullable=False, default=0.0),
    Column("stop_price", Float),
    Column("target_price", Float),
    Column("realized_pnl", Float, nullable=False, default=0.0),
    Column("realized_r", Float, nullable=False, default=0.0),
    Column("status", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("ix_replay_positions_replay_status", replay_positions.c.replay_id, replay_positions.c.status)

replay_decisions = Table(
    "replay_decisions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("position_id", ForeignKey("replay_positions.id", ondelete="SET NULL")),
    Column("ticker", String(32), nullable=False),
    Column("decision_date", Date, nullable=False),
    Column("phase", String(16), nullable=False),
    Column("action", String(16), nullable=False),
    Column("quantity", Integer),
    Column("allocation_sek", Float),
    Column("stop_price", Float),
    Column("target_price", Float),
    Column("risk_label", String(80)),
    Column("priority", Integer, nullable=False, default=0),
    Column("status", String(16), nullable=False),
    Column("note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("ix_replay_decisions_pending", replay_decisions.c.replay_id, replay_decisions.c.status)

replay_fills = Table(
    "replay_fills",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("decision_id", ForeignKey("replay_decisions.id", ondelete="SET NULL")),
    Column("position_id", ForeignKey("replay_positions.id", ondelete="SET NULL")),
    Column("ticker", String(32), nullable=False),
    Column("trading_date", Date, nullable=False),
    Column("side", String(8), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("price", Float, nullable=False),
    Column("gross_value", Float, nullable=False),
    Column("courtage", Float, nullable=False),
    Column("realized_pnl", Float),
    Column("realized_r", Float),
    Column("reason", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("ix_replay_fills_replay_date", replay_fills.c.replay_id, replay_fills.c.trading_date)

replay_watchlist_items = Table(
    "replay_watchlist_items",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(32), nullable=False),
    Column("added_date", Date, nullable=False),
    Column("status", String(16), nullable=False),
    Column("note", Text),
    Column("removed_date", Date),
    Column("removal_reason", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "ix_replay_watchlist_replay_ticker",
    replay_watchlist_items.c.replay_id,
    replay_watchlist_items.c.ticker,
)

screening_presets = Table(
    "screening_presets",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("description", Text),
    Column("configuration", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

replay_screening_runs = Table(
    "replay_screening_runs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("preset_id", ForeignKey("screening_presets.id", ondelete="SET NULL")),
    Column("trading_date", Date, nullable=False),
    Column("configuration", JSON, nullable=False),
    Column("results", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

replay_metric_snapshots = Table(
    "replay_metric_snapshots",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("trading_date", Date, nullable=False),
    Column("cash", Float, nullable=False),
    Column("market_value", Float, nullable=False),
    Column("equity", Float, nullable=False),
    Column("total_return", Float, nullable=False),
    Column("expectancy_r", Float),
    Column("win_rate", Float),
    Column("cumulative_r", Float, nullable=False),
    Column("sharpe_ratio", Float),
    Column("sortino_ratio", Float),
    Column("closed_positions", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "ux_replay_metric_date",
    replay_metric_snapshots.c.replay_id,
    replay_metric_snapshots.c.trading_date,
    unique=True,
)

replay_events = Table(
    "replay_events",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("replay_id", ForeignKey("replay_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("sequence_number", Integer, nullable=False),
    Column("event_type", String(64), nullable=False),
    Column("trading_date", Date, nullable=False),
    Column("phase", String(16), nullable=False),
    Column("ticker", String(32)),
    Column("position_id", String(36)),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index(
    "ux_replay_event_sequence",
    replay_events.c.replay_id,
    replay_events.c.sequence_number,
    unique=True,
)
