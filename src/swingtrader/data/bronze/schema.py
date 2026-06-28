"""Database schema definitions for bronze market data."""

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
)

metadata = MetaData()

BRONZE_MARKET_DAILY_PRICES_TABLE = "bronze_market_daily_prices"

bronze_market_daily_prices = Table(
    BRONZE_MARKET_DAILY_PRICES_TABLE,
    metadata,
    Column("provider", String(64), primary_key=True),
    Column("ticker", String(32), primary_key=True),
    Column("trading_date", Date, primary_key=True),
    Column("open", Numeric(18, 6), nullable=True),
    Column("high", Numeric(18, 6), nullable=True),
    Column("low", Numeric(18, 6), nullable=True),
    Column("close", Numeric(18, 6), nullable=True),
    Column("adjusted_close", Numeric(18, 6), nullable=True),
    Column("volume", BigInteger, nullable=True),
    Column("dividends", Numeric(18, 6), nullable=True),
    Column("stock_splits", Numeric(18, 6), nullable=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("request_id", String(64), nullable=False),
    Index("ix_bronze_market_daily_prices_ticker_date", "ticker", "trading_date"),
)
