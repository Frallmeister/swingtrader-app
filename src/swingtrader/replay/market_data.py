"""Bounded market-data access used by replay and indicator services."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
from sqlalchemy import distinct, select
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.loaders import load_bronze_daily_prices
from swingtrader.data.bronze.schema import bronze_market_daily_prices
from swingtrader.replay.domain import DailyBar, ReplayPhase


class ReplayMarketData:
    def __init__(self, engine: Engine):
        self.engine = engine

    def trading_dates(
        self, *, provider: str, tickers: Sequence[str], start_date: date, end_date: date
    ) -> list[date]:
        statement = (
            select(distinct(bronze_market_daily_prices.c.trading_date))
            .where(bronze_market_daily_prices.c.provider == provider)
            .where(bronze_market_daily_prices.c.ticker.in_(list(tickers)))
            .where(bronze_market_daily_prices.c.trading_date >= start_date)
            .where(bronze_market_daily_prices.c.trading_date <= end_date)
            .order_by(bronze_market_daily_prices.c.trading_date)
        )
        with self.engine.connect() as connection:
            return list(connection.execute(statement).scalars())

    def next_trading_date(
        self, *, provider: str, tickers: Sequence[str], after: date, end_date: date
    ) -> date | None:
        dates = self.trading_dates(
            provider=provider, tickers=tickers, start_date=after, end_date=end_date
        )
        return next((trading_date for trading_date in dates if trading_date > after), None)

    def prices(
        self,
        *,
        provider: str,
        tickers: Sequence[str],
        end_date: date,
        start_date: date | None = None,
    ) -> pd.DataFrame:
        frame = load_bronze_daily_prices(
            engine=self.engine,
            provider=provider,
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            columns=["open", "high", "low", "close", "adjusted_close", "volume"],
        )
        if frame.empty:
            return frame
        return frame.set_index(["provider", "ticker", "trading_date"]).sort_index()

    def visible_prices(
        self,
        *,
        provider: str,
        ticker: str,
        current_date: date,
        phase: ReplayPhase,
        start_date: date | None = None,
    ) -> tuple[pd.DataFrame, float | None]:
        prices = self.prices(
            provider=provider,
            tickers=[ticker],
            start_date=start_date,
            end_date=current_date,
        )
        if prices.empty:
            return prices, None
        current_key = (provider, ticker, pd.Timestamp(current_date))
        current_open: float | None = None
        if phase is ReplayPhase.MORNING and current_key in prices.index:
            current_open = float(prices.loc[current_key, "open"])
            prices = prices.loc[prices.index.get_level_values("trading_date").date < current_date]
        return prices, current_open

    def daily_bar(self, *, provider: str, ticker: str, trading_date: date) -> DailyBar | None:
        frame = load_bronze_daily_prices(
            engine=self.engine,
            provider=provider,
            tickers=ticker,
            start_date=trading_date,
            end_date=trading_date,
            columns=["open", "high", "low", "close"],
        )
        if frame.empty:
            return None
        row = frame.iloc[0]
        if row[["open", "high", "low", "close"]].isna().any():
            return None
        return DailyBar(
            trading_date=trading_date,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )

    def close_prices(
        self, *, provider: str, tickers: Sequence[str], trading_date: date
    ) -> dict[str, float]:
        frame = load_bronze_daily_prices(
            engine=self.engine,
            provider=provider,
            tickers=list(tickers),
            start_date=trading_date,
            end_date=trading_date,
            columns=["close"],
        )
        return {
            str(row.ticker): float(row.close)
            for row in frame.itertuples(index=False)
            if pd.notna(row.close)
        }
