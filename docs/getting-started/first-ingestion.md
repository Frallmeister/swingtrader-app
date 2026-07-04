# First Ingestion

This page shows the first useful local data workflows: bronze onboarding for first rows, the runnable daily market data job for ongoing updates, and the lower-level library functions they wrap.

## Database

By default, local ingestion uses SQLite at `data/swingtrader.sqlite`. Override this with the `SWINGTRADER_DATABASE_URL` environment variable when needed.

Example PowerShell override:

```powershell
$env:SWINGTRADER_DATABASE_URL = "sqlite+pysqlite:///data/swingtrader.sqlite"
```

## Bronze Onboarding Check

The onboarding workflow compares active tickers with the bronze daily price table. A ticker is considered onboarded once any bronze daily price row exists for that provider. Use `backfill=True` to create first bronze rows for missing active tickers:

```python
from datetime import date

from swingtrader.data.ingestion.onboarding import sync_active_ticker_bronze_onboarding

result = sync_active_ticker_bronze_onboarding(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1),
    backfill=True,
)

print(result.backfill_tickers)
```

This workflow is bronze-only. It does not decide whether a ticker has enough data for inference or model training.

## Daily Market Data Job

The daily market data job is the preferred local entrypoint for updating bronze daily market prices for the active trading universe:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3
```

The job reads active tickers, derives per-ticker update plans from existing bronze rows, and calls the historical ingestion library for each ticker.

Tickers with no bronze rows are reported as not onboarded and skipped by the daily update job. Run bronze onboarding first for newly active tickers.

Use an explicit exclusive end date for deterministic local runs:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3 --end-date 2026-07-04
```

## Historical Daily Prices Library

Historical daily market data ingestion resolves active tickers when no explicit ticker list is provided, downloads yfinance daily prices, and ingests the normalized records into the bronze layer by upserting rows into `bronze_market_daily_prices`.

The database engine is created through `create_database_engine()`. For PostgreSQL, set `SWINGTRADER_DATABASE_URL` to a SQLAlchemy PostgreSQL URL and use the same ingestion functions:

```powershell
$env:SWINGTRADER_DATABASE_URL = "postgresql+psycopg://user:password@host:5432/database"
```

Do not commit real credentials to the repository.

```python
from datetime import date

from swingtrader.data.ingestion.market_data import ingest_historical_daily_prices

result = ingest_historical_daily_prices(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1),
    limit=3,
)

print(result)
```
