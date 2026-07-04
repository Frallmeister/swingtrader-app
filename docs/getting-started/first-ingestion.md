# First Ingestion

This page shows the first useful local data workflows: bronze onboarding for first rows, the runnable daily market data job for ongoing updates, and the lower-level library functions they wrap.

## Database

By default, local ingestion uses SQLite at `data/swingtrader.sqlite`. Override this with the `SWINGTRADER_DATABASE_URL` environment variable when needed.

Example PowerShell override:

```powershell
$env:SWINGTRADER_DATABASE_URL = "sqlite+pysqlite:///data/swingtrader.sqlite"
```

## Market Data Onboarding Job

Run the market data onboarding job first to create initial bronze daily price rows for active tickers that are missing data:

```powershell
uv run python -m swingtrader.data.jobs.onboard_market_data --limit 3
```

The job uses the configured `initial_start_date` from `market_data.yml` unless you pass `--start-date`. Use an explicit exclusive end date for deterministic local runs:

```powershell
uv run python -m swingtrader.data.jobs.onboard_market_data --limit 3 --end-date 2026-07-04
```

Already-onboarded tickers are skipped. This command is the operational entrypoint for first setup and for active tickers added later.

## Bronze Onboarding API

The lower-level onboarding workflow compares active tickers with the bronze daily price table. A ticker is considered onboarded once any bronze daily price row exists for that provider. Use `backfill=True` to create first bronze rows for missing active tickers from Python:

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

After onboarding and daily updates have populated enough history, use the eligibility checks described in [Ticker Eligibility](../data/eligibility.md) to distinguish active tickers from inference-ready and training-eligible tickers.

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
