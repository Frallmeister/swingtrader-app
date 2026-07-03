# First Ingestion

This page shows the current library-level data workflow. A runnable daily update command is planned, but not implemented yet.

## Database

By default, local ingestion uses SQLite at `data/swingtrader.sqlite`. Override this with the `SWINGTRADER_DATABASE_URL` environment variable when needed.

Example PowerShell override:

```powershell
$env:SWINGTRADER_DATABASE_URL = "sqlite+pysqlite:///data/swingtrader.sqlite"
```

## Historical Daily Prices

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

## Bronze Onboarding Check

The onboarding workflow compares active tickers with the bronze daily price table. A ticker is considered onboarded once any bronze daily price row exists for that provider.

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