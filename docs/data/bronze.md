# Bronze Storage

Bronze storage keeps source-oriented records close to provider data while adding project-level metadata for reproducibility.

## Daily Market Prices

The implemented table is `bronze_market_daily_prices`.

Primary key:

```text
(provider, ticker, trading_date)
```

Important fields include:

- `provider`
- `ticker`
- `trading_date`
- OHLC values
- `adjusted_close`
- `volume`
- dividends and stock splits
- `fetched_at`
- `request_id`

## Idempotency

The writer upserts by `(provider, ticker, trading_date)`. Rerunning the same ingestion window should update existing rows instead of duplicating them.

On conflict, market values plus `fetched_at` and `request_id` are updated from the incoming row.

## SQLite And PostgreSQL

Local development currently defaults to SQLite. PostgreSQL support is planned for deployment through the existing SQLAlchemy abstraction and optional PostgreSQL dependency.

Use the `SWINGTRADER_DATABASE_URL` environment variable to point the application at a non-default database. `resolve_database_engine()` resolves that setting when no explicit database URL is passed and creates known application tables if they do not exist yet.

## Loading Daily Prices In Notebooks

After running bronze onboarding or the daily market data update job, use the pandas loader to inspect downloaded source rows in a notebook:

```python
from pathlib import Path

from swingtrader.core.db import resolve_database_engine
from swingtrader.data.bronze.loaders import load_bronze_daily_prices

repo_root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "pyproject.toml").exists())
database_url = f"sqlite+pysqlite:///{(repo_root / 'data' / 'swingtrader.sqlite').as_posix()}"
engine = resolve_database_engine(database_url=database_url)

prices = load_bronze_daily_prices(
    engine=engine,
    tickers=["AAK.ST", "ADDT-B.ST", "AFRY.ST"],
    start_date="2026-06-01",
    end_date="2026-06-30",
)

prices.head()
```

Use `columns` to limit the returned DataFrame. The loader always includes `provider`, `ticker`, and `trading_date` first, then appends the requested non-key columns:

```python
prices = load_bronze_daily_prices(
    engine=engine,
    tickers="AAK.ST",
    start_date="2020-01-01",
    columns="close",
)
```

Both `tickers` and `columns` accept either a single string or a sequence of strings.

This helper is for bronze EDA and source inspection. It does not create technical indicators, model-ready features, targets, or readiness decisions.

The returned DataFrame uses notebook-friendly dtypes: `trading_date` is pandas datetime, `volume` is nullable integer, and OHLC/dividend/split columns are floats.

## Future Bronze Tables

The bronze layer should eventually store more than daily market prices. Planned additions include source-oriented macroeconomic and financial time-series tables once macro clients and ingestion workflows are implemented.

Those future tables should follow the same principles: preserve provider identity, observation dates, fetch metadata, and enough provenance to rebuild downstream features without redownloading data unnecessarily.

## What Bronze Should Not Contain

Bronze tables should not contain:

- model-ready feature joins;
- train, validation, or test split markers;
- imputed feature values;
- prediction outputs;
- trade decisions.

Those belong in future feature, modeling, and application tables.