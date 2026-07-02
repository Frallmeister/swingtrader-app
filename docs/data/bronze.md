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

## What Bronze Should Not Contain

Bronze tables should not contain:

- model-ready feature joins;
- train, validation, or test split markers;
- imputed feature values;
- prediction outputs;
- trade decisions.

Those belong in future feature, modeling, and application tables.