# Bronze Market Data Schema

The first bronze market data table stores daily OHLCV records downloaded from `yfinance`.
The schema is source-oriented: it keeps provider values close to the downloaded data while
adding enough provenance to make reruns traceable and idempotent.

## Table

`bronze_market_daily_prices`

| Column | Meaning |
| --- | --- |
| `provider` | External data source that produced the row. Initially `yfinance`. |
| `ticker` | Provider ticker symbol, such as `AAK.ST`. |
| `trading_date` | Market date for the daily bar. |
| `open` | Daily open price, stored as fixed precision numeric data. |
| `high` | Daily high price, stored as fixed precision numeric data. |
| `low` | Daily low price, stored as fixed precision numeric data. |
| `close` | Daily close price, stored as fixed precision numeric data. |
| `adjusted_close` | Adjusted close when available from the provider. Nullable initially. |
| `volume` | Daily traded volume. |
| `dividends` | Dividend amount reported for the date, when available. |
| `stock_splits` | Stock split value reported for the date, when available. |
| `fetched_at` | UTC timestamp for the ingestion attempt that wrote the row. |
| `request_id` | Identifier for the ingestion request that wrote the row. |

Instrument metadata such as name, currency, exchange, country, sector, industry, and asset type
is intentionally kept out of this price table. Those values belong in ticker metadata or reference
tables derived from the packaged universe files.

## Idempotency

Rows are identified by the composite primary key:

```text
provider, ticker, trading_date
```

This means a rerun for the same provider, ticker, and trading date must update or replace the
existing row instead of inserting a duplicate. `request_id` is not part of the unique key because
each ingestion attempt may have a new request id while describing the same market observation.

The intended write behavior for ingestion jobs is an upsert on the primary key. On conflict, the
latest provider values, `fetched_at`, and `request_id` should replace the existing row. This keeps
the bronze table reproducible for local SQLite while mapping cleanly to Postgres `ON CONFLICT`
semantics later.

`fetched_at` is supplied by ingestion code rather than a database default because it represents
the provider fetch time, not merely the database insert time. Ingestion code should write UTC
timestamps.

## Portability

The schema is defined with SQLAlchemy core types so it can be created in local SQLite during early
development and moved to Postgres with minimal changes when the app is deployed. Price and
corporate action values use fixed precision numeric columns to avoid making floating point storage
part of the database contract.
