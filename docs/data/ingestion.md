# Ingestion

Ingestion code decides what to retrieve, which date windows to request, how to normalize provider responses, and how to persist source-oriented records.

## Historical Daily Prices

The implemented market data ingestion entrypoint is `ingest_historical_daily_prices(...)`.

It:

- resolves active tickers when no explicit tickers are provided;
- downloads daily prices from yfinance one ticker at a time;
- normalizes rows into the bronze daily price shape;
- upserts rows into `bronze_market_daily_prices`;
- records request metadata such as `request_id` and `fetched_at`;
- records per-ticker failures without stopping successful tickers by default.

## Date Semantics

`start_date` is inclusive and `end_date` is exclusive, matching the yfinance request style used by the client.

## Failure Handling

By default, ingestion records ticker-level failures in the returned result. Passing `raise_on_failure=True` raises a `MarketDataIngestionError` after all tickers have been attempted if any ticker failed.

## Bronze Onboarding

The onboarding sync checks whether active tickers exist in bronze storage. It classifies active tickers as:

- `missing`: no bronze daily price rows exist for the provider;
- `onboarded`: at least one bronze daily price row exists.

Backfill only targets missing tickers. Historical completeness is intentionally out of scope and belongs to future readiness and eligibility rules.

## Planned Daily Job

A runnable daily update job is planned. It should wrap existing ingestion functions rather than reimplementing provider download or bronze upsert logic.