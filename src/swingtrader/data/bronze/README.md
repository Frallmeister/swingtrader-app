# Bronze data

## Responsibility

The bronze data layer stores normalized provider data close to its source representation. It preserves the observation identity and retrieval metadata needed to rerun ingestion safely, update previously fetched observations idempotently, rebuild downstream features and targets, and inspect historical source data during analysis.

Bronze is a data-layer responsibility, not a synonym for YFinance. The current implementation is focused on daily market prices fetched through the active market-data ingestion workflows. Future bronze datasets may represent other source-oriented domains, such as macroeconomic observations or financial time series, if those ingestion paths are added later.

## Current contents

The implemented table is `bronze_market_daily_prices`. It stores daily OHLCV values, adjusted close, dividends, stock splits, provider identity, trading date, fetch timestamp, and request identifier. Rows are keyed by provider, ticker, and trading date so repeated ingestion windows can update the same market observation instead of creating duplicates.

This package currently contains:

- schema definitions for bronze daily market prices;
- writer functions with natural-key upsert behavior;
- query helpers for coverage, freshness, and source-quality summaries;
- pandas loaders for notebook EDA and source inspection.

Local development uses SQLite through SQLAlchemy. The writer uses dialect-aware upsert statements for SQLite and PostgreSQL, and callers should obtain ready-to-use data engines through `swingtrader.data.db` when they need known data tables initialized.

## Design principles

Bronze records should remain source-oriented and reproducible. They should keep provider identifiers, observation dates, fetch metadata, and request metadata rather than hiding source lineage behind model-specific transformations. Bronze helpers may summarize stored source rows for onboarding, update planning, readiness checks, or data inspection, but they should not create model-ready explanatory variables.

## Package boundaries

Provider clients and download decisions belong in neighboring ingestion and client layers. Feature engineering, target and label generation, prediction outputs, model artifacts, and trading decisions do not belong in bronze storage. Those concerns should live in future feature, modeling, and application layers.

## Further documentation

See [Bronze storage](../../../../docs/data/bronze.md) for the detailed table documentation, idempotency behavior, loader examples, and portability notes.
