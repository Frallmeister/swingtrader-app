# Data Layer Overview

The data package owns acquisition, storage, and preparation of market and macroeconomic data.

```text
data/
  clients/    Provider-specific API clients and HTTP wrappers.
  ingestion/  Retrieval and normalization workflows.
  bronze/     Source-oriented schemas and writers.
  features/   Planned model-ready transformations.
  jobs/       Thin runnable entrypoints for local data workflows.
```

## Boundaries

Provider details belong in `clients`. Retrieval decisions belong in `ingestion`. Source persistence belongs in `bronze`. Feature transformations belong in `features`. Thin operational commands belong in `jobs`.

## Implemented Data Paths

- yfinance daily price download and normalization.
- Historical daily price ingestion into bronze storage.
- Active ticker bronze onboarding checks.
- Runnable market data onboarding job for active tickers with no bronze rows.
- Runnable daily update job for already-onboarded active tickers.
- Idempotent upsert behavior for bronze daily prices.
- Pandas loading from bronze daily prices for notebook inspection and EDA.
- Bronze-backed ticker inference readiness and training eligibility checks.
- Ready-to-use data database initialization through `swingtrader.data.db`.

## Planned Data Paths

- Model-ready feature generation.
- Feature persistence, if justified by later modeling workflows.
- Macro data clients and ingestion.
- Market data quality summaries, such as available history length per ticker.
- Production data freshness monitoring.