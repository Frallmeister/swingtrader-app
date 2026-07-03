# Data Layer Overview

The data package owns acquisition, storage, and preparation of market and macroeconomic data.

```text
data/
  clients/    Provider-specific API clients and HTTP wrappers.
  ingestion/  Retrieval and normalization workflows.
  bronze/     Source-oriented schemas and writers.
  features/   Planned model-ready transformations.
  jobs/       Planned runnable entrypoints.
```

## Boundaries

Provider details belong in `clients`. Retrieval decisions belong in `ingestion`. Source persistence belongs in `bronze`. Feature transformations belong in `features`. Thin operational commands belong in `jobs`.

## Implemented Data Paths

- yfinance daily price download and normalization.
- Historical daily price ingestion into bronze storage.
- Active ticker bronze onboarding checks.
- Idempotent upsert behavior for bronze daily prices.

## Planned Data Paths

- Daily update job.
- Feature tables.
- Macro data clients and ingestion.
- Data quality and readiness checks.
- Market data quality summaries, such as available history length per ticker.
- Production data freshness monitoring.