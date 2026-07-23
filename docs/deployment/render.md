# Render Deployment

Render deployment is planned, not implemented yet.

## Intended Services

The deployed application is expected to use:

- a PostgreSQL database service;
- scheduled jobs for market-data updates and production inference;
- a FastAPI web service for bounded data and application requests;
- a separately built TypeScript and React frontend, likely deployed as a static site;
- environment variables for configuration and secrets.

The exact number of scheduled services may change as operational measurements become available. Market ingestion and inference may initially run in one scheduled workflow and be split only when reliability or runtime requires it.

## Data Flow

The intended daily production sequence is:

```text
market update
    -> selected feature calculation
    -> model inference
    -> persisted prediction snapshot
    -> FastAPI reads persisted results
    -> React presents ranked candidates
```

The frontend must not access PostgreSQL directly. The API must not trigger full-market feature generation or inference inline. Heavy work belongs in scheduled jobs so request latency and service resource usage remain predictable.

## Planned Environment Variables

- `SWINGTRADER_DATABASE_URL`
- future authentication and authorization settings
- future provider credentials if needed
- future model-artifact and runtime settings

## Deployment Principles

- No secrets committed to the repository.
- Jobs should be idempotent and safe to retry.
- Logs should contain model version, request identifiers, ticker counts, row counts, and failures where applicable.
- Persisted predictions should identify the model, feature set, calculation date, and data cutoff used.
- The API should return explicit schemas rather than pandas-specific representations.
- The frontend should consume only the documented HTTP API.
