# Architecture Overview

Swingtrader is a data-first decision-support application. External market data must be reproducible and model inputs must be point-in-time safe before modeling, ranking, or user-interface code depends on them.

## Research Flow

The implemented repository currently supports the research foundation through purged temporal splitting:

```mermaid
flowchart LR
    A[Curated ticker universes] --> B[Market-data ingestion]
    B --> C[Bronze storage]
    C --> D[Eligibility and loading]
    D --> E[Feature and target generation]
    E --> F[Canonical temporal datasets]
    F --> G[Purged temporal splits]
    G --> H[Model training and evaluation]
```

Feature and target generation currently runs in memory. Persistence of model-ready datasets remains optional and should be introduced only when reproducibility or operational evidence justifies it.

## Planned Production Flow

Production ranking should run as a scheduled workflow rather than inside an HTTP request:

```mermaid
flowchart LR
    A[Scheduled market update] --> B[Bronze storage]
    B --> C[Selected feature calculation]
    C --> D[Model inference]
    D --> E[Persisted prediction snapshots]
    E --> F[FastAPI backend]
    F --> G[TypeScript and React frontend]
```

The backend may serve bounded chart-data or indicator requests on demand, but it should not execute full-universe feature generation or model inference while a user waits for an API response.

## Implemented

- Curated universe YAML files and active ticker resolution.
- yfinance daily price download and normalization.
- Historical ingestion into `bronze_market_daily_prices` with idempotent upserts.
- Runnable onboarding and daily market-data update jobs.
- Bronze-backed inference-readiness and training-eligibility checks.
- Pandas loading from bronze daily prices.
- Reusable technical indicators.
- In-memory return, trend, momentum, volatility, price-action, volume, and market-structure feature generation.
- Versioned V1 forward-return and V2 ATR barrier-event target generation.
- Canonical unsplit temporal dataset construction.
- Purged fixed train, validation, and locked-test splitting with diagnostics.
- Immutable experiment specifications and optional local MLflow tracking.
- Local SQLite and configurable SQLAlchemy database URLs.
- Automated Ruff, pytest, and strict MkDocs checks.

## Planned

- Baseline model training, evaluation, and feature selection.
- Local and scheduled production inference with prediction persistence.
- A FastAPI backend under `swingtrader.api`.
- A separate TypeScript and React application under `frontend/`.
- Render deployment and scheduled jobs.
- PostgreSQL production storage.
- Macro-data ingestion and macro/context features after the OHLCV-only V1 path is useful.

## Package Boundaries

The intended dependency direction is:

```text
jobs -> ingestion -> clients
jobs -> bronze
jobs -> feature and inference services

ingestion -> bronze
eligibility -> bronze
features -> indicators and shared numerical contracts
modeling.datasets -> data outputs and feature specifications
modeling.experiments -> modeling.datasets
api -> application services and persisted outputs
frontend -> HTTP API only
```

The data layer must not import API or frontend implementation code. The API must not own numerical feature algorithms or model training. The React frontend must not access the database or depend on Python internals.

The current canonical pandas market frame remains an internal numerical-layer contract. Database, API, and frontend boundaries should use ordinary records or explicit schemas rather than exposing pandas index conventions.

See [Architecture Decisions](decisions/index.md) for accepted decisions and their rationale.
