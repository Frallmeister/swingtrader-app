# Architecture Overview

Swingtrader is a data-first decision-support application. External market data must be reproducible and model inputs must be point-in-time safe before modeling, ranking, or user-interface code depends on them.

## Research Flow

The implemented repository currently supports the research foundation through purged temporal splitting:

The shapes distinguish inputs, actions, data products, and planned work:

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 30}}}%%
flowchart TB
    universe[/Curated ticker universes/]
    ingest([Ingest market data])
    bronze[(Bronze prices)]
    eligible([Resolve eligibility and load history])
    generate([Generate features and targets])
    dataset[(Canonical temporal dataset)]
    split([Apply purged temporal split])
    train([Train and evaluate models<br/>(planned)])

    universe --> ingest --> bronze --> eligible --> generate --> dataset --> split
    split -.-> train

    classDef input fill:#455a64,stroke:#263238,color:#ffffff
    classDef action fill:#9a4d00,stroke:#5d2e00,color:#ffffff
    classDef artifact fill:#2e7d32,stroke:#17451c,color:#ffffff
    classDef planned fill:#424242,stroke:#9e9e9e,color:#ffffff,stroke-dasharray:6 4
    class universe input
    class ingest,eligible,generate,split action
    class bronze,dataset artifact
    class train planned
```

Feature and target generation currently runs in memory. Persistence of model-ready datasets remains optional and should be introduced only when reproducibility or operational evidence justifies it.

## Planned Production Flow

Production ranking should run as a scheduled workflow rather than inside an HTTP request:

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 30}}}%%
flowchart TB
    update([Scheduled market update])
    bronze[(Bronze prices)]
    features([Calculate selected features])
    inference([Run model inference])
    predictions[(Prediction snapshots)]
    api([FastAPI backend])
    frontend([React frontend])

    update --> bronze --> features --> inference --> predictions --> api --> frontend

    classDef action fill:#9a4d00,stroke:#5d2e00,color:#ffffff
    classDef artifact fill:#2e7d32,stroke:#17451c,color:#ffffff
    classDef service fill:#1565c0,stroke:#0d3d75,color:#ffffff
    class update,features,inference action
    class bronze,predictions artifact
    class api,frontend service
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

The main dependency direction is shown below. Arrows point from a consumer to the package or service it may depend on. Dashed edges and labels mark planned boundaries rather than implemented dependencies.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 22, "rankSpacing": 28}}}%%
flowchart TB
    subgraph entrypoints["Operational entrypoints"]
        jobs["data.jobs"]
    end

    subgraph data["Data and numerical layer"]
        clients["data.clients"]
        ingestion["data.ingestion"]
        bronze["data.bronze"]
        eligibility["data.eligibility"]
        indicators["indicators"]
        features["data.features"]
    end

    subgraph modeling["Modeling layer"]
        datasets["modeling.datasets"]
        experiments["modeling.experiments"]
    end

    subgraph application["Planned application boundary"]
        services["Application services"]
        api["FastAPI backend"]
        frontend["React frontend"]
    end

    jobs --> ingestion
    jobs --> bronze
    jobs -.-> services
    ingestion --> clients
    ingestion --> bronze
    eligibility --> bronze
    eligibility --> ingestion
    eligibility --> clients
    features --> indicators
    datasets --> bronze
    datasets --> eligibility
    datasets --> features
    experiments --> datasets
    api -.-> services
    frontend -.-> api

    classDef entry fill:#455a64,stroke:#263238,color:#ffffff
    classDef dataNode fill:#2e7d32,stroke:#17451c,color:#ffffff
    classDef modelNode fill:#1565c0,stroke:#0d3d75,color:#ffffff
    classDef planned fill:#424242,stroke:#9e9e9e,color:#ffffff,stroke-dasharray:6 4
    class jobs entry
    class clients,ingestion,bronze,eligibility,indicators,features dataNode
    class datasets,experiments modelNode
    class services,api,frontend planned
```

The data layer must not import API or frontend implementation code. The API must not own numerical feature algorithms or model training. The React frontend must not access the database or depend on Python internals.

The current canonical pandas market frame remains an internal numerical-layer contract. Database, API, and frontend boundaries should use ordinary records or explicit schemas rather than exposing pandas index conventions.

See [Architecture Decisions](decisions/index.md) for accepted decisions and their rationale.
