# Swingtrader App

Swingtrader is a personal trading decision-support application built around a data-first workflow. The project downloads market data for a curated trading universe, stores source-oriented bronze records, builds model-ready features and targets, and will eventually present ranked trade candidates for manual review.

!!! note "Current status"
    The repository implements the local market-data foundation, runnable onboarding and daily-update jobs, bronze-backed eligibility checks, pandas loading, versioned V1 forward-return and V2 ATR barrier-event targets, and in-memory return, trend, momentum, volatility, price-action, volume, and market-structure feature generation.

    The pre-modeling contracts and corporate-action semantics are now stabilized. The next implementation work is to add repository-owned experiment specifications and local MLflow tracking, then build leakage-safe temporal datasets.

    Model training, production inference, prediction persistence, the FastAPI backend, the TypeScript/React frontend, deployed scheduling, and macro-data ingestion remain planned.

The long-term goal is not automatic order placement. The application should support disciplined manual trading decisions by ranking candidate tickers, showing relevant supporting data, and helping with risk-aware position sizing.

## Main Paths

- [Getting started](getting-started/installation.md): install dependencies, run tests, and perform the first local data workflow.
- [Architecture](architecture/overview.md): understand package boundaries and research and production data flows.
- [Architecture decisions](architecture/decisions/index.md): understand accepted design decisions and their consequences.
- [Roadmap](architecture/roadmap.md): understand the completed stabilization work and modeling sequence.
- [Data](data/overview.md): understand clients, ingestion, bronze storage, ticker onboarding, eligibility checks, and features.
- [Modeling](modeling/overview.md): understand readiness, targets, and evaluation concepts.
- [ATR barrier targets](modeling/atr-barrier-targets.md): understand the V2 next-open stop-loss and take-profit label contract.
- [Operations](operations/daily-runbook.md): run the implemented local daily workflow and understand the planned production sequence.
- [Reference](reference/glossary.md): shared vocabulary and API reference.

## Documentation Status

This documentation is a living document. Pages use these meanings:

- **Implemented**: code exists in the repository and is covered by tests.
- **Planned**: intended project direction, but not yet implemented.
- **Open decision**: design choice that still needs review before implementation.

Accepted architectural decisions are recorded as ADRs and should be superseded by a new ADR rather than silently rewritten when the direction changes.
