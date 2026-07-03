# Swingtrader App

Swingtrader is a personal trading decision-support application built around a data-first workflow. The project is intended to download market data for a curated active trading universe, store source-oriented records in bronze tables, build model-ready features, train ranking models, and eventually present ranked trade candidates in a small web application.

!!! note "Current status"
    The current implementation covers project setup, active ticker universe resolution, yfinance historical daily price ingestion, bronze daily price storage, and bronze onboarding checks for newly active tickers. Feature generation, modeling, deployment, and frontend workflows are planned.

The long-term goal is not automatic order placement. The application should support disciplined manual trading decisions by ranking candidate tickers, showing relevant supporting data, and helping with risk-aware position sizing.

## Main Paths

- [Getting started](getting-started/installation.md): install dependencies, run tests, and perform the first local data workflow.
- [Architecture](architecture/overview.md): understand package boundaries, data flow, and roadmap.
- [Data](data/overview.md): understand clients, ingestion, bronze storage, ticker onboarding, and planned features.
- [Modeling](modeling/overview.md): understand planned readiness, eligibility, target, and evaluation concepts.
- [Operations](operations/daily-runbook.md): track how the future scheduled workflow should run.
- [Reference](reference/glossary.md): shared vocabulary and API reference.

## Documentation Status

This documentation is a living document. Pages use these meanings:

- **Implemented**: code exists in the repository and is covered by tests.
- **Planned**: intended project direction, but not yet implemented.
- **Open decision**: design choice that still needs review before implementation.