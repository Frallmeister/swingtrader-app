# Swingtrader

Swingtrader is a personal swing-trading decision-support application built around reproducible market data ingestion, bronze storage, ticker readiness checks, feature generation, model training, and eventually ranked trade candidates for manual review.

The long-term goal is to support disciplined manual trading decisions, not automatic order placement.

## Current Status

The project currently implements the data foundation:

* active ticker universe resolution from YAML configuration
* yfinance historical daily price download and normalization
* bronze daily price storage with idempotent upserts
* initial market data onboarding for active tickers with no bronze rows
* daily market data updates for already-onboarded tickers
* inference-readiness and training-eligibility checks based on bronze data quality
* in-memory adjusted-close return feature generation
* local SQLite support and configurable SQLAlchemy database URLs
* MkDocs-based project documentation
* pytest/ruff-based local quality checks

Broader feature engineering, target persistence, model training, inference, prediction storage, dashboarding, deployment, and macro/market-context features are planned.

## Documentation

The project documentation is the main living reference for implemented behavior, planned architecture, and open design decisions.

Useful entry points:

* [Documentation home](docs/index.md)
* [Installation](docs/getting-started/installation.md)
* [Development workflow](docs/getting-started/development.md)
* [First ingestion](docs/getting-started/first-ingestion.md)
* [Architecture overview](docs/architecture/overview.md)
* [Ticker universes](docs/architecture/ticker-universes.md)
* [Data overview](docs/data/overview.md)
* [Ticker onboarding](docs/data/ticker-onboarding.md)
* [Ticker eligibility](docs/data/eligibility.md)
* [Roadmap](docs/architecture/roadmap.md)

## Quick Start

Use `uv` for dependency and environment management.

```powershell
uv sync --all-extras --dev --group notebook --group docs
```

Run the standard local checks:

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
uv run --group docs mkdocs build --strict
```

Serve the documentation locally:

```powershell
uv run --group docs mkdocs serve
```

## Local Data Workflow

By default, local ingestion uses SQLite at:

```text
data/swingtrader.sqlite
```

Override this with `SWINGTRADER_DATABASE_URL` when needed.

Example PowerShell override:

```powershell
$env:SWINGTRADER_DATABASE_URL = "sqlite+pysqlite:///data/swingtrader.sqlite"
```

### Initial Market Data Onboarding

Run the onboarding job first to create initial bronze daily price rows for active tickers that are missing data:

```powershell
uv run python -m swingtrader.data.jobs.onboard_market_data --limit 3
```

Use an explicit exclusive end date for deterministic local runs:

```powershell
uv run python -m swingtrader.data.jobs.onboard_market_data --limit 3 --end-date 2026-07-04
```

The onboarding job skips tickers that already have bronze rows. It is the operational entrypoint for first setup and for newly added active tickers.

### Daily Market Data Update

After tickers have been onboarded, run the daily update job to refresh already-onboarded active tickers:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3
```

Use an explicit exclusive end date for deterministic local runs:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3 --end-date 2026-07-04
```

Tickers with no bronze rows are reported as not onboarded and skipped by the daily update job. Run the onboarding job first for new active tickers.

### Ticker Eligibility Checks

After onboarding and daily updates have populated enough history, use the eligibility layer to distinguish active tickers from inference-ready and training-eligible tickers.

```python
from datetime import date

from swingtrader.data.eligibility import (
    check_inference_readiness,
    check_training_eligibility,
)

inference_result = check_inference_readiness(reference_date=date(2026, 7, 4))
print(inference_result.ready_tickers)
print(inference_result.not_ready_tickers)

training_result = check_training_eligibility()
print(training_result.eligible_tickers)
print(training_result.not_eligible_tickers)
```

The current eligibility checks are based on bronze daily price state and data quality. Future feature and label checks should extend this layer rather than changing bronze onboarding semantics.

## Project Layout

```text
src/swingtrader/
  configs/        Source-controlled project configuration.
  core/           Shared infrastructure and utilities.
  data/           Provider clients, ingestion, bronze storage, jobs, eligibility checks, and features.
  modeling/       Planned model training, inference, and evaluation code.
  web/            Planned web application.

tests/            Unit and integration-style tests.

docs/             MkDocs project documentation.
```

## Operational Concepts

The data workflow intentionally separates these concepts:

```text
active ticker:
  ticker included in the configured active trading universe

onboarded ticker:
  active ticker with at least one bronze daily price row

inference-ready ticker:
  ticker with enough recent and clean bronze data for production ranking

training-eligible ticker:
  ticker with enough historical and clean bronze data for model training/evaluation
```

This separation avoids treating universe membership, storage presence, production inference readiness, and training suitability as the same thing.

## Development Notes

Keep application code in `src/swingtrader`, tests in `tests`, and long-form guidance in `docs`.

Before opening a pull request, run:

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
uv run --group docs mkdocs build --strict
```

## Copyright

Copyright © 2026 Fredrik Hansson. All rights reserved.

This software and its source code are proprietary and confidential. No permission is granted to use, copy, modify, distribute, sublicense, or create derivative works from this software, in whole or in part, without prior written permission from the copyright holder.
