# Swingtrader


[![CI](https://github.com/Frallmeister/swingtrader-app/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Frallmeister/swingtrader-app/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-blue.svg)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-ruff-black.svg)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://docs.astral.sh/uv/)

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
* in-memory adjusted-close return, trend, and momentum feature generation, plus high/low/close volatility features
* local SQLite support and configurable SQLAlchemy database URLs
* MkDocs-based project documentation
* pytest/ruff-based local quality checks

Feature persistence, target persistence, model training, inference, prediction storage, dashboarding, deployment, and macro/market-context features are planned.

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

### Feature Generation

Feature helpers currently run in memory on ordered pandas dataframes. Inputs must use a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, in that exact order, plus the source price columns required by the feature family. The identifiers must not also appear as ordinary columns. Column-oriented data, such as rows loaded from bronze, is converted at the caller boundary with `set_index(...).sort_index()`.

```python
from swingtrader.data.features import (
    add_momentum_features,
    add_return_features,
    add_trend_features,
    add_volatility_features,
)
from swingtrader.data.features.pipeline import add_default_features

prices = prices.set_index(["provider", "ticker", "trading_date"]).sort_index()

# Run the standard families in one explicit call...
features = add_default_features(prices)

# ...or compose the family builders manually.
features = add_return_features(prices, horizons=(1, 5, 10, 20))
features = add_trend_features(features)
features = add_momentum_features(features)
features = add_volatility_features(features)
```

The codebase separates two responsibilities. **Indicators** in `swingtrader.indicators` calculate reusable technical quantities (moving averages, ADX, ATR, RSI, MACD, PPO, MFI, Bollinger Bands, squeeze momentum) that are meaningful outside any particular model. **Features** in `swingtrader.data.features` transform raw data and indicators into model inputs, deciding which source columns to use, how to combine and normalize them, and what the model-facing columns are named.

Return features add trailing adjusted-close returns. Trend features add moving-average ratios. Momentum features add PPO, PPO signal, PPO histogram, PPO percentile, RSI, stochastic, MFI, and squeeze momentum columns. Volatility features add `atr_percent` (Average True Range as a percentage of close), `bollinger_bandwidth`, and `bollinger_percent_b` columns. `add_default_features` runs the four families in a fixed order and is equivalent to applying them manually. All features are grouped by provider and ticker, leaving warm-up rows missing until the relevant windows are available. External consumers that need identifiers as columns convert back explicitly with `features.reset_index()`.

Standalone indicators can be imported directly for notebooks, tests, and future API or frontend charting:

```python
from swingtrader.indicators import adx, atr, ema, macd, rsi
```

Each public indicator accepts either a single ordered instrument or a canonical multi-instrument market frame, and preserves the input index and row order.

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
