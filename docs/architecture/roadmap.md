# Roadmap

This roadmap is a planning aid, not a replacement for GitHub issues. It describes the intended order of work and separates implemented functionality from planned capabilities.

## Completed Foundation

- Project packaging, testing, linting, and MkDocs documentation.
- Active ticker universe resolution.
- yfinance historical daily price download and normalization.
- Bronze daily price schema, idempotent writer, and pandas loader.
- Runnable ticker onboarding and daily market-data update jobs.
- Bronze-backed inference-readiness and training-eligibility rules.
- Reusable indicator library.
- In-memory return, trend, momentum, volatility, price-action, volume, and market-structure feature generation.
- V1 model target and evaluation contract.
- V1 forward-return and binary-target generation.

## Repository Stabilization

Complete these items before treating baseline model results as durable:

1. Define and apply consistent corporate-action semantics for cross-session price calculations.
2. Add split-equivalence and generic point-in-time feature contract tests.
3. Standardize feature-column collision behavior and remove accidental private cross-package contracts.
4. Introduce a versioned feature-set specification and experiment manifest.
5. Classify selected feature calculations as bounded-window, expanding, or path-dependent for future production planning.

## Model Development

1. Implement leakage-safe temporal train, validation, and test dataset construction.
2. Add date-based splits with target-horizon purging.
3. Establish dummy-classifier and date-matched random-ranking baselines.
4. Train and evaluate the first XGBoost classifier and regression candidates.
5. Evaluate classification quality, calibration, daily cross-sectional ranking, and stability over time.
6. Perform feature ablation and select a reproducible OHLCV V1 feature set.

## Production Preparation

- Define the persisted prediction and ranking schema.
- Implement a scheduled selected-feature and model-inference workflow.
- Store prediction snapshots for API consumption.
- Add PostgreSQL schema migrations when production tables begin to evolve.
- Add the FastAPI backend without placing heavy model computation in request handlers.
- Initialize the TypeScript and React frontend and generate API types from OpenAPI where practical.
- Deploy the database, scheduled jobs, backend, and frontend on Render.

## Later Scope

- Market-data quality visibility in the frontend.
- Risk and position-sizing support.
- Executable backtesting and strategy evaluation.
- Historical-universe and survivorship improvements.
- Macro, benchmark, sector, fundamental, news, or sentiment inputs after the OHLCV-only path is useful.

## Open Decisions

- Exact temporal split and walk-forward schedule.
- Minimum label count for training dataset inclusion.
- Exact historical training-universe representation.
- Which feature blocks justify production computation or persisted state.
- Prediction retention and model-version lifecycle.
- Authentication and authorization implementation.
- Which market-data quality indicators should be visible in the frontend.
