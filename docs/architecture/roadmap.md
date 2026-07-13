# Roadmap

This roadmap is a planning aid, not a replacement for GitHub issues. It describes the intended order of work and separates implemented functionality from planned capabilities.

## Completed Foundation

- Project packaging, testing, and linting.
- Active ticker universe resolution.
- yfinance historical daily price download and normalization.
- Bronze daily price schema and idempotent writer.
- Historical ingestion library function.
- Local/dev ticker universe generation utility.
- Bronze onboarding sync and runnable onboarding job for newly active tickers.
- Runnable daily market data update job.
- Documentation foundation with MkDocs.
- Documentation for real local commands and operations.
- Bronze-backed inference-readiness and training-eligibility rules.
- Bronze daily-price pandas loader.

## Near Term

1. Complete the bounded foundation cleanup.
2. Define the V1 prediction target and evaluation contract.
3. Implement forward-return and binary direction label generation.
4. Implement initial reusable OHLCV-derived features.
5. Implement leakage-safe temporal train, validation, and test dataset construction.
6. Train and evaluate the first baseline models.

## Medium Term

- Baseline ranking model.
- Local inference workflow.
- Prediction/ranking output schema.
- Persistent feature or label storage if in-memory experiments show it is justified.

## Longer Term

- Render deployment with scheduled jobs.
- PostgreSQL production storage.
- Dashboard for latest ranked candidates.
- Market data quality visibility in the web app.
- Risk and position sizing support.
- Backtesting and evaluation reports.
- Macro data ingestion and macro/context features after the OHLCV-only V1 path is useful.

## Open Decisions

- First prediction target and horizon.
- Ranking metric priority.
- Feature and label gates for future modeling readiness.
- Minimum label count for training dataset inclusion.
- Exact training universe representation.
- How prediction output should be stored and displayed.
- Whether and when model-ready features should be persisted.
- Which market data quality indicators should be visible in the web app.