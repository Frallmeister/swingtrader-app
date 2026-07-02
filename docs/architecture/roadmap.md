# Roadmap

This roadmap is a planning aid, not a replacement for GitHub issues. It describes the intended order of work and separates implemented functionality from planned capabilities.

## Completed Foundation

- Project packaging, testing, and linting.
- Active ticker universe resolution.
- yfinance historical daily price download and normalization.
- Bronze daily price schema and idempotent writer.
- Historical ingestion library function.
- Local/dev ticker universe generation utility.
- Bronze onboarding sync for newly active tickers.

## Near Term

1. Documentation foundation with MkDocs.
2. Runnable daily market data update job.
3. Documentation update for real commands and operations.
4. Inference readiness and training eligibility rules.
5. Training universe configuration.
6. Initial market feature generation.
7. Model target and evaluation strategy.
8. DataLoader split design.

## Medium Term

- Forward-return label generation.
- First feature and label tables.
- First training dataset builder.
- Baseline ranking model.
- Local inference workflow.
- Prediction/ranking output schema.

## Longer Term

- Render deployment with scheduled jobs.
- PostgreSQL production storage.
- Dashboard for latest ranked candidates.
- Risk and position sizing support.
- Backtesting and evaluation reports.
- Macro and market-context features.

## Open Decisions

- First prediction target and horizon.
- Ranking metric priority.
- Minimum history for inference readiness.
- Minimum history and label count for training eligibility.
- Exact training universe representation.
- How prediction output should be stored and displayed.