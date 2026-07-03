# Daily Runbook

This page describes the planned automated daily workflow that should eventually run as a scheduled server job. It is not intended to be a manual checklist for a user to perform every day.

The page exists now so the future daily update job has a clear operating target.

## Current Developer Checks

Today, the scheduled job does not exist yet. The commands below are developer checks for the implemented library functions, not the intended production workflow.

Useful local checks:

```powershell
uv run pytest tests/data/ingestion/test_market_data.py tests/data/ingestion/test_onboarding.py
```

## Planned Automated Daily Update Flow

1. Configure logging.
2. Resolve active tickers.
3. Detect newly active tickers missing from bronze.
4. Backfill missing active tickers if needed.
5. Fetch recent daily market data for active tickers.
6. Upsert bronze rows idempotently.
7. Log ticker count, date range, rows written, and failures.
8. Later: refresh affected feature rows.
9. Later: run production inference.

## Planned Command

A future job should run with a command similar to:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data
```

The exact module name and arguments will be defined when the daily update job is implemented.