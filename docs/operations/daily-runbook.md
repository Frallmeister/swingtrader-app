# Daily Runbook

This page describes the daily market data workflow that should eventually run as a scheduled server job. It is not intended to be a manual checklist for a user to perform every day.

The first implementation is a local runnable job for bronze daily market prices.

## Local Command

Run the daily market data update job locally with:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data
```

For a smoke run over the first few active tickers:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3
```

For deterministic manual runs, pass an exclusive end date:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --end-date 2026-07-04
```

Use `--fail-on-ticker-failure` when a scheduler should treat any ticker-level failure as a failed job run. Successful tickers are still written before the command exits.

## Implemented Daily Update Flow

1. Configure logging.
2. Resolve active tickers from the active trading universe.
3. Load market data settings from `src/swingtrader/configs/market_data.yml`.
4. Read each active ticker's latest bronze daily price state.
5. Build per-ticker update plans:
	- tickers with no bronze rows start from the configured initial start date;
	- tickers with existing rows start from their latest stored `trading_date`;
	- tickers already current for the requested exclusive `end_date` are skipped.
6. Call the existing historical ingestion function for each planned ticker update.
7. Upsert bronze rows idempotently.
8. Log active ticker count, update ticker count, skipped ticker count, planned update count, row counts, and failures.

The job uses bronze storage as the source of truth for progress. It does not maintain a separate checkpoint.

## Optional Backfill

Pass `--backfill` to run the explicit bronze onboarding sync for missing active tickers before the daily update planner runs:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --backfill
```

The normal daily planner can still initialize an empty bronze table from the configured initial start date. The backfill flag adds an explicit onboarding step and summary for missing active tickers.

## Developer Checks

Useful local checks:

```powershell
uv run pytest tests/data/jobs/test_update_market_data.py
uv run pytest tests/data/ingestion/test_market_data.py tests/data/ingestion/test_onboarding.py
```

## Planned Extensions

- Render scheduling.
- Feature refresh for affected tickers.
- Inference-readiness filtering.
- Production inference after feature generation and modeling exist.
- Macro data ingestion jobs.