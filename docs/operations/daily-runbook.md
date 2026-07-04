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
	- tickers with existing rows start from their latest stored `trading_date`;
	- tickers with no bronze rows are reported as not onboarded and are not updated by this job;
	- onboarded tickers already current for the requested exclusive `end_date` are skipped.
6. Call the existing historical ingestion function for each planned ticker update.
7. Upsert bronze rows idempotently.
8. Log active ticker count, update ticker count, not-onboarded ticker count, skipped ticker count, planned update count, row counts, and failures.

The job uses bronze storage as the source of truth for progress. It does not maintain a separate checkpoint.

Use the bronze onboarding workflow for first loads and newly active tickers before expecting the daily update job to refresh them.

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