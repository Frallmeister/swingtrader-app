# Daily Runbook

This page describes the implemented local market-data workflow and the intended production sequence. It is not intended to be a manual checklist for a user to perform every day.

The current implementation covers bronze daily market prices. Feature refresh, inference, and prediction persistence remain planned.

## Onboarding Prerequisite

Run the market data onboarding job during first setup and whenever new tickers are added to the active universe:

```powershell
uv run python -m swingtrader.data.jobs.onboard_market_data
```

The onboarding job creates first bronze rows for active tickers that have no stored daily prices. Already-onboarded tickers are skipped.

The daily update job below intentionally does not initialize missing tickers. It reports them as not onboarded and keeps already-onboarded active tickers current.

Onboarding is not the same as inference readiness or training eligibility. Those checks are documented in [Ticker Eligibility](../data/eligibility.md).

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

Use the market data onboarding job for first loads and newly active tickers before expecting the daily update job to refresh them.

## Developer Checks

Useful local checks:

```powershell
uv run pytest tests/data/jobs/test_update_market_data.py
uv run pytest tests/data/ingestion/test_market_data.py tests/data/ingestion/test_onboarding.py
```

## Planned Production Sequence

After model development establishes a selected feature set and model artifact, extend the scheduled workflow in this order:

1. Update bronze market data.
2. Resolve inference-ready tickers.
3. Calculate only the selected production features.
4. Run model inference using an explicit model and feature-set version.
5. Persist a dated prediction snapshot.
6. Expose the persisted snapshot through FastAPI.

The React frontend should read the latest persisted snapshot through the API. It should not cause steps 1 through 5 to run.

Macro-data ingestion and other context jobs remain later extensions.
