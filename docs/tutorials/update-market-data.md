# Update Market Data

Use this to update bronze daily market prices for the active trading universe.

## Smoke Run

Start with a small ticker limit:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3
```

## Deterministic Run

Pass an exclusive end date when you want a repeatable local run:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --limit 3 --end-date 2026-07-04
```

## Full Local Run

Run without a limit to update all active tickers:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data
```

The job derives each ticker's update plan from bronze storage. Tickers with no bronze rows start from the configured initial start date in `src/swingtrader/configs/market_data.yml`.

## Failure Handling

By default, ticker-level failures are logged and successful tickers are still written. Use this when a scheduler should fail on any ticker-level failure:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --fail-on-ticker-failure
```

## Optional Backfill

Use `--backfill` to run the explicit bronze onboarding sync before recent update planning:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --backfill
```