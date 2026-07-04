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

The job derives each ticker's update plan from bronze storage. Tickers with no bronze rows are reported as not onboarded and skipped; use the bronze onboarding workflow before expecting daily updates for newly active tickers.

## Failure Handling

By default, ticker-level failures are logged and successful tickers are still written. Use this when a scheduler should fail on any ticker-level failure:

```powershell
uv run python -m swingtrader.data.jobs.update_market_data --fail-on-ticker-failure
```

