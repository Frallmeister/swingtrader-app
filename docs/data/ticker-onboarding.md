# Ticker Onboarding

Ticker activation is a data event, not only a config edit.

## Current Scope

The implemented onboarding sync is bronze-only. It compares the active trading universe with `bronze_market_daily_prices` for a provider.

A ticker is:

- `missing` if no bronze daily price rows exist;
- `onboarded` if at least one bronze daily price row exists.

This deliberately avoids deciding whether the ticker has enough history for inference or training.

## Backfill

When `backfill=True`, the sync calls historical ingestion for missing tickers only. Existing onboarded tickers are skipped even if they have short histories.

```python
from datetime import date

from swingtrader.data.ingestion.onboarding import sync_active_ticker_bronze_onboarding

result = sync_active_ticker_bronze_onboarding(
    start_date=date(2024, 1, 1),
    end_date=date(2026, 1, 1),
    backfill=True,
)
```

## Future Readiness

Later work should add rules for:

- recent feature availability;
- minimum history length;
- enough label rows for training;
- data quality warnings and hard exclusions;
- inference-ready versus training-eligible tickers.