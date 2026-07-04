# Ticker Eligibility

Ticker eligibility separates desired universe configuration from actual data readiness.

## Concepts

The active trading universe is the deployed production candidate set. Runtime data jobs use it to decide which tickers should be kept updated for live screening.

Inference-ready tickers have enough recent data to be ranked by the production model. An active ticker is not automatically inference-ready.

Training-eligible tickers have enough historical data to be included safely in model training and evaluation. A ticker can be training-eligible without being active for live trading.

The future training universe may be broader than the active trading universe. It may include inactive tickers, broader Swedish universes, and eventually non-Swedish markets.

## Version 1 Bronze Rules

The first implementation checks bronze daily price state only. Feature and label tables do not exist yet, so feature and label rules are documented as future hard blockers.

For inference readiness, a ticker must have:

- at least 252 bronze daily price rows;
- latest bronze `trading_date` no more than 4 calendar days before the reference date;
- no missing `adjusted_close` values;
- no more than 5% rows with null or zero `volume`;
- valid `close * volume` observations for at least 95% of the latest 60 bronze rows;
- median `close * volume` of at least SEK 5,000,000 over those latest valid turnover observations.

For training eligibility, a ticker must have:

- at least 756 bronze daily price rows;
- no missing `adjusted_close` values;
- no more than 5% rows with null or zero `volume`;
- valid `close * volume` observations for at least 95% of the latest 60 bronze rows;
- median `close * volume` of at least SEK 5,000,000 over those latest valid turnover observations.

The 4-calendar-day inference recency rule is a pragmatic first threshold. The trading horizon may be as short as 5 days, so stale inputs can make rankings irrelevant; this threshold should be calibrated later against market calendars, realistic run schedules, and model sensitivity to data lag.

The liquidity threshold assumes the current Swedish trading focus. Broader training universes may include foreign stocks, so foreign-market prices should be converted to SEK before applying this threshold or replaced with market-specific liquidity rules.

## Implemented Checks

Use `check_inference_readiness(...)` to evaluate production inference readiness:

```python
from datetime import date

from swingtrader.data.eligibility import check_inference_readiness

result = check_inference_readiness(reference_date=date(2026, 7, 4))

print(result.ready_tickers)
print(result.not_ready_tickers)
```

Use `check_training_eligibility(...)` to evaluate training candidates:

```python
from swingtrader.data.eligibility import check_training_eligibility

result = check_training_eligibility(tickers=("AAK.ST", "BOL.ST"))

print(result.eligible_tickers)
```

When no explicit tickers are passed, the checks resolve the active trading universe. Passing explicit tickers lets future dataset code evaluate a broader training universe without treating active status as a training rule.

## Future Feature And Label Rules

Once feature and label tables exist, inference readiness must also require recent feature rows for the production model input window.

Training eligibility must also require enough feature and label rows for the intended train, validation, and test split design.

Those future checks should extend the eligibility module rather than changing bronze onboarding. Onboarding remains responsible only for deciding whether active tickers have any bronze daily price rows.