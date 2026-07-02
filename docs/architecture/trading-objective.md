# Trading Objective

Swingtrader is intended to become a personal, long-running trading decision-support system.

## Target Use

- Maintain a curated active trading universe, initially Swedish Large Cap and Mid Cap equities.
- Keep market data up to date for that active universe.
- Generate model-ready features and labels from reproducible stored data.
- Rank trade candidates over a swing-trading horizon of roughly 5-10 trading days up to a few weeks.
- Support manual order placement with better discipline, repeatability, and risk awareness.

## Non-Goals

- Automatic broker execution.
- High-frequency trading.
- Multi-user SaaS workflows.
- Treating model output as a complete trade decision.

## Future Decision Support

The frontend should eventually help answer:

- Which active tickers are currently strongest candidates?
- Why is a ticker highly ranked?
- Is the ticker data fresh enough for inference?
- What position size is reasonable for the planned risk?
- What stop-loss level makes the trade invalid?

The system should reduce subjective and inconsistent decisions without removing human review.