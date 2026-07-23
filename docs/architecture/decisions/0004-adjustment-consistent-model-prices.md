# ADR 0004: Adjustment-Consistent Model Prices

- **Status:** Accepted
- **Date:** 2026-07-23

## Context

Bronze market data contains raw `open`, `high`, `low`, and `close` together with `adjusted_close`. Several feature families compare prices across sessions. Using raw OHLC for some families and adjusted close for others lets stock splits and dividend adjustments appear as artificial gaps, directional movement, volatility, momentum, and market-structure changes.

Reusable indicators also serve notebooks, charting, and analysis where raw prices may be intentional. Encoding one source convention inside the indicator layer would reduce that reusability.

## Decision

Canonical model feature builders express the OHLC columns they consume on the adjusted-close scale. For each row, selected raw price columns are multiplied by `adjusted_close / close`, and transformed close is set directly to `adjusted_close`.

The transformation belongs to `swingtrader.data.features`, not `swingtrader.indicators`. Indicators remain source-agnostic and calculate from the price representation supplied by their caller. Source volume is retained when a feature combines an adjusted price series with volume. Turnover is the exception: it multiplies raw `close` by raw `volume`. A split moves price and share count inversely, so their product is already split-invariant and needs no adjustment. Raw `close * volume` is also the economically meaningful traded turnover.

Cross-session feature families must be covered by synthetic corporate-action tests showing that equivalent economic histories produce equivalent model features when one raw history encodes a split discontinuity.

## Consequences

- Trend, momentum, volatility, price-action, and market-structure model features use one economic price history.
- Same-session candle geometry is unchanged because every OHLC value on a row receives the same factor.
- Absolute indicator outputs remain expressed in the caller-supplied price units.
- Historical feature datasets and model artifacts must be regenerated after adopting this decision.
- Provider revisions to adjusted history can revise past feature values, so retained model experiments should eventually reference immutable data snapshots.
