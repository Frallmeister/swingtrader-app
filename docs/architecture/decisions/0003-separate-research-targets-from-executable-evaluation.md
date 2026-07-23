# ADR 0003: Separate Research Targets from Executable Evaluation

- **Status:** Accepted
- **Date:** 2026-07-23

## Context

The V1 label uses adjusted-close returns from the completed bar on date `t` to a future observed session. It is suitable for testing whether OHLCV-derived features contain predictive and ranking signal, but it is not a directly executable trade return because the completed bar is only known after its close.

Mixing signal discovery with entry assumptions, spread, slippage, stops, position sizing, and portfolio constraints would make the first modeling iteration harder to interpret.

## Decision

Treat the current close-to-close label and classification threshold as a research target. Use it for leakage-safe model development, probability evaluation, and daily cross-sectional ranking evaluation.

Evaluate executable strategy performance in a separate later layer with explicit next-session entries, exits, transaction costs, risk rules, position sizing, and portfolio constraints. Research results must not be described as executable returns without that second evaluation layer.

## Consequences

- Initial model work can isolate predictive signal before strategy mechanics are introduced.
- Research metrics and strategy metrics remain conceptually distinct.
- Model artifacts must record their label definition and parameters.
- A model that ranks the research target well may still fail executable backtesting and should not be promoted automatically.
