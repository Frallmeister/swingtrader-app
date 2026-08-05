# ADR 0009: Discretionary Replay State Machine

- Status: Accepted
- Date: 2026-08-05

## Context

The manual backtest requires realistic next-open decisions, mandatory review of every position, morning revisions after observing gaps, partial exits, configurable barriers, courtage, and persistent auditability. Treating these as independent frontend controls would permit invalid temporal transitions and look-ahead leakage.

The application also needs charting and screening before a trained model exists. Existing public indicator functions already form the reusable numerical boundary for notebooks, future APIs, and frontend charting.

## Decision

Implement replay as a Python application service with an explicit phase state machine:

```text
Evening[t] -> Morning[t+1] -> Evening[t+1]
```

The FastAPI backend owns phase transitions, visible market data, order validation, courtage, cash, barrier resolution, positions, metrics, and persistence. React owns presentation and transient editing state. The frontend never reads the database directly.

Store normal current-state tables for simple resume plus an append-only replay event table for auditability. Do not introduce a general event-sourcing framework.

Version one uses raw OHLCV and every public indicator. Indicator signatures provide calculation parameters, while a small registry provides output names and chart metadata. Multi-output indicators are calculated once as a group; chart outputs are independently visible and every output is independently available to continuous screening expressions.

Model predictions and feature blocks are deferred. They may later be added as optional screening data without changing replay execution.

## Consequences

- Future candles are withheld by the backend rather than hidden in React.
- Every open position requires an explicit evening decision.
- Morning sales and reductions precede buys; unaffordable buys are rejected rather than resized silently.
- Courtage and same-candle ambiguity policy are snapshotted when the replay is created.
- The first release remains useful for indicator-driven discretionary trading.
- The registry adds limited metadata but does not duplicate indicator calculations or create a second feature framework.
