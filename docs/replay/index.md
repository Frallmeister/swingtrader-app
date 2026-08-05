# Discretionary Replay

The replay application is a local decision-support environment for stepping through historical daily market data without revealing future information. It is deliberately useful before a screening model exists: the first version screens and charts the reusable indicators in `swingtrader.indicators` directly.

## Workflow

A replay alternates between two explicit phases:

```text
Evening[t] -> Morning[t+1] -> Evening[t+1]
```

During an evening, the complete daily candle is visible. Every open position must receive an explicit `keep`, `reduce`, or `sell` decision before the replay can advance. New buys remain provisional.

The following morning reveals only the new opening prices. Evening decisions may be retained, revised, or cancelled before execution. Confirmed sells and reductions execute before buys, and a buy is rejected when its gross value plus courtage exceeds the available cash.

After morning execution, the rest of that day's candle is revealed and intraday stop/target barriers are processed. A replay chooses one immutable ambiguity policy when created:

- stop first;
- target first;
- the same deterministic candle path used by the triple-barrier target.

## Included capabilities

- resumable replay sessions identified by UUID;
- Avanza Mini, Small, Medium, and Fixed courtage snapshots;
- whole-share buys, complete exits, and partial reductions;
- daily stop-loss and take-profit revision;
- watchlists for delayed discretionary entries;
- configurable chart indicators with every public output returned as one group;
- independently toggleable chart outputs and independently screenable outputs;
- continuous screening expressions such as `close / EMA(10)`;
- saved screening presets;
- fills and an append-only decision/event audit trail;
- return, expectancy, win rate, cumulative R, Sharpe, Sortino, and metric history;
- current-position annualized simple return versus trading sessions held.

## Information boundary

The backend enforces visibility. In morning mode it does not return the current day's high, low, or close to the browser. Lightweight Charts therefore cannot reveal candles that the replay has not reached.

Models and model-derived features are intentionally outside the first version. Future prediction scores can become additional continuous screening operands without changing replay execution.
