# Backtesting Pilot

The backtesting pilot converts historical daily candidate scores into a small,
auditable portfolio simulation. It implements the executable evaluation layer
required by
[ADR 0003](../architecture/decisions/0003-separate-research-targets-from-executable-evaluation.md)
without introducing a general backtesting framework.

## Scope

The pilot supports:

- long positions only;
- raw daily `open`, `high`, `low`, and `close` prices;
- ranked entry candidates;
- next-session-open entries and timeout exits;
- ATR-based risk sizing, stops, and fixed take-profit levels;
- a maximum number of concurrent positions;
- proportional entry and exit commissions;
- conservative same-bar stop/target resolution;
- a transaction table, daily equity table, and compact metric summary.

It intentionally does not model short positions, leverage, partial fills,
slippage, taxes, dividends, currency conversion, trailing stops, or intraday
price paths.

## Input Frames

Both inputs use the canonical unique, sorted `MultiIndex`:

```text
provider, ticker, trading_date
```

`prices` must contain raw tradable OHLC columns:

```text
open, high, low, close
```

`adjusted_close` is not used for execution. Historical adjusted values are
revised backwards for distributions and are therefore not prices that could
have been traded at the time.

`signals` must contain:

```text
score, atr
```

Higher scores are considered first. Filter the signal frame before the call if
a model-specific decision threshold should apply. The `atr` column must be
calculated from raw OHLC prices available through the signal date, not from the
adjustment-consistent model-price representation.

```python
from swingtrader.indicators import atr
from swingtrader.modeling.backtest import run_backtest

signals = predictions.loc[:, ["score"]].copy()
signals["atr"] = atr(prices.loc[:, ["high", "low", "close"]], length=14)
signals = signals.loc[signals["score"] >= decision_threshold]
```

## Daily Procedure

For each completed session `t`:

1. execute previously planned timeout exits at the next open;
2. execute opening-gap stop or take-profit exits;
3. execute ranked entry orders planned after the previous close;
4. evaluate the day's high and low for fixed stop or target touches;
5. mark remaining positions to the raw close;
6. schedule timeout exits and new entries for the next observed session.

Position size is planned after the close using current marked-to-market equity:

```text
risk budget = equity * risk_fraction
ATR distance = signal ATR * stop_atr_multiple
planned shares = floor(risk budget / ATR distance)
```

At the next open:

```text
stop = entry open - ATR distance
take profit = entry open + reward_risk_ratio * ATR distance
```

The actual quantity is reduced if the planned purchase and entry commission do
not fit available cash. Timeout exits execute before entries so a position slot
and its cash can be reused at the same open.

Daily OHLC cannot reveal whether a stop or target was touched first when both
occur inside the same bar. The pilot assumes the stop was reached first and
records `ambiguous_intrabar = true` on that transaction.

## Running the Pilot

```python
result = run_backtest(
    prices,
    signals,
    initial_cash=100_000,
    risk_fraction=0.005,
    max_positions=10,
    max_holding_sessions=5,
    stop_atr_multiple=1.0,
    reward_risk_ratio=2.0,
    commission_rate=0.0025,
)

trades = result["trades"]
equity = result["equity"]
summary = result["summary"]
```

The result contains no daily candidate-rejection ledger. Only completed
transactions and the compact portfolio equity history are retained.

## Metrics

The summary reports:

- `win_rate`;
- `average_reward_risk_win`;
- `average_reward_risk_loss`;
- `total_reward_risk`;
- `total_return`;
- `expectancy`;
- `average_holding_sessions`;
- `commissions_paid`.

Reward/risk values use net profit after commissions divided by the trade's
initial ATR risk. `total_reward_risk` is the sum of trade reward/risk values and
`expectancy` is their arithmetic mean. The fixed take-profit ratio is a gross
price-distance rule, so commissions make a take-profit transaction slightly
less than the configured reward/risk ratio on a net basis.

## Pilot Limitations

The current active ticker list does not by itself provide historical index
membership. The caller remains responsible for supplying point-in-time-safe
signals and an appropriate historical universe. Open positions are liquidated
at the final available close so the final portfolio return is fully reported.
