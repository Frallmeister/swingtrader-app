# Triple-Barrier Target

The `triple_barrier_targets:1` target set represents outcomes with one direct three-class label. It asks:

> After a signal on session `t` and entry at the next observed open, which occurs first: the ATR-scaled take-profit barrier, the ATR-scaled stop-loss barrier, or the time barrier?

| Property | Value |
| --- | --- |
| Target set | `triple_barrier_targets:1` |
| Learning task | Three-class classification |
| Primary target | `triple_barrier_label_5d` |
| Entry rule | Next observed session open |
| Positive outcome | `1`: take-profit first |
| Neutral outcome | `0`: timeout |
| Negative outcome | `-1`: stop-loss first |
| Main limitation | Daily OHLC cannot reveal which barrier was touched first inside one bar |

The triple-barrier set is an independent target variant. It is generated on its own from canonical prices and does not include forward-return or cross-sectional outputs. It defines a distinct learning objective and does not supersede the other target variants.

## Default Contract

| Parameter | Default |
| --- | ---: |
| ATR length | 14 sessions |
| Stop distance | `2.0 * ATR_t` |
| Reward/risk ratio | `2.0` |
| Horizons | 5, 10, and 15 observed sessions |
| Same-bar policy | `stop_loss_first` |

For entry price `E`, signal-row ATR `A`, stop multiple `k`, and reward/risk ratio `R`:

```text
stop_price = E - k * A
take_profit_price = E + R * k * A
```

Raw OHLC values are first expressed on the adjusted-close scale using `adjusted_close / close`. ATR uses only data available through the completed signal bar. Entry and barrier observations are future information used only to construct labels.

## Label and Time Semantics

For each horizon `h`, this target set emits two user-facing outcome columns:

| Column | Meaning |
| --- | --- |
| `triple_barrier_label_{h}d` | `1` for take-profit first, `0` for timeout, `-1` for stop-loss first |
| `time_to_event_{h}d` | 1-based session of a barrier hit, or `h` for timeout |

It also emits `target_end_date_{h}d`, the corresponding resolution date required by purged temporal splitting. This is dataset metadata rather than a second representation of the outcome.

A successfully labeled row always has both a label and a time. For example:

| Outcome | Label | Time for a 5-session horizon |
| --- | ---: | ---: |
| Take-profit on session 2 | `1` | `2` |
| Timeout | `0` | `5` |
| Stop-loss on session 3 | `-1` | `3` |

Rows remain missing when:

- the data ends before a barrier hit and before the full timeout horizon;
- the signal-row ATR or required OHLC data is invalid;
- the configured stop would be non-positive; or
- both barriers are touched in one bar and `intrabar_policy="exclude"` is used.

A barrier hit before the available data ends remains labelable even when the complete timeout horizon is unavailable.

## Observed-Session Semantics

Session 1 is the next observed row after the signal row and supplies the entry open. A 5-session label evaluates sessions `t+1` through `t+5`; horizons do not count calendar days.

Input must use the canonical, unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`. Required value columns are `open`, `high`, `low`, `close`, and `adjusted_close`.

## Gap and Same-Bar Handling

Each future session checks the open before the intrabar range:

1. `open <= stop_price` gives label `-1`;
2. `open >= take_profit_price` gives label `1`;
3. otherwise, high and low are evaluated.

When one daily bar touches both barriers, supported policies are:

| Policy | Resolution |
| --- | --- |
| `stop_loss_first` | label `-1`; conservative default |
| `take_profit_first` | label `1` |
| `candle_path` | green/doji: stop first; red: take-profit first |
| `exclude` | leave label, time, and resolution date missing |

The chosen policy is recorded in the target-family manifest. This target set intentionally does not add another ambiguity flag; callers that need ambiguity analysis should compare separately generated target sets with different policies.

## Using the Builder

```python
from swingtrader.modeling.datasets import add_triple_barrier_targets

prices = (
    prices
    .set_index(["provider", "ticker", "trading_date"])
    .sort_index()
)

labeled = add_triple_barrier_targets(
    prices,
    atr_length=14,
    stop_atr_multiple=2.0,
    reward_risk_ratio=2.0,
    horizons=(5, 10, 15),
    intrabar_policy="stop_loss_first",
)
```

Generate the triple-barrier target set, which produces only triple-barrier outputs, with:

```python
from swingtrader.modeling.datasets import generate_triple_barrier_labels

labeled = generate_triple_barrier_labels(prices)
```

The current baseline harness and evaluation reports are binary-specific. This target set can be used by the temporal dataset layer, but multiclass estimators and evaluation must handle `-1`, `0`, and `1` explicitly. Estimators requiring zero-based class IDs should encode the labels at the model boundary without changing the stored target contract.
