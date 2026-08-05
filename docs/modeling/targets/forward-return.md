# Significant Forward Return Target

The forward-return outcomes are described by the `forward_return_targets:1` target set. Its deterministic manifest records ordered target families, parameters, required inputs, produced columns, builder import paths, and the maximum future horizon. The `significant_up_5d_classification` supervised-task specification selects `target_significant_up_5d` unambiguously from that set.

Target sets differ from feature sets because target families intentionally use future observations and expose `maximum_horizon_sessions`, which dataset construction uses to validate the selected task horizon. Temporal splitting uses each retained sample's actual `target_end_date`. Feature sets describe information available to the model at prediction time and must remain point-in-time safe.

The forward-return, triple-barrier, and cross-sectional target sets are independent target variants. None universally supersedes the others; each defines a distinct learning objective and can be generated on its own from canonical prices. A behavior or parameter change that alters an existing target's meaning must create a new version of that target set rather than silently changing an established experiment contract. Exact reproduction also requires the source revision containing the configured target builders.

This page defines the forward-return target contract. The next-open stop-loss and take-profit contract is documented in [Triple Barrier Target](triple-barrier.md).
The label-generation code for this contract is implemented in the modeling datasets package, the in-memory OHLCV feature set is implemented in the data package, canonical unsplit temporal dataset construction is documented in [Temporal Datasets](../temporal-datasets.md), and fixed leakage-safe assignment is documented in [Temporal Splitting](../temporal-splitting.md). Shared reporting guidance is documented in [Model Evaluation](../evaluation.md). Model training, persistence, inference, and backtesting remain follow-up implementation work.

| Property | Value |
| --- | --- |
| Target set | `forward_return_targets:1` |
| Learning task | Binary classification |
| Selected task output | One selected target value per sample; not multiclass or multilabel |
| Primary target | `target_significant_up_5d` |
| Supporting outcomes | `forward_return_5d`, `forward_return_10d`, `forward_return_15d` |
| Signal date | Completed daily bar on session `t` |
| Resolution date | Fifth observed session after the signal |
| Positive class | Five-session adjusted-close return exceeds the economic threshold |
| Intended use | Probability estimation and cross-sectional candidate ranking |
| Main limitation | Close-to-close research target; not an executable trade path |

## Input Frame Contract

Modeling uses the same canonical market-price representation as indicators and features: a unique, sorted `MultiIndex` named `provider`, `ticker`, and `trading_date`, in that exact order. Those identifiers must not also be ordinary columns. Bronze loader output is column-oriented and should be converted once at the caller boundary:

```python
from swingtrader.modeling.datasets import FORWARD_RETURN_TARGET_SET

prices = (
    prices
    .set_index(["provider", "ticker", "trading_date"])
    .sort_index()
)
labels = FORWARD_RETURN_TARGET_SET.apply(prices)
```

Target builders preserve this index. Feature and label frames can therefore be aligned or joined directly without reconstructing observation identity.

## Model Objective

The model should estimate the probability that a stock produces a meaningful positive adjusted-close return over the next five observed trading sessions.

The model is intended primarily as a candidate-ranking tool. A useful model should assign progressively higher probabilities to stocks with progressively better realized outcomes.

Five trading sessions are the primary horizon because the model should identify stocks where a meaningful upward move may begin soon. This does not require an eventual trading strategy to hold every position for exactly five sessions.

## Continuous Outcomes

For ticker `i` on trading date `t`, the primary continuous outcome is:

```text
forward_return_5d = adjusted_close_at_t_plus_5 / adjusted_close_at_t - 1
```

The horizon is measured in observed trading sessions for the ticker, not calendar days.

Adjusted close is used so that historical corporate actions do not create artificial research-label returns.

The label generator also calculates these diagnostic continuous outcomes:

```text
forward_return_10d
forward_return_15d
```

The 10-session and 15-session outcomes are initially for EDA and later model comparison. They are not primary classification targets.

Rows without the required future observation for a horizon must have a missing outcome for that horizon. They must not be assigned to the negative class.

## Primary Binary Target

The primary target is:

```text
target_significant_up_5d = forward_return_5d > return_threshold
```

The positive class represents a return that exceeds both round-trip courtage and the selected economic return hurdle.

The simpler target `forward_return_5d > 0` is intentionally not used because it would classify negligible positive price noise as a successful outcome.

## Return Threshold

The return threshold accounts for:

- courtage of `0.25%` of transaction value on both purchase and sale;
- a net five-session profit equivalent to a `50%` annualized return over `252` trading sessions.

Let:

```text
P = purchase value before courtage
S = sale value before courtage
c = courtage rate on each transaction
```

The net profit after courtage is:

```text
net_profit = S * (1 - c) - P * (1 + c)
```

The required five-session net return, measured relative to `P`, is:

```text
commission = 0.0025
annual_return_target = 0.50
trading_days_per_year = 252
prediction_horizon = 5

required_net_return =
    (1 + annual_return_target)
    ** (prediction_horizon / trading_days_per_year)
    - 1
```

The gross return threshold must satisfy:

```text
S * (1 - commission) - P * (1 + commission)
    = P * required_net_return
```

Dividing by `P` and solving for the gross adjusted-close return gives:

```text
return_threshold =
    (1 + commission + required_net_return)
    / (1 - commission)
    - 1
```

With these assumptions:

```text
required_net_return ~= 0.00807739
return_threshold ~= 0.01311017
```

The initial positive-class threshold is therefore approximately `1.311%` gross adjusted-close return over five observed trading sessions.

After courtage on both purchase and sale, this retains a net five-session profit equivalent to a `50%` annualized return, measured relative to purchase value before courtage.

The `50%` annualized hurdle is an initial modeling assumption. Its suitability should later be examined using label prevalence, return distributions, and model usefulness.

## Prediction And Execution Interpretation

Features may use the completed daily bar on date `t`, including its closing values.

The close-to-close target is therefore a research target and not a directly executable trade return. A model score produced using the completed bar at `t` could only be acted on after that bar is available.

This target does not attempt to model exact entry price, exit price, spread, slippage, order execution, stop-loss behavior, take-profit behavior, position sizing, or portfolio construction.

## Feature Scope

The model should use only features derived from available OHLCV history.

This target should not require:

- macroeconomic data;
- benchmark-index data;
- sector or industry data;
- fundamental company data;
- news or sentiment data.

Index-relative labels are covered by the separate [Cross-Sectional Return Targets](cross-sectional.md). The initial objective here is to determine whether OHLCV-derived features contain useful predictive and ranking signal on their own.

## Evaluation

Use the shared chronological validation, classification, calibration, ranking, and locked-test rules in [Model Evaluation](../evaluation.md). Reports for this target should additionally include:

- mean `forward_return_5d` by prediction quantile;
- positive-label rate by prediction quantile;
- mean `forward_return_5d` and hit rate among top-ranked candidates;
- per-date Spearman correlation between predicted probability and `forward_return_5d`.

Baselines should include:

- a dummy probability classifier based on the training-set class prior;
- random candidate selection from the same dates and eligible stock universe.

A future evaluation may also compare the model with the equal-weighted return of the available stock universe. A formal benchmark-index comparison is deferred until the project has index data, a defined candidate-selection rule, realistic execution assumptions, and an end-to-end strategy simulation.

## Assumptions And Limitations

This target assumes:

- a five-session primary prediction horizon;
- adjusted close for research labels;
- OHLCV-derived model features only;
- the currently available ticker universe;
- chronological validation;
- proportional courtage of `0.25%` on both purchase and sale.

This target does not yet account for:

- minimum courtage amounts or alternative courtage classes;
- bid-ask spread;
- slippage;
- liquidity-dependent execution;
- exact executable entry and exit prices;
- stop-loss or take-profit rules;
- overlapping simultaneous positions;
- capital allocation or portfolio constraints;
- survivorship or historical-universe changes beyond the available ticker data.

These limitations should be retained when interpreting initial model results.
