# Target And Evaluation

This page defines the V1 model target and evaluation contract.

The label-generation code for this contract is implemented in the modeling datasets package. Feature engineering, temporal dataset construction, model training, evaluation code, persistence, inference, and backtesting remain follow-up implementation work.

## V1 Model Objective

The V1 model should estimate the probability that a stock produces a meaningful positive adjusted-close return over the next five observed trading sessions.

The model is intended primarily as a candidate-ranking tool. A useful model should assign progressively higher probabilities to stocks with progressively better realized outcomes.

Five trading sessions are the primary horizon because the model should identify stocks where a meaningful upward move may begin soon. This does not require an eventual trading strategy to hold every position for exactly five sessions.

## Continuous Outcomes

For ticker `i` on trading date `t`, the primary continuous outcome is:

```text
forward_return_5d =
    adjusted_close_at_t_plus_5
    / adjusted_close_at_t
    - 1
```

The horizon is measured in observed trading sessions for the ticker, not calendar days.

Adjusted close is used so that historical corporate actions do not create artificial research-label returns.

The implemented V1 label generator also calculates these diagnostic continuous outcomes:

```text
forward_return_10d
forward_return_15d
```

The 10-session and 15-session outcomes are initially for EDA and later model comparison. They are not primary V1 classification targets.

Rows without the required future observation for a horizon must have a missing outcome for that horizon. They must not be assigned to the negative class.

## Primary Binary Target

The primary V1 target is:

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

With the V1 assumptions:

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

V1 does not attempt to model exact entry price, exit price, spread, slippage, order execution, stop-loss behavior, take-profit behavior, position sizing, or portfolio construction.

## Feature Scope

V1 should use only features derived from available OHLCV history.

V1 should not require:

- macroeconomic data;
- benchmark-index data;
- sector or industry data;
- fundamental company data;
- news or sentiment data.

Index-relative labels are deferred. The initial objective is to determine whether OHLCV-derived features contain useful predictive and ranking signal on their own.

## Validation Contract

Evaluation must use chronological validation. Random row-level splitting is not acceptable.

The eventual temporal-dataset implementation must ensure that future observations used to construct labels cannot leak into training features or cross-validation boundaries.

The exact walk-forward schedule, split dates, and any purge or embargo implementation should be defined by the later temporal-dataset task.

Evaluation reports should include:

- evaluated date range;
- number of observations;
- number of unique tickers;
- positive-class prevalence;
- number of evaluated trading dates.

## Classification Evaluation

The model should be evaluated as a probability classifier.

Classification evaluation should include at least:

- precision-recall AUC;
- ROC AUC;
- log loss or Brier score;
- positive-class precision;
- positive-class recall;
- positive-class prevalence.

Accuracy alone is insufficient because it can be misleading when the positive class is uncommon.

Any decision threshold used to convert probabilities into predicted classes must be reported separately from the return threshold used to generate the target.

## Calibration Evaluation

Predicted probabilities should be evaluated for calibration by predicted-probability bucket.

Each calibration bucket should report at least:

- predicted-probability range;
- number of observations;
- mean predicted probability;
- realized positive-label rate.

Calibration reporting should make it visible whether a predicted probability can be interpreted as a meaningful probability, not only as a ranking score.

## Ranking Evaluation

The model should also be evaluated as a ranking model.

Ranking metrics should be calculated cross-sectionally within each evaluation date among the eligible stocks scored on that date, then summarized across dates.

Prediction deciles should be formed separately for each date where enough candidates are available. Spearman correlation should likewise be calculated per date and summarized across dates rather than calculated only from all observations pooled together.

Top-ranked evaluation must state the selection rule, such as a fixed `top_k` or top prediction decile, and report the number of selected candidates per date. Comparisons with random selection must use the same dates and the same number of selected candidates.

Ranking evaluation should include at least:

- mean `forward_return_5d` by prediction decile;
- positive-label rate by prediction decile;
- mean `forward_return_5d` among top-ranked candidates;
- hit rate among top-ranked candidates;
- Spearman correlation between predicted probability and `forward_return_5d`;
- number of candidates generated per date or week.

The desired result is monotonic ranking behavior: higher model scores should correspond to progressively higher realized returns and positive-label rates.

Top-ranked results must include the number of selected observations so that strong performance from very few candidates is not mistaken for broadly useful ranking performance.

## Baselines

The V1 model should be compared with:

- a dummy probability classifier based on the training-set class prior;
- random candidate selection from the same dates and eligible stock universe.

A future evaluation may also compare the model with the equal-weighted return of the available stock universe.

A formal benchmark-index comparison is deferred until the project has index data, a defined candidate-selection rule, realistic execution assumptions, and an end-to-end strategy simulation.

## Assumptions And Limitations

V1 assumes:

- a five-session primary prediction horizon;
- adjusted close for research labels;
- OHLCV-derived model features only;
- the currently available ticker universe;
- chronological validation;
- proportional courtage of `0.25%` on both purchase and sale.

V1 does not yet account for:

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

## Non-Goals

This contract does not implement or define production behavior for:

- feature engineering;
- temporal dataset construction;
- dataset splitting;
- model training or tuning;
- backtesting;
- database schemas;
- feature or label persistence;
- macro or index ingestion;
- index-relative targets;
- executable trading rules;
- stop-loss or take-profit simulation;
- position sizing;
- portfolio construction;
- production inference;
- a web interface.

Database persistence, temporal dataset construction, and executable training/evaluation workflows should be handled by separate follow-up implementation issues.