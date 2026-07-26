# Model Evaluation

Evaluation is a downstream consumer of a split-aware temporal dataset. The target pages define outcome meaning; this page defines the common evaluation principles that should apply when baseline training and reporting are implemented.

## Temporal Validation Contract

Evaluation must use chronological validation. Random row-level splitting is not acceptable. `FixedTemporalSplitter` applies shared chronological ranges, purges samples whose actual `target_end_date` crosses a split boundary, and optionally embargoes additional observed signal dates. See [Temporal Splitting](temporal-splitting.md).

Every report should identify:

- the experiment and dataset manifest digests;
- the evaluated date range;
- the number of observations, trading dates, and unique tickers;
- target prevalence or distribution;
- the candidate-selection rule when ranking metrics are reported.

## Classification Evaluation

Probability classifiers should report at least precision-recall AUC, ROC AUC, log loss or Brier score, positive-class precision and recall, and positive-class prevalence. Accuracy alone is insufficient when the positive class is uncommon.

A decision threshold that converts probabilities into predicted classes is separate from any economic threshold used to construct the target and must be reported separately.

## Calibration Evaluation

Calibration should be evaluated by predicted-probability bucket. Each bucket should report its probability range, observation count, mean predicted probability, and realized positive-label rate. This shows whether a score can be interpreted as a probability rather than only as an ordering signal.

## Ranking Evaluation

Ranking metrics should be calculated cross-sectionally within each evaluation date and then summarized across dates. Prediction quantiles and rank correlations must not be calculated only from all observations pooled together.

Reports should include the selection rule, such as a fixed `top_k` or top prediction decile, together with the number of selected candidates per date. Random-selection baselines must use the same dates and candidate counts.

For targets with a corresponding continuous outcome, useful ranking diagnostics include:

- mean realized outcome by prediction quantile;
- positive-label rate by prediction quantile;
- mean realized outcome and hit rate among top-ranked candidates;
- per-date Spearman correlation between score and continuous outcome;
- number of candidates generated per date or week.

The desired behavior is monotonic: higher model scores should correspond to progressively stronger realized outcomes and positive-label rates.

## Locked-Test Policy

The locked test is used only after model configuration, preprocessing, hyperparameters, and decision rules have been selected using train and validation data. Inspecting the test and then changing those choices turns the test into another validation set and requires a new locked period for an unbiased final estimate.
