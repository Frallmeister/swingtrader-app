# Model Evaluation

The baseline evaluation layer consumes one standardized prediction frame for one temporal split. It reports pooled binary-classification quality, probability calibration, daily cross-sectional ranking, date-matched random comparisons, and dataset context through a reusable `EvaluationReport`.

See [Baseline Models and Evaluation Harness](baseline-models.md) for fitting and orchestration. Target pages define outcome meaning; this page defines how model scores are evaluated.

## Temporal Validation Contract

Evaluation uses chronological split assignments from `FixedTemporalSplitter`. Random row-level splitting is not acceptable. The splitter applies shared calendar ranges, purges samples whose actual `target_end_date` crosses a split boundary, and optionally embargoes additional observed signal dates. See [Temporal Splitting](temporal-splitting.md).

Each report records:

- split date range, row count, trading-date count, and ticker count;
- target prevalence and continuous-outcome coverage;
- feature count and missingness summary;
- classification threshold, calibration buckets, score quantiles, `top_k`, random seed, and ranking-return source column;
- pooled metrics and per-date tables.

Validation and test are evaluated independently. A validation report does not require or read locked-test rows.

## Classification Evaluation

The aggregate and per-date tables include:

- precision-recall AUC;
- average precision;
- ROC AUC;
- log loss;
- Brier score;
- positive-class precision and recall;
- positive-class prevalence and row count.

Precision-recall AUC and ROC AUC are undefined for dates without the required target classes and are stored as missing values in the per-date table. MLflow receives only finite aggregate metrics; the complete tables remain available as artifacts.

Accuracy is intentionally omitted from the required report because it can look strong when the positive class is uncommon. The threshold used to create `predicted_class` is an explicit `EvaluationConfig` value and is not the same as the return threshold or barrier definition used to create the target.

## Calibration Evaluation

Scores are divided into fixed-width probability buckets. Each bucket records its bounds, sample count, mean score, and observed positive rate. Empty buckets are retained so reports from different models have the same shape.

The calibration plot compares mean predicted score with observed positive rate and includes the ideal diagonal. Calibration matters when a score will later inform filtering, risk, or position sizing; ranking quality alone does not make a score a reliable probability.

## Cross-Sectional Ranking Evaluation

Ranking is calculated within each trading date because the product compares stocks available at the same decision time. The evaluator assigns daily score quantiles, selects the daily top-`k`, and computes daily Spearman correlation when both score and continuous outcome vary.

The report contains:

- continuous return and positive-label rate by daily score quantile;
- daily top-`k` return and positive rate;
- top-quantile return and positive rate;
- per-date Spearman correlation;
- a deterministic date-matched random top-`k` comparison;
- the empirical distribution of model and random top-`k` returns across dates.

`score_quantiles_by_date.csv` retains one row per represented date and quantile. `score_quantiles.csv` summarizes those rows with equal weight per date rather than allowing dates with larger candidate universes to dominate. When a daily universe is smaller than the requested number of quantiles, the lowest and highest candidates still occupy the boundary quantiles and intermediate buckets may be empty.

Exact score ties are resolved with a deterministic pseudo-random secondary key derived from `EvaluationConfig.random_seed` and canonical sample identity. Consequently, a constant-score model has the same top-`k` selections and quantile outcomes as the date-matched random comparator rather than inheriting ticker or row order as an accidental ranking signal.

Top-`k` return lift uses only dates where both the model and random selections have an observed continuous outcome. The aggregate report records `top_k_return_comparison_date_count`, and the per-date tables record the number of observed ranking returns in each selected set. Positive-rate comparisons use every date because the binary target is complete.

The desired behavior is monotonic: higher scores should correspond to progressively stronger outcomes and positive-label rates. A single strong top bucket is useful but less convincing than a stable progression across quantiles and dates.

## Research Outcome Boundary

The `ranking_return` column is copied from the explicitly configured target column, whose name is retained in the report manifest and Markdown report. It is a research diagnostic rather than an executable trade return: it does not apply next-session entry assumptions and excludes transaction costs, spreads, slippage, stop-loss or take-profit execution, position sizing, and portfolio constraints. Do not interpret ranking-return tables or plots as strategy P&L.

## Dataset and Missingness Context

Feature missingness is measured on the evaluated split before model-specific imputation. The report records missing counts and fractions by feature, the number of affected features, and the total missing feature-cell fraction. This context helps distinguish model changes from changes in data coverage or warm-up behavior.

For the logistic baseline, the missingness table describes raw evaluation inputs; actual imputation uses medians fitted only on training rows and retained in the model artifact.

## Reproducibility

`EvaluationConfig` retains every report-level choice:

```python
from swingtrader.modeling.training import EvaluationConfig

config = EvaluationConfig(
    classification_threshold=0.5,
    calibration_bins=10,
    score_quantiles=10,
    top_k=5,
    random_seed=23,
)
```

The random comparison and score-tie resolution derive stable scores from the random seed and canonical sample identity, so they are independent of row iteration order. When the harness is used, this seed must match the evaluation seed declared in `ExperimentSpec.random_seeds`. JSON, CSV, Markdown, compressed prediction, and SVG artifacts are written deterministically for the same report inputs.

## Locked-Test Policy

The locked test is used only after model configuration, preprocessing, hyperparameters, threshold, and ranking rule have been selected from train and validation. Inspecting the test and then changing those choices turns it into another validation set and requires a new locked period for an unbiased final estimate.
