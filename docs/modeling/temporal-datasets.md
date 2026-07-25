# Temporal Datasets

The temporal dataset layer creates the canonical unsplit modeling product between feature/target generation and later temporal splitting or model training.

## Dataset Specification

`TemporalDatasetSpec` contains only choices that determine the unsplit data:

- a versioned `FeatureSetSpec`;
- a versioned `TargetSetSpec`;
- one `SupervisedTaskSpec`;
- a resolved `UniverseSpec` with concrete provider and ticker membership;
- an inclusive `data_cutoff`.

Split dates, model hyperparameters, random seeds, and MLflow are intentionally absent. An `ExperimentSpec` exposes the same lower-level contract through `experiment_spec.dataset_spec`.

## Construction

```python
from swingtrader.modeling.datasets import build_temporal_dataset

bundle = build_temporal_dataset(
    engine=engine,
    spec=experiment_spec.dataset_spec,
)
```

The builder loads all required bronze source columns through the cutoff and computes every configured feature block over the complete legitimate historical prefix. Targets are computed independently from the same canonical price frame. This is required for expanding and path-dependent features whose values would change if history were truncated at a train-split boundary.

Use `construct_temporal_dataset()` when a caller already owns the canonical historical frame and cutoff-aware eligibility metadata.

## Bundle Contract

`TemporalDatasetBundle` owns four aligned outputs:

- `features`: exactly the feature-set output columns in contract order;
- `targets`: exactly the target-set output columns in contract order;
- `samples`: `target_end_date`, `training_eligible_at_cutoff`, and eligibility failure reasons;
- `manifest`: deterministic specification identity and dataset diagnostics.

All three frames use the same unique, sorted `MultiIndex`:

```text
provider, ticker, trading_date
```

The selected supervised target is complete in every retained row. Feature missing values are not imputed or removed: warm-up periods and source-quality gaps must remain visible until split-aware preprocessing is implemented. Tickers that fail current training-eligibility gates remain in the declared universe and are marked in metadata; a completely missing declared ticker is an error.

For fixed-horizon targets, `target_end_date` is derived from observed sessions within each provider/ticker history. Event targets can declare an explicit target output such as `target_end_date_5d`, allowing the bundle to retain the actual event or timeout resolution date.

## Tabular Adapter

```python
from swingtrader.modeling.datasets import to_tabular_dataset

tabular = to_tabular_dataset(bundle)
X = tabular.X
y = tabular.y
samples = tabular.samples
```

The adapter performs no split, purging, imputation, scaling, sampling, or model-specific dtype conversion. Those operations must be fitted or applied inside the later temporal training workflow.

## Current Boundary

Implemented:

- full-prefix feature and target computation;
- deterministic sample alignment and schema validation;
- selected-target filtering while preserving feature NaNs;
- task-specific target resolution dates;
- cutoff-aware training-eligibility metadata;
- manifest diagnostics and a framework-neutral tabular adapter.

Still planned:

- date-based train, validation, and test assignment;
- target-horizon purging and expanding-window folds;
- model-specific preprocessing and training;
- persisted or content-addressed dataset snapshots.
