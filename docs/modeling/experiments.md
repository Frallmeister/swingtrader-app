# Experiment Specifications and MLflow Tracking

Model experiments have two separate sources of provenance:

1. repository-owned specifications define what an experiment means;
2. MLflow records one execution of that specification and its results.

MLflow is therefore an execution ledger, not the source of experiment semantics. A run remains understandable from the serialized `ExperimentSpec` even before a dataset is built or a model is fitted.

## Install the Modeling Extra

MLflow is optional because data ingestion, feature generation, and target generation do not require it. The modeling extra installs `mlflow-skinny` together with the SQL storage dependencies needed for direct local SQLite tracking. It deliberately avoids the full MLflow distribution, whose additional model-flavor dependencies are unnecessary for this adapter.

```powershell
uv sync --extra modeling --dev
```

The tracking helpers import MLflow only when `start_experiment_run()` is called. Importing experiment contracts does not require the optional dependency.

## Experiment Contracts

The experiment layer composes four immutable specifications:

| Contract | Purpose |
| --- | --- |
| `UniverseSpec` | Records provider and concrete ticker membership; it is owned by the lower-level dataset package and re-exported here. |
| `TemporalSplitSpec` | Declares shared, inclusive train, validation, and test calendar ranges plus an optional pre-boundary embargo. |
| `ModelSpec` | Records the model implementation identity and JSON-compatible hyperparameters. |
| `ExperimentSpec` | Composes the feature set, target set, selected task, universe, data cutoff, split, model, and random seeds. |

Each specification has a deterministic manifest and SHA-256 digest. Ticker ordering does not affect a universe digest because membership order is not meaningful. Changes such as adding a ticker, changing a split date or embargo, selecting another target, changing a hyperparameter, or changing a seed do affect the experiment digest.

The Git revision is intentionally not part of the static `ExperimentSpec`. It describes the code used for a particular execution and is logged by the MLflow adapter when Git metadata is available.

## Define an Experiment Before Fitting

The following example uses the current candidate feature set and V2 ATR barrier task. The ticker tuple is illustrative; production experiment definitions should use the actual resolved training universe.

```python
from datetime import date

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.modeling.datasets.catalog import V2_PRIMARY_TASK, V2_TARGET_SET
from swingtrader.modeling.experiments import (
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
    UniverseSpec,
)

experiment_spec = ExperimentSpec(
    name="logistic_atr_barrier_baseline",
    version="1",
    feature_set=DEFAULT_FEATURE_SET,
    target_set=V2_TARGET_SET,
    task=V2_PRIMARY_TASK,
    universe=UniverseSpec(
        name="se_large_mid_cap_training",
        version="2026-07-24",
        provider="yfinance",
        tickers=("ABB.ST", "SAAB-B.ST", "VOLV-B.ST"),
    ),
    data_cutoff=date(2025, 12, 31),
    split=TemporalSplitSpec(
        name="initial_holdout",
        version="1",
        train_start=date(2010, 1, 1),
        train_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2025, 12, 31),
        embargo_sessions=0,
    ),
    model=ModelSpec(
        name="logistic_regression",
        version="1",
        model_type="sklearn.linear_model.LogisticRegression",
        hyperparameters={
            "C": 1.0,
            "class_weight": None,
            "max_iter": 1_000,
        },
    ),
    random_seeds={"model": 17, "sampling": 17},
)

print(experiment_spec.identifier)
print(experiment_spec.digest)
print(experiment_spec.to_json())
```

Build the canonical unsplit dataset from the lower-level part of the experiment specification, then apply the experiment's fixed split policy:

```python
from swingtrader.modeling.datasets import build_temporal_dataset
from swingtrader.modeling.experiments import FixedTemporalSplitter

bundle = build_temporal_dataset(
    engine=engine,
    spec=experiment_spec.dataset_spec,
)
splitter = FixedTemporalSplitter(experiment_spec.split)
split_result = splitter.assign(bundle)
train_index = split_result.indices("train")
validation_index = split_result.indices("validation")
locked_test_index = split_result.indices("test")
```

The dataset builder computes each row's actual `target_end_date` and aligned sample metadata. The splitter applies the same inclusive calendar ranges to every ticker, then purges a candidate row when its target ends after that split's end. Optional embargo removes additional global observed signal dates from the end of train and validation. The locked test indices are explicit and are not yielded by `splitter.split(bundle)`. See [Temporal Splitting](temporal-splitting.md) for the complete boundary semantics and diagnostics.

## Start a Local MLflow Run

By default, run metadata is stored in the local SQLite database `./mlflow.db`, while MLflow stores generated artifacts under `./mlruns`. Set `MLFLOW_TRACKING_URI` or pass `tracking_uri=` to use another local database or a future tracking service.

```python
from datetime import date
from pathlib import Path

from swingtrader.modeling.experiments import (
    DatasetSplitSummary,
    DatasetSummary,
    start_experiment_run,
)

summary = DatasetSummary(
    train=DatasetSplitSummary(
        rows=120_000,
        ticker_count=3,
        start_date=date(2010, 1, 4),
        end_date=date(2021, 12, 30),
        class_prevalence=0.18,
    ),
    validation=DatasetSplitSummary(
        rows=24_000,
        ticker_count=3,
        start_date=date(2022, 1, 3),
        end_date=date(2023, 12, 29),
        class_prevalence=0.17,
    ),
    test=DatasetSplitSummary(
        rows=25_000,
        ticker_count=3,
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 30),
        class_prevalence=0.16,
    ),
)

with start_experiment_run(
    experiment_spec,
    experiment_name="swingtrader-baselines",
    dataset_summary=summary,
) as run:
    # Fit and evaluate the model here.
    run.log_metrics(
        {
            "validation.roc_auc": 0.71,
            "validation.average_precision": 0.29,
        }
    )
    run.log_artifact(Path("reports/validation.html"), artifact_path="reports")
```

The initialized run records:

- experiment, feature-set, target-set, universe, split, and model identities and digests;
- selected task and target column;
- data cutoff, hyperparameters, random seeds, and Git commit when available;
- split row counts, ticker counts, date ranges, and optional class prevalence;
- the complete canonical experiment manifest at `manifests/experiment.json`;
- metrics and generated artifacts logged by the training workflow.

Dataset summaries deliberately contain no feature or target rows. Before a run starts, their ticker counts and observed date ranges are checked against the experiment's declared universe and temporal split. The current Yahoo Finance bronze data remains the historical source of truth, while stronger source-data versioning or materialized dataset snapshots can be added later if the data source becomes mutable or multi-provider.

The lightweight project dependency intentionally does not install MLflow's server or UI. To inspect recorded runs, launch a transient full MLflow CLI against the same database:

```powershell
uvx --from mlflow==3.14.0 mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` to inspect experiments and compare runs. The server is optional for logging; project code writes directly to SQLite through the lightweight client.

## Compare Runs

Compare runs with both configuration and outcome in view:

1. confirm the experiment, feature-set, target-set, universe, and split digests;
2. compare model hyperparameters and random seeds;
3. inspect row counts, date ranges, ticker counts, and prevalence for unexpected dataset drift;
4. compare discrimination, calibration, cross-sectional ranking, and strategy-oriented metrics;
5. inspect generated reports and plots before promoting conclusions.

A metric difference is not attributable to the model when the underlying experiment digests or dataset summaries also changed.

## Current Boundary

Implemented here:

- immutable experiment specifications;
- canonical unsplit temporal dataset specifications and construction;
- purged fixed train/validation/locked-test assignment with optional embargo and diagnostics;
- deterministic manifests and identities;
- local MLflow run initialization;
- Git-revision, parameter, summary, metric, and artifact logging.

Still planned:

- expanding-window folds;
- baseline and XGBoost training workflows;
- standardized evaluation reports;
- remote tracking, registry promotion, and production serving.
