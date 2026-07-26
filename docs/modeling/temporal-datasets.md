# Temporal Datasets

The temporal dataset layer creates the canonical unsplit modeling product between feature/target generation and downstream temporal splitting or model training.

The diagram below shows the complete local workflow. Rectangles are specifications or runtime containers, rounded boxes are operations, cylinders are tabular data, and double-bordered boxes are manifests. `build_temporal_dataset()` is the standard entry point when the source data lives in bronze storage. It loads the required history and eligibility state, then delegates the in-memory construction work to `construct_temporal_dataset()`.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 32, "rankSpacing": 48}}}%%
flowchart TB
    feature_spec["FeatureSetSpec"]
    target_spec["TargetSetSpec"]
    task_spec["SupervisedTaskSpec"]
    universe_spec["UniverseSpec"]
    cutoff["data_cutoff"]
    dataset_spec["TemporalDatasetSpec"]
    bronze[(Bronze market history)]
    eligibility[(Cutoff-aware eligibility)]
    build([build_temporal_dataset])
    prices[(Canonical market frame)]
    construct([construct_temporal_dataset])
    bundle["TemporalDatasetBundle"]
    features[(features)]
    targets[(targets)]
    samples[(samples)]
    manifest[[TemporalDatasetManifest]]
    adapter([to_tabular_dataset])
    tabular["TabularDataset"]

    feature_spec --> dataset_spec
    target_spec --> dataset_spec
    task_spec --> dataset_spec
    universe_spec --> dataset_spec
    cutoff --> dataset_spec
    dataset_spec --> build
    bronze --> build
    build --> prices
    build --> eligibility
    build --> construct
    dataset_spec --> construct
    prices --> construct
    eligibility --> construct
    construct --> bundle
    bundle --> features
    bundle --> targets
    bundle --> samples
    bundle --> manifest
    bundle --> adapter --> tabular

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef action fill:#fff3e0,stroke:#ef6c00
    classDef artifact fill:#e8f5e9,stroke:#2e7d32
    class feature_spec,target_spec,task_spec,universe_spec,cutoff,dataset_spec,bundle,tabular contract
    class build,construct,adapter action
    class bronze,eligibility,prices,features,targets,samples,manifest artifact
```

## Dataset Specification

`TemporalDatasetSpec` contains only choices that determine the unsplit data:

- a versioned `FeatureSetSpec`;
- a versioned `TargetSetSpec`;
- one `SupervisedTaskSpec`;
- a resolved `UniverseSpec` with concrete provider and ticker membership;
- an inclusive `data_cutoff`.

Split dates, model hyperparameters, random seeds, and MLflow are intentionally absent. An `ExperimentSpec` exposes the same lower-level contract through `experiment_spec.dataset_spec`.

## Construction

### Build from bronze storage

`build_temporal_dataset()` is the normal application-facing entry point:


```python
from swingtrader.modeling.datasets import build_temporal_dataset

bundle = build_temporal_dataset(
    engine=engine,
    spec=experiment_spec.dataset_spec,
)
```

The builder loads all required bronze source columns through the cutoff, evaluates cutoff-aware training eligibility, and delegates to `construct_temporal_dataset()`. The lower-level constructor computes every configured feature block over the complete legitimate historical prefix. Targets are computed independently from the same canonical price frame. This is required for expanding and path-dependent features whose values would change if history were truncated at a train-split boundary.

### Construct from an in-memory frame

Use `construct_temporal_dataset()` when a caller already owns the canonical historical frame and cutoff-aware eligibility metadata:

```python
from swingtrader.modeling.datasets import construct_temporal_dataset

bundle = construct_temporal_dataset(
    prices,
    spec=dataset_spec,
    eligibility=eligibility,
)
```

`prices` uses the canonical market index, while `eligibility` maps every declared ticker to its cutoff-aware `TickerEligibility` value. This lower-level boundary is useful in tests and notebooks because it performs the same feature generation, target generation, alignment, filtering, and manifest construction without loading bronze data itself.


## Bundle Contract

`TemporalDatasetBundle` is a frozen dataclass container that owns independently copied frames and one bundle-level manifest. It prevents reassignment of its fields, but the contained pandas DataFrames remain mutable objects and should be treated as owned runtime artifacts rather than deeply immutable values.

The bundle is a container, the cylinders are row-aligned DataFrames, and the double-bordered box is bundle-level metadata rather than another row-aligned frame:

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 28, "rankSpacing": 34}}}%%
flowchart TB
    bundle["TemporalDatasetBundle"]
    index{{"Shared index<br/>(provider, ticker, trading_date)"}}
    features[(features<br/>contract-ordered columns)]
    targets[(targets<br/>contract-ordered columns)]
    samples[(samples<br/>target end and eligibility metadata)]
    manifest[["manifest<br/>identity and diagnostics<br/>(not row-aligned)"]]
    target_rule([Drop rows with a missing selected target])
    feature_rule([Retain missing feature values])

    bundle --> features
    bundle --> targets
    bundle --> samples
    bundle --> manifest
    index -.->|aligns| features
    index -.->|aligns| targets
    index -.->|aligns| samples
    target_rule -->|filters all aligned frames| bundle
    feature_rule --> features

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef artifact fill:#e8f5e9,stroke:#2e7d32
    classDef state fill:#f3e5f5,stroke:#6a1b9a
    classDef action fill:#fff3e0,stroke:#ef6c00
    class bundle contract
    class features,targets,samples,manifest artifact
    class index state
    class target_rule,feature_rule action
```

The selected supervised target is complete in every retained row. Feature warm-up and source-quality gaps remain visible until split-aware preprocessing. Tickers that fail current training-eligibility gates remain in the declared universe and are marked in `samples`; a completely missing declared ticker is an error.

For fixed-horizon targets, `target_end_date` is derived from observed sessions within each provider/ticker history. Event targets can instead declare an explicit target output such as `target_end_date_5d`, preserving the actual event or timeout resolution date.

## Tabular Adapter

```python
from swingtrader.modeling.datasets import to_tabular_dataset

tabular = to_tabular_dataset(bundle)
X = tabular.X
y = tabular.y
samples = tabular.samples
```

The adapter performs no split, purging, imputation, scaling, sampling, or model-specific dtype conversion. Those operations belong to the [baseline training and validation workflow](workflows.md#baseline-training-and-validation), where preprocessing must be fitted on training rows and only applied to validation or test rows.

## Current Boundary

Implemented:

- full-prefix feature and target computation;
- deterministic sample alignment and schema validation;
- selected-target filtering while preserving feature NaNs;
- task-specific target resolution dates;
- cutoff-aware training-eligibility metadata;
- manifest diagnostics and a framework-neutral tabular adapter.

Fixed train/validation/test assignment, actual-target-end purging, and optional embargo are implemented downstream in [Temporal Splitting](temporal-splitting.md).

Still planned:

- expanding-window folds;
- model-specific preprocessing and training;
- persisted or content-addressed dataset snapshots.
