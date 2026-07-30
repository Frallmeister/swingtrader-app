# Modeling Object Model

This page is the canonical reference for relationships between modeling specifications and runtime artifacts. Workflow order is documented separately under [Modeling Workflows](../modeling/workflows.md).

Rectangles represent contracts or runtime containers, cylinders represent row-aligned tabular data, double-bordered boxes represent manifests or reports, and rounded boxes represent operations. A dashed edge denotes a derived or non-owning relationship rather than a stored field.

## Specification Object Model

`ExperimentSpec` stores the complete static experiment configuration. Its `dataset_spec` property constructs a `TemporalDatasetSpec` from the five dataset-defining fields; `TemporalDatasetSpec` is not a separately stored field on the experiment.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 34, "rankSpacing": 48}}}%%
classDiagram
    class FeatureBlockSpec
    class FeatureSetSpec
    class TargetFamilySpec
    class TargetSetSpec
    class SupervisedTaskSpec
    class UniverseSpec
    class TemporalDatasetSpec
    class TemporalSplitSpec
    class TemporalCrossValidationSpec
    class ModelSpec
    class ExperimentSpec

    FeatureSetSpec *-- FeatureBlockSpec : ordered blocks
    TargetSetSpec *-- TargetFamilySpec : ordered families
    TemporalDatasetSpec *-- FeatureSetSpec : feature_set
    TemporalDatasetSpec *-- TargetSetSpec : target_set
    TemporalDatasetSpec *-- SupervisedTaskSpec : task
    TemporalDatasetSpec *-- UniverseSpec : universe
    ExperimentSpec *-- FeatureSetSpec : feature_set
    ExperimentSpec *-- TargetSetSpec : target_set
    ExperimentSpec *-- SupervisedTaskSpec : task
    ExperimentSpec *-- UniverseSpec : universe
    ExperimentSpec *-- TemporalSplitSpec : split
    ExperimentSpec *-- ModelSpec : model
    TemporalCrossValidationSpec ..> TemporalSplitSpec : outer train boundary
    ExperimentSpec ..> TemporalDatasetSpec : dataset_spec property

    classDef feature fill:#e3f2fd,stroke:#1565c0
    classDef target fill:#f3e5f5,stroke:#6a1b9a
    classDef dataset fill:#e8f5e9,stroke:#2e7d32
    classDef experiment fill:#fff3e0,stroke:#ef6c00
    cssClass "FeatureBlockSpec,FeatureSetSpec" feature
    cssClass "TargetFamilySpec,TargetSetSpec,SupervisedTaskSpec" target
    cssClass "UniverseSpec,TemporalDatasetSpec" dataset
    cssClass "TemporalSplitSpec,TemporalCrossValidationSpec,ModelSpec,ExperimentSpec" experiment
```

Feature and target block/set specifications are executable: their recorded parameters drive builders, while declared inputs, outputs, and index preservation are enforced in declaration order. Every specification is versioned or composed from versioned contracts and can be serialized into deterministic manifest data. The experiment digest covers static choices; runtime provenance such as the Git revision belongs to the tracking layer.

## Runtime Artifact Object Model

Dataset construction produces an aligned bundle. Temporal splitting does not copy feature or target data into separate split objects; it annotates sample metadata and returns positional indices into the original bundle.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 30, "rankSpacing": 44}}}%%
flowchart TB
    dataset_spec["TemporalDatasetSpec"]
    build([build_temporal_dataset])
    bundle["TemporalDatasetBundle"]
    features[(features DataFrame)]
    targets[(targets DataFrame)]
    samples[(samples DataFrame)]
    dataset_manifest[[TemporalDatasetManifest]]
    adapter([to_tabular_dataset])
    tabular["TabularDataset"]
    X[(X DataFrame)]
    y[(y Series)]
    tabular_samples[(samples DataFrame)]
    split_spec["TemporalSplitSpec"]
    splitter([FixedTemporalSplitter.assign])
    split_result["TemporalSplitResult"]
    split_samples[(split-annotated samples)]
    split_manifest[[TemporalSplitManifest]]
    summaries["TemporalSplitSummary × 3"]
    positions{{Train, validation, and test positions}}

    dataset_spec --> build --> bundle
    bundle --> features
    bundle --> targets
    bundle --> samples
    bundle --> dataset_manifest
    bundle --> adapter --> tabular
    tabular --> X
    tabular --> y
    tabular --> tabular_samples
    bundle --> splitter
    split_spec --> splitter
    splitter --> split_result
    split_result --> split_samples
    split_result --> split_manifest
    split_manifest --> summaries
    split_result --> positions
    positions -.->|index original aligned frames| features
    positions -.->|index original aligned frames| targets
    positions -.->|index original aligned frames| samples

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef action fill:#fff3e0,stroke:#ef6c00
    classDef artifact fill:#e8f5e9,stroke:#2e7d32
    classDef state fill:#f3e5f5,stroke:#6a1b9a
    class dataset_spec,bundle,tabular,split_spec,split_result,summaries contract
    class build,adapter,splitter action
    class features,targets,samples,dataset_manifest,X,y,tabular_samples,split_samples,split_manifest artifact
    class positions state
```

`TemporalDatasetBundle` and `TemporalSplitResult` are frozen dataclass containers that take deep copies of their pandas frames during construction. Freezing prevents field reassignment but does not make the contained DataFrames deeply immutable.

## Train-Only Cross-Validation Artifacts

`build_expanding_temporal_folds()` returns compact `TemporalFold` objects containing positional indices and inclusive candidate partition boundaries used for target-end containment. Because purging can remove rows at a partition end, the stored end dates need not be retained signal dates. The indices always reference the original aligned bundle and are subsets of outer train.

```mermaid
flowchart LR
    bundle["TemporalDatasetBundle"]
    split_result["TemporalSplitResult"]
    cv_spec["TemporalCrossValidationSpec"]
    build([build_expanding_temporal_folds])
    folds["TemporalFold tuple"]
    train_indices{{Train indices}}
    validation_indices{{Validation indices}}

    bundle --> build
    split_result -->|outer train positions only| build
    cv_spec --> build
    build --> folds
    folds --> train_indices
    folds --> validation_indices

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef action fill:#fff3e0,stroke:#ef6c00
    classDef state fill:#f3e5f5,stroke:#6a1b9a
    class bundle,split_result,cv_spec,folds contract
    class build action
    class train_indices,validation_indices state
```

## Baseline Runtime Artifacts

`run_baseline_experiment()` combines the experiment specification, canonical bundle, and split result without copying split-specific datasets into new long-lived containers. It fits one baseline artifact on train positions and creates an independent standardized prediction frame and `EvaluationReport` for each requested evaluation split. Validation is always requested; test is optional.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 32, "rankSpacing": 46}}}%%
flowchart TB
    experiment["ExperimentSpec"]
    bundle["TemporalDatasetBundle"]
    split_result["TemporalSplitResult"]
    tabular["TabularDataset"]
    harness([run_baseline_experiment])
    model["BaselineModelArtifact"]
    result["BaselineExperimentResult"]
    validation_predictions[(Validation prediction frame)]
    test_predictions[(Test prediction frame)]
    validation_report[[Validation EvaluationReport]]
    test_report[[Test EvaluationReport]]
    artifacts[(JSON, CSV, Markdown, and SVG artifacts)]
    mlflow([ExperimentRun logging])

    bundle --> tabular
    experiment --> harness
    bundle --> harness
    split_result --> harness
    tabular -.->|train, validation, and optional test positions| harness
    harness --> model
    harness --> validation_predictions --> validation_report
    harness -->|explicit opt-in| test_predictions --> test_report
    model --> result
    validation_report --> result
    test_report --> result
    result --> artifacts
    result --> mlflow

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef action fill:#fff3e0,stroke:#ef6c00
    classDef artifact fill:#e8f5e9,stroke:#2e7d32
    class experiment,bundle,split_result,tabular,model,result contract
    class harness,mlflow action
    class validation_predictions,test_predictions,validation_report,test_report,artifacts artifact
```

The prediction frame has the stable ordered columns `split`, `target`, `score`, `predicted_class`, and `ranking_return`. `EvaluationReport` retains the exact `EvaluationConfig`, the original ranking-return source column when supplied, pooled metrics, dataset context, source predictions, per-date metrics, calibration buckets, daily and aggregate score-quantile tables, top-`k` and random selections, and feature missingness.

`BaselineModelArtifact` is a structural interface rather than a persistence framework. Each concrete baseline exposes `predict_scores()` and a JSON-compatible `to_manifest()`. The logistic artifact retains its train-fitted scikit-learn median-imputation values, post-imputation means, and scales so validation and test transformations do not depend on either evaluation split.
