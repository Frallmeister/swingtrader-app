# Modeling Overview

The modeling layer turns point-in-time-safe market history into reproducible supervised-learning datasets, assigns leakage-safe temporal splits, and trains and evaluates reference models through a standardized baseline harness. The individual contracts and operations are described on focused pages; this page provides the end-to-end red line.

The implemented target contracts are grouped under [Targets](targets/index.md), dataset construction is documented in [Temporal Datasets](temporal-datasets.md), split semantics in [Temporal Splitting](temporal-splitting.md), baseline fitting in [Baseline Models and Evaluation Harness](baseline-models.md), and experiment provenance in [Experiment Specifications and MLflow Tracking](experiments.md). [Modeling Workflows](workflows.md) describes how these components are composed, while the [Modeling Object Model](../reference/modeling-object-model.md) is the canonical reference for relationships between specifications and runtime artifacts.

Rectangles are specifications or runtime containers, rounded boxes are operations, cylinders are tabular data, double-bordered boxes are manifests or reports, and hexagons are split assignments. Dashed edges lead to planned work.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 32, "rankSpacing": 48}}}%%
flowchart TB
    experiment["ExperimentSpec"]
    dataset_spec["Derived TemporalDatasetSpec"]
    source[(Bronze history)]
    eligibility[(Cutoff-aware eligibility)]
    builder([Build temporal dataset])
    bundle["TemporalDatasetBundle"]
    features[(features)]
    targets[(targets)]
    samples[(samples)]
    manifest[[TemporalDatasetManifest]]
    splitter([Apply fixed temporal split])
    train{{Train positions}}
    validation{{Validation positions}}
    test{{Locked-test positions}}
    harness([Run baseline experiment])
    model[[Fitted model artifact]]
    validation_report[[Validation report]]
    test_report[[Locked-test report]]
    advanced([Train nonlinear candidates<br/>planned])

    experiment -->|dataset choices| dataset_spec
    dataset_spec --> builder
    source --> builder
    builder --> eligibility
    builder --> bundle
    bundle --> features
    bundle --> targets
    bundle --> samples
    eligibility --> samples
    bundle --> manifest
    experiment -->|split policy| splitter
    bundle --> splitter
    splitter --> train
    splitter --> validation
    splitter --> test
    experiment -->|model, parameters, seeds| harness
    bundle --> harness
    train --> harness
    validation --> harness
    test -->|explicit final opt-in| harness
    harness --> model
    harness --> validation_report
    harness --> test_report
    bundle -.-> advanced
    splitter -.-> advanced
    experiment -.-> advanced

    classDef contract fill:#e3f2fd,stroke:#1565c0
    classDef action fill:#fff3e0,stroke:#ef6c00
    classDef artifact fill:#e8f5e9,stroke:#2e7d32
    classDef state fill:#f3e5f5,stroke:#6a1b9a
    classDef planned fill:#fafafa,stroke:#9e9e9e,color:#616161,stroke-dasharray:6 4
    class experiment,dataset_spec,bundle contract
    class builder,splitter,harness action
    class source,eligibility,features,targets,samples,manifest,model,validation_report,test_report artifact
    class train,validation,test state
    class advanced planned
```

## Implemented Components

Feature and target builders consume the same canonical market-price DataFrame: a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, with identifiers absent from ordinary columns. V1 calculates forward returns and a fixed-return classification target. V2 preserves those outputs and adds next-open ATR-scaled barrier events, target resolution dates, and measurable same-bar ambiguity.

`TemporalDatasetSpec` defines only the unsplit data product. Features and targets are calculated independently over the same full historical prefix, then aligned with sample metadata and a deterministic manifest. Feature NaNs are retained; rows are excluded only when the selected target is unavailable.

The experiment package applies split policy downstream, using each row's actual target resolution date for purging and an optional global pre-boundary embargo. `ExperimentSpec.dataset_spec` keeps dataset construction independent of model, seed, and MLflow concerns, while the optional MLflow adapter records executions and runtime provenance.

The training package implements a constant-prior classifier, deterministic date-matched random ranker, regularized logistic regression with train-only median imputation and scaling, a canonical prediction frame, and reusable classification, calibration, cross-sectional ranking, artifact, and MLflow logging workflows. Validation is the routine evaluation split; locked-test evaluation requires explicit opt-in.

Feature and target persistence, materialized dataset snapshots, expanding-window folds, nonlinear model candidates, model registration, and production inference remain planned.

## Inference Readiness

Inference readiness currently evaluates bronze daily-price state only. Production inference will later add model-ready feature availability, recency, and input-window requirements. The current bronze conditions are defined under [Ticker Eligibility — Version 1 Bronze Rules](../data/eligibility.md#version-1-bronze-rules).

## Training Eligibility

Training eligibility remains distinct from active-universe membership and inference readiness. The temporal dataset records cutoff-aware eligibility as sample metadata rather than silently filtering declared universe members. The splitter preserves this metadata without filtering it; training code can add minimum usable-sample rules appropriate to the selected task and split design. The current bronze conditions are defined under [Ticker Eligibility — Version 1 Bronze Rules](../data/eligibility.md#version-1-bronze-rules).

## Planned Components

- expanding-window walk-forward folds;
- XGBoost classification and regression candidates;
- feature ablation and selection;
- local model registry;
- production inference workflow.
