# Modeling Overview

Modeling code now includes reusable V1 and V2 label generation, explicit versioned target-set contracts, immutable experiment specifications, and optional local MLflow tracking. The V1 research-return contract is documented in [Target and Evaluation](target-and-evaluation.md), the V2 execution-oriented labels are documented in [ATR Barrier Targets](atr-barrier-targets.md), and experiment provenance is documented in [Experiment Specifications and MLflow Tracking](experiments.md).

The modeling package will own dataset construction, training workflows, evaluation, model artifact management, and production inference.

## Implemented Components

The modeling datasets package defines immutable target-family, target-set, and supervised-task contracts plus concrete V1 and V2 target catalogs. Target builders consume the same canonical market-price DataFrame as indicators and features: a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, with identifiers absent from ordinary columns. Column-oriented bronze rows are converted once at the caller boundary. Target generation preserves the canonical index, allowing features and labels to align directly on observation identity.

V1 calculates 5-, 10-, and 15-session forward returns from adjusted close and adds the nullable Boolean `target_significant_up_5d` target. V2 preserves those outputs and adds next-open ATR-scaled barrier events, nullable take-profit-before-stop-loss targets, event timing, resolution dates, and measurable same-bar ambiguity for 5-, 10-, and 15-session horizons.

The experiment package composes feature and target identities with a resolved universe, data cutoff, temporal split, selected task, model configuration, and random seeds. Its deterministic manifests can be serialized before fitting. A thin optional MLflow adapter records executions, Git revision, dataset summaries, metrics, reports, and the canonical experiment manifest through a local SQLite backend and artifact directory.

Target generation and feature generation remain in memory for now. Feature and label persistence, database schemas, temporal dataset construction, split purging, model training, and standardized evaluation reports are planned follow-up work.

## Inference Readiness

The current implementation evaluates inference readiness from bronze daily-price state only.

An active ticker is not automatically inference-ready. It must first have enough recent and clean bronze rows. The implemented bronze-backed rules are documented in [Ticker Eligibility](../data/eligibility.md).

Once production inference exists, inference readiness will also require model-ready feature availability, feature recency, and the input window required by the production model.

## Training Eligibility

A training-eligible ticker currently has enough historical and clean bronze daily-price data to be considered for future model training.

Training eligibility and inference readiness are related but separate concepts.

Training code should consume eligibility checks instead of blindly training on the active trading universe. The future training universe may be broader than active tickers.

Once temporal modeling datasets exist, training eligibility will also require enough feature rows, labels, and usable observations for the intended train, validation, and test split design.

## Retraining Cadence

The expected retraining cadence is local/manual a few times per year, not continuous production retraining.

## Planned Components

- canonical temporal dataset builder;
- purged train, validation, and test splits;
- baseline ranking model;
- evaluation reports;
- local model registry;
- production inference workflow.