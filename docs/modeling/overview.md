# Modeling Overview

Modeling is planned, not implemented yet.

The modeling package will own dataset construction, training workflows, evaluation, model artifact management, and production inference.

## Inference Readiness

An inference-ready ticker has the recent feature data required for the production model to make a prediction.

An active ticker is not automatically inference-ready. It must first have enough bronze data, generated features, and recent rows. The current bronze-backed rules are documented in [Ticker Eligibility](../data/eligibility.md).

## Training Eligibility

A training-eligible ticker has enough historical feature and label data to be included safely in train, validation, and test splits.

Training eligibility and inference readiness are related but separate concepts.

Training code should consume eligibility checks instead of blindly training on the active trading universe. The future training universe may be broader than active tickers.

## Retraining Cadence

The expected retraining cadence is local/manual a few times per year, not continuous production retraining.

## Planned Components

- feature and label readers;
- train, validation, and test split builder;
- baseline ranking model;
- evaluation reports;
- local model registry;
- production inference workflow.