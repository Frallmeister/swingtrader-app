# Modeling Overview

Modeling code is planned, not implemented yet. The V1 target and evaluation contract is documented in [Target and Evaluation](target-and-evaluation.md).

The modeling package will own dataset construction, training workflows, evaluation, model artifact management, and production inference.

## Inference Readiness

The current implementation evaluates inference readiness from bronze daily-price state only.

An active ticker is not automatically inference-ready. It must first have enough recent and clean bronze rows. The implemented bronze-backed rules are documented in [Ticker Eligibility](../data/eligibility.md).

Once feature generation and production inference exist, inference readiness will also require model-ready feature availability, feature recency, and the input window required by the production model.

## Training Eligibility

A training-eligible ticker currently has enough historical and clean bronze daily-price data to be considered for future model training.

Training eligibility and inference readiness are related but separate concepts.

Training code should consume eligibility checks instead of blindly training on the active trading universe. The future training universe may be broader than active tickers.

Once label and feature pipelines exist, training eligibility will also require enough feature rows, labels, and usable observations for the intended train, validation, and test split design.

## Retraining Cadence

The expected retraining cadence is local/manual a few times per year, not continuous production retraining.

## Planned Components

- feature and label readers;
- V1 label-generation workflow;
- train, validation, and test split builder;
- baseline ranking model;
- evaluation reports;
- local model registry;
- production inference workflow.