# Modeling Overview

Modeling code has started with reusable V1 label generation. The V1 target and evaluation contract is documented in [Target and Evaluation](target-and-evaluation.md).

The modeling package will own dataset construction, training workflows, evaluation, model artifact management, and production inference.

## Implemented Components

The modeling datasets package implements V1 target labels from daily price DataFrames compatible with the bronze daily-price loader. The label generator preserves source observations, calculates 5-, 10-, and 15-session forward returns from adjusted close, and adds the nullable Boolean `target_significant_up_5d` target.

Label generation and initial return features remain in memory for now. Feature and label persistence, database schemas, temporal splitting, model training, and evaluation reports are planned follow-up work.

## Inference Readiness

The current implementation evaluates inference readiness from bronze daily-price state only.

An active ticker is not automatically inference-ready. It must first have enough recent and clean bronze rows. The implemented bronze-backed rules are documented in [Ticker Eligibility](../data/eligibility.md).

Once production inference exists, inference readiness will also require model-ready feature availability, feature recency, and the input window required by the production model.

## Training Eligibility

A training-eligible ticker currently has enough historical and clean bronze daily-price data to be considered for future model training.

Training eligibility and inference readiness are related but separate concepts.

Training code should consume eligibility checks instead of blindly training on the active trading universe. The future training universe may be broader than active tickers.

Once broader feature pipelines and temporal datasets exist, training eligibility will also require enough feature rows, labels, and usable observations for the intended train, validation, and test split design.

## Retraining Cadence

The expected retraining cadence is local/manual a few times per year, not continuous production retraining.

## Planned Components

- broader reusable OHLCV-derived features;
- feature readers;
- train, validation, and test split builder;
- baseline ranking model;
- evaluation reports;
- local model registry;
- production inference workflow.