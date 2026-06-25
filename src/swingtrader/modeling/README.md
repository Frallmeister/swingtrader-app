# Modeling

Model development and inference code. This package owns training workflows, model definitions, evaluation, artifact loading, and production prediction jobs that read engineered features from the database and write model outputs back to the database.

## Intended Workflow

During development, modeling code will train and validate models from engineered feature data produced by the data package. A future DataLoader should create reproducible train, validation, and test splits from feature tables rather than reading raw API data directly.

After deployment, the current production model should be able to run inference on active tickers that have complete recent features. Newly activated tickers do not necessarily require immediate retraining if the model can generalize from ticker-agnostic market and macro features, but they should only be used for inference after the data layer has backfilled bronze data, built features, and validated readiness.

Training eligibility and inference readiness are related but separate concepts:

- Inference-ready tickers have the recent feature data required for the production model to make predictions.
- Training-eligible tickers have enough historical feature and label data to be included safely in future train, validation, and test splits.

Future training jobs should derive their universe from active tickers that also pass data-quality and history-length requirements. This prevents newly activated or partially backfilled tickers from entering model training before their data is suitable.
