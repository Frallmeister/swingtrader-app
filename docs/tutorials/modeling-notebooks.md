# Modeling Notebooks

The repository contains executable onboarding notebooks under `notebooks/workflows/modeling`:

1. `01_data_features_and_targets.ipynb` introduces the canonical market frame, feature-set and target-set contracts, and direct feature and label generation.
2. `02_temporal_dataset_and_splitting.ipynb` builds a `TemporalDatasetBundle` from bronze storage, inspects its aligned artifacts, and applies a leakage-safe fixed temporal split.
3. `03_baseline_models_and_evaluation.ipynb` compares the constant-prior, date-matched random-ranking, and regularized-logistic baselines on validation and inspects the standardized reports and local artifacts.
4. `04_feature_selection_and_temporal_cross_validation.ipynb` compares ordered model input schemas with expanding folds confined to outer train, then demonstrates manually passing a chosen schema to the existing outer evaluation harness.

Run the notebooks from the repository's Jupyter environment after completing local installation and market-data ingestion. They use public application APIs and intentionally avoid reimplementing production logic inside notebook cells.

The baseline and cross-validation notebooks leave the locked test disabled. Use inner folds and outer validation for preprocessing, feature-schema, model, hyperparameter, threshold, and ranking-rule choices. Enable locked-test evaluation only after those choices are frozen.
