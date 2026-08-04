# Modeling Notebooks

The repository contains executable onboarding notebooks under `notebooks/workflows/modeling`:

1. `01_data_features_and_targets.ipynb` introduces the canonical market frame, feature-set and target-set contracts, and direct feature and label generation.
2. `02_temporal_dataset_and_splitting.ipynb` builds a `TemporalDatasetBundle` from bronze storage, inspects its aligned artifacts, and applies a leakage-safe fixed temporal split.
3. `03_baseline_models_and_evaluation.ipynb` compares the constant-prior, date-matched random-ranking, and regularized-logistic baselines on validation and inspects the standardized reports and local artifacts.
4. `04_feature_selection_and_temporal_cross_validation.ipynb` compares ordered model input schemas with expanding folds confined to outer train, then demonstrates manually passing a chosen schema to the existing outer evaluation harness.
5. `05_interactive_entry_labeling.ipynb` runs the resumable rolling-window workflow for binary entry labels, Plotly candle selection, hover-driven ATR risk guides, and commission-aware forward-outcome heatmaps.
6. `10_cross_sectional_xgboost_ranking.ipynb` compares XGBoost regression, classification, and learning-to-rank formulations on outer validation.
7. `11_cross_sectional_xgboost_ranker_tuning.ipynb` loads one broad feature set once, slices in-memory feature candidates, and jointly compares those candidates with XGBRanker parameters on expanding folds inside outer train.

Run the notebooks from the repository's Jupyter environment after completing local installation and market-data ingestion. The labeling notebook additionally requires an explicit validation-end boundary before a new session is created. They use public application APIs and intentionally avoid reimplementing production logic inside notebook cells.

The baseline, cross-validation, XGBoost comparison, and ranker-tuning notebooks leave the locked test disabled. Use inner folds and outer validation for preprocessing, feature-schema, model, hyperparameter, threshold, and ranking-rule choices. Enable locked-test evaluation only after those choices are frozen.
