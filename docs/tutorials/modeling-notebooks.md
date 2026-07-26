# Modeling Notebooks

The repository contains executable onboarding notebooks under `notebooks/workflows/modeling`:

1. `01_data_features_and_targets.ipynb` introduces the canonical market frame, feature-set and target-set contracts, and direct feature and label generation.
2. `02_temporal_dataset_and_splitting.ipynb` builds a `TemporalDatasetBundle` from bronze storage, inspects its aligned artifacts, and applies a leakage-safe fixed temporal split.

Run the notebooks from the repository's Jupyter environment after completing local installation and market-data ingestion. They use public application APIs and intentionally avoid reimplementing production logic inside notebook cells.

A baseline-model notebook should be added when the split-aware preprocessing and training workflow is implemented; adding it earlier would document an API that does not yet exist.
