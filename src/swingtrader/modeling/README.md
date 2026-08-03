# Modeling

Model development and inference code lives here. The package owns reusable target calculations, versioned target contracts, canonical unsplit temporal dataset construction, immutable experiment specifications, purged fixed temporal splitting, baseline fitting, standardized evaluation, interactive entry-labeling support, a small daily-bar backtesting pilot, generated model artifacts, and optional local MLflow tracking. A notebook-led cross-sectional XGBoost comparison is available for exploration. Reusable nonlinear training contracts, model registration, and production inference remain follow-up work.

## Dataset Package

The `swingtrader.modeling.datasets` package contains:

- `contracts.py`, which defines immutable, executable target-family and target-set specifications plus supervised-task selection;
- `catalog.py`, which defines the concrete forward-return, triple-barrier, and cross-sectional target sets and their primary tasks;
- `labels.py`, `cross_sectional.py`, and `triple_barrier.py`, which implement forward-return, cross-sectional return, and next-open triple-barrier targets;
- `specifications.py`, which binds a feature set, target set, selected task, resolved universe, and data cutoff;
- `temporal.py`, which builds aligned feature, target, and sample-metadata frames over the full historical prefix;
- `tabular.py`, which exposes framework-neutral `X`, `y`, and sample metadata without splitting or preprocessing.

Features, targets, and sample metadata use the same unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`. Feature and target specifications pass their recorded parameters to builders, enforce required inputs and index preservation, and select exactly their declared outputs before temporal alignment. The dataset builder retains feature warm-up missing values, removes only rows where the selected supervised target is unavailable, records each sample's `target_end_date`, and evaluates ticker training eligibility using data at or before the dataset cutoff.

`build_temporal_dataset()` loads the required bronze columns. `construct_temporal_dataset()` is the source-independent constructor used by tests and callers that already own a canonical historical frame. Both return an unsplit `TemporalDatasetBundle`; the experiment splitter then assigns shared calendar ranges, purges boundary-crossing targets, and optionally applies a pre-boundary embargo.

## Experiment Package

The `swingtrader.modeling.experiments` package defines temporal-split, model, and experiment specifications with deterministic manifests and digests. `FixedTemporalSplitter` applies one purged train/validation/locked-test holdout and returns split-annotated sample metadata plus diagnostics. `UniverseSpec` is owned by the lower-level dataset specification package and re-exported from `swingtrader.modeling.experiments` for compatibility. `ExperimentSpec.dataset_spec` exposes exactly the subset required to construct the canonical unsplit dataset.

The optional MLflow adapter records runtime provenance such as the Git revision, dataset summaries, metrics, reports, and plots. It does not own dataset semantics or require a complete materialized dataset snapshot.

## Training Package

The `swingtrader.modeling.training` package contains:

- `contracts.py`, which defines `EvaluationConfig` and the ordered prediction-frame schema;
- `baselines.py`, which implements constant-prior, deterministic random-ranking, and regularized logistic baselines;
- `evaluation.py`, which computes pooled classification, calibration, per-date, cross-sectional ranking, random-comparison, and missingness results;
- `reporting.py`, which writes deterministic JSON, CSV, Markdown, compressed prediction, and SVG artifacts;
- `harness.py`, which fits on train, evaluates validation by default, optionally evaluates locked test, and logs results through `ExperimentRun`;
- `ranking.py`, which prepares date-grouped XGBoost ranking inputs and calculates compact cross-sectional ranking diagnostics for notebooks.

Logistic preprocessing is fitted on training rows only and retained with the fitted coefficients. Validation and test reports are independent; locked-test rows are not read during routine validation runs.

## Interactive Entry Labeling

`labeling.py` keeps the manually supervised workflow deliberately compact. It owns deterministic rolling windows, one authoritative binary label per provider/ticker/date, resumable session state, indicator-enriched Plotly figures, commission-aware forward-outcome heatmaps, and transactional SQLite/PostgreSQL upserts. The notebook under `notebooks/workflows/modeling` owns click/hover callbacks and mutable working selections.

## Backtesting Pilot

`backtest.py` provides one deliberately small, pandas-based executable simulation. It consumes raw OHLC prices plus ranked `score` and raw-price `atr` signals, enters at the next open, applies ATR risk sizing and fixed stop/target rules, and returns transaction, equity, and summary tables. It is a proof of concept rather than a general backtesting framework.

See the main documentation:

- [Modeling overview](../../../docs/modeling/overview.md)
- [Modeling workflows](../../../docs/modeling/workflows.md)
- [Temporal datasets](../../../docs/modeling/temporal-datasets.md)
- [Temporal splitting](../../../docs/modeling/temporal-splitting.md)
- [Target overview](../../../docs/modeling/targets/index.md)
- [Triple-barrier targets](../../../docs/modeling/targets/triple-barrier.md)
- [Cross-sectional return targets](../../../docs/modeling/targets/cross-sectional.md)
- [Baseline models and evaluation harness](../../../docs/modeling/baseline-models.md)
- [Model evaluation](../../../docs/modeling/evaluation.md)
- [Cross-sectional XGBoost ranking study](../../../docs/modeling/cross-sectional-ranking-study.md)
- [Interactive entry labeling](../../../docs/modeling/data-labeling.md)
- [Backtesting pilot](../../../docs/modeling/backtesting.md)
- [Experiment specifications and MLflow tracking](../../../docs/modeling/experiments.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Roadmap](../../../docs/architecture/roadmap.md)
