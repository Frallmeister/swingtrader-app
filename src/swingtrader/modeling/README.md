# Modeling

Model development and inference code lives here. The package owns reusable target calculations, versioned target contracts, canonical unsplit temporal dataset construction, immutable experiment specifications, purged fixed temporal splitting, and optional local MLflow tracking. Training, evaluation, model artifacts, and production inference remain follow-up work.

## Implemented Dataset Code

The `swingtrader.modeling.datasets` package contains:

- `contracts.py`, which defines immutable target-family, target-set, and supervised-task specifications;
- `catalog.py`, which defines the concrete V1 and V2 target sets and their primary classification tasks;
- `labels.py` and `barriers.py`, which implement forward-return and next-open ATR barrier targets;
- `specifications.py`, which binds a feature set, target set, selected task, resolved universe, and data cutoff;
- `temporal.py`, which builds aligned feature, target, and sample-metadata frames over the full historical prefix;
- `tabular.py`, which exposes framework-neutral `X`, `y`, and sample metadata without splitting or preprocessing.

Features, targets, and sample metadata use the same unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`. The builder retains feature warm-up missing values, removes only rows where the selected supervised target is unavailable, records each sample's `target_end_date`, and evaluates ticker training eligibility using data at or before the dataset cutoff.

`build_temporal_dataset()` loads the required bronze columns. `construct_temporal_dataset()` is the source-independent constructor used by tests and callers that already own a canonical historical frame. Both return an unsplit `TemporalDatasetBundle`; the experiment splitter then assigns shared calendar ranges, purges boundary-crossing targets, and optionally applies a pre-boundary embargo. Imputation, scaling, and model-specific conversion remain downstream responsibilities.

## Implemented Experiment Code

The `swingtrader.modeling.experiments` package defines temporal-split, model, and experiment specifications with deterministic manifests and digests. `FixedTemporalSplitter` applies one purged train/validation/locked-test holdout and returns split-annotated sample metadata plus diagnostics. `UniverseSpec` is owned by the lower-level dataset specification package and re-exported from `swingtrader.modeling.experiments` for compatibility. `ExperimentSpec.dataset_spec` exposes exactly the subset required to construct the canonical unsplit dataset.

The optional MLflow adapter records runtime provenance such as the Git revision, dataset summaries, metrics, reports, and plots. It does not own dataset semantics or require a complete materialized dataset snapshot.

See the main documentation:

- [Modeling overview](../../../docs/modeling/overview.md)
- [Temporal datasets](../../../docs/modeling/temporal-datasets.md)
- [Temporal splitting](../../../docs/modeling/temporal-splitting.md)
- [Target and evaluation](../../../docs/modeling/target-and-evaluation.md)
- [ATR barrier targets](../../../docs/modeling/atr-barrier-targets.md)
- [Experiment specifications and MLflow tracking](../../../docs/modeling/experiments.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Roadmap](../../../docs/architecture/roadmap.md)
