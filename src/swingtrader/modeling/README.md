# Modeling

Model development and inference code lives here. The package currently owns reusable target calculations, versioned target contracts, immutable experiment specifications, and optional local MLflow tracking. Temporal dataset construction, split purging, training, evaluation, model artifacts, and production inference remain follow-up work.

## Implemented Target Code

The `swingtrader.modeling.datasets` package contains:

- `contracts.py`, which defines immutable target-family, target-set, and supervised-task specifications;
- `catalog.py`, which defines the concrete V1 and V2 target sets and their
  primary classification tasks;
- `labels.py`, which contains reusable return-target builders, target-set
  execution, and the `generate_v1_labels()` and `generate_v2_labels()` wrappers;
- `barriers.py`, which implements next-open ATR barrier-event targets, gap
  handling, and deterministic same-bar ambiguity policies.

The V1 target set adds 5-, 10-, and 15-session forward adjusted-close returns plus the nullable Boolean `target_significant_up_5d` column. V2 preserves those outputs and adds ATR-scaled take-profit/stop-loss outcomes over the same horizons.

Target builders consume the same canonical market-price frame as indicators and features: a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, with those identifiers absent from ordinary columns. Builders preserve that index so features and labels align on the same observation identity. Column-oriented bronze rows are converted at the caller boundary with `set_index(...).sort_index()`.

Calculations remain in memory and do not load from or write to the database. Exact reproduction requires both the serialized target manifest and the source revision containing the configured builders.

## Implemented Experiment Code

The `swingtrader.modeling.experiments` package contains:

- `contracts.py`, which defines immutable universe, temporal-split, model, and
  experiment specifications with deterministic manifests and digests;
- `tracking.py`, which lazily imports the optional MLflow dependency and
  initializes local runs from an `ExperimentSpec`;
- dataset-summary contracts and a small run handle for logging metrics and
  generated artifacts without exposing MLflow internals throughout training code.

The complete experiment manifest is available before fitting. MLflow records
runtime provenance such as the Git revision, dataset counts and date ranges,
class prevalence, metrics, reports, and plots. It does not replace the
repository-owned contracts or require a complete materialized dataset snapshot.

See the main documentation for the current modeling plan:

- [Modeling overview](../../../docs/modeling/overview.md)
- [Target and evaluation](../../../docs/modeling/target-and-evaluation.md)
- [ATR barrier targets](../../../docs/modeling/atr-barrier-targets.md)
- [Experiment specifications and MLflow tracking](../../../docs/modeling/experiments.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Roadmap](../../../docs/architecture/roadmap.md)
