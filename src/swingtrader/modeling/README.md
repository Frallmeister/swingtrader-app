# Modeling

Model development and inference code lives here. The package owns the beginning of the modeling workflow: reusable target calculations and explicit, versioned target contracts. Dataset splitting, training, evaluation, model artifacts, and production inference remain follow-up work.

## Implemented Target Code

The `swingtrader.modeling.datasets` package contains:

- `contracts.py`, which defines immutable target-family, target-set, and supervised-task specifications;
- `target_catalog.py`, which defines the concrete V1 target set and primary classification task;
- `labels.py`, which contains reusable target builders and the compatibility wrapper `generate_v1_labels()`.

The V1 target set adds 5-, 10-, and 15-session forward adjusted-close returns plus the nullable Boolean `target_significant_up_5d` column. Calculations remain in memory and do not load from or write to the database.

Exact reproduction requires both the serialized target manifest and the source revision containing the configured builders.

See the main documentation for the current modeling plan:

- [Modeling overview](../../../docs/modeling/overview.md)
- [Target and evaluation](../../../docs/modeling/target-and-evaluation.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Roadmap](../../../docs/architecture/roadmap.md)
