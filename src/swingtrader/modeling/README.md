# Modeling

Model development and inference code lives here. The package is planned to own dataset construction, training workflows, evaluation, model artifacts, and production inference over engineered feature data.

## Implemented Dataset Code

The `swingtrader.modeling.datasets.labels` module implements the V1 target-label contract for in-memory pandas workflows. Use `generate_v1_labels()` with a DataFrame compatible with `load_bronze_daily_prices()` and containing `provider`, `ticker`, `trading_date`, and `adjusted_close`.

The function adds 5-, 10-, and 15-session forward returns plus the nullable Boolean `target_significant_up_5d` column. It does not load from or write to the database.

See the main documentation for the current modeling plan:

- [Modeling overview](../../../docs/modeling/overview.md)
- [Target and evaluation](../../../docs/modeling/target-and-evaluation.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Roadmap](../../../docs/architecture/roadmap.md)
