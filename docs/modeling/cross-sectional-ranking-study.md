# Cross-Sectional XGBoost Ranking Study

The reusable notebook `notebooks/workflows/modeling/10_cross_sectional_xgboost_ranking.ipynb` compares three exploratory XGBoost formulations on the same cross-sectional dataset:

- regression on five-session market-relative return;
- classification of future top-quintile membership;
- learning to rank with ordinal relevance grades and one provider/date query group.

This is an exploratory workflow rather than an extension of the reusable baseline harness. It fits on the purged outer train split, evaluates validation only, and leaves the locked test untouched.

## Comparison Metrics

Every model output is treated as a ranking score. The notebook reports:

- NDCG at the configured top `k`, using ordinal relevance grades;
- daily rank IC, defined as Spearman correlation with continuous market-relative return;
- the fraction of dates with positive rank IC;
- mean return among the top `k` candidates.

The date-level results remain available for distribution plots and regime inspection. Aggregate values alone are not sufficient evidence that one formulation is stable.

## Reusable Helpers

`swingtrader.modeling.training.ranking` contains only the two small operations shared by the notebook:

- `prepare_xgboost_ranking_data()` sorts rows into contiguous provider/date groups and returns XGBoost query IDs;
- `evaluate_cross_sectional_scores()` calculates the common date-level and aggregate ranking diagnostics.

Model fitting and parameter choices remain visible in the notebook so they can be changed quickly during exploration.

## Working With the Notebook

The committed notebook is a generic base. When dates, features, targets, or parameters are changed for a material study:

1. run the notebook;
2. export it to HTML under `notebooks/exports/`;
3. retain the HTML locally as the record of that run;
4. restore the notebook to its generic base state before committing unrelated experiments.

A more formal training implementation is justified only after the notebook identifies a formulation worth retaining.
