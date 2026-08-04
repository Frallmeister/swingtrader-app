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
- mean market-relative forward return among the top `k` candidates.

The date-level results remain available for distribution plots and regime inspection. Aggregate values alone are not sufficient evidence that one formulation is stable.

## Ranker Feature and Parameter Tuning

`notebooks/workflows/modeling/11_cross_sectional_xgboost_ranker_tuning.ipynb` narrows the study to `XGBRanker`. It calculates one broad `FeatureSetSpec` once, then slices that loaded frame into curated feature candidates for every trial. Expanding folds remain confined to outer train.

The search evaluates the highest-ranked stocks using:

- mean future cross-sectional percentile among the top `k`;
- mean and median market-relative forward return among the top `k`;
- top-quintile enrichment and the fraction of dates with positive top-`k` relative return;
- fold-level dispersion and worst-fold performance.

The generic base uses a small deterministic sample of hyperparameters for every feature candidate. The search size is intentionally visible and modest; it is an exploratory comparison, not a reusable production tuner. One chosen trial is fitted on complete outer train and inspected once on outer validation. The locked test remains untouched.

## Reusable Helpers

`swingtrader.modeling.training.ranking` contains only the two small operations shared by the notebooks:

- `prepare_xgboost_ranking_data()` sorts rows into contiguous provider/date groups and returns XGBoost query IDs;
- `evaluate_cross_sectional_scores()` calculates the common date-level and aggregate ranking diagnostics.

Model fitting, feature candidates, tuning metrics, and parameter choices remain visible in the notebooks so they can be changed quickly during exploration.

## Working With the Notebooks

The committed notebooks are generic bases. When dates, features, targets, or parameters are changed for a material study:

1. run the notebook;
2. export it to HTML under `notebooks/exports/`;
3. retain the HTML locally as the record of that run;
4. restore the notebook to its generic base state before committing unrelated experiments.

A more formal training implementation is justified only after the notebooks identify a formulation worth retaining.
