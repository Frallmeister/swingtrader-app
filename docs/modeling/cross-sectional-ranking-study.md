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

`notebooks/workflows/modeling/11_cross_sectional_xgboost_ranker_tuning.ipynb` narrows the study to `XGBRanker`. It calculates one broad `FeatureSetSpec` once, then slices every trial from the loaded frame. Expanding folds remain confined to outer train.

The notebook starts from one interpretable feature selection and repeats a small Optuna study. After each feature iteration, it averages gain importance across folds and the strongest trials, removes a small low-importance fraction, and randomly adds individual excluded features. Progress output reports each completed Optuna trial and feature iteration.

The search evaluates the highest-ranked stocks using:

- mean future cross-sectional percentile among the top `k`;
- mean and median market-relative forward return among the top `k`;
- top-quintile enrichment and the fraction of dates with positive top-`k` relative return;
- fold-level dispersion and worst-fold performance;
- train-versus-fold generalization gaps.

The search size and feature-update settings remain visible in the notebook. This is an exploratory comparison, not a reusable production tuner. Outer validation requires an explicitly chosen inner-fold trial, and the locked test remains untouched.

The notebook applies the current configured Large/Mid Cap universe retrospectively across the study period. Results are conditional on today's configured universe and may contain survivorship bias; they are not a reconstruction of historical point-in-time membership.

## Reusable Helpers

`swingtrader.modeling.training.ranking` contains only the two small operations shared by the notebooks:

- `prepare_xgboost_ranking_data()` sorts rows into contiguous provider/date groups and returns XGBoost query IDs;
- `evaluate_cross_sectional_scores()` calculates the common date-level and aggregate ranking diagnostics.

Model fitting, feature-selection rules, tuning metrics, and parameter choices remain visible in the notebooks so they can be changed quickly during exploration.

## Working With the Notebooks

The committed notebooks are generic bases. When dates, features, targets, or parameters are changed for a material study:

1. run the notebook;
2. export it to HTML under `notebooks/exports/`;
3. retain the HTML locally as the record of that run;
4. restore the notebook to its generic base state before committing unrelated experiments.

A more formal training implementation is justified only after the notebooks identify a formulation worth retaining.
