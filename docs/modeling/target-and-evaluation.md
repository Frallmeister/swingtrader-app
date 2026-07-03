# Target And Evaluation

The first model target and evaluation strategy are open design questions.

## Candidate Targets

- 5-day forward return.
- 10-day forward return.
- probability of positive return over a horizon.
- probability of outperforming the active universe or benchmark.
- rank score for cross-sectional candidate ordering.

## Candidate Metrics

- forward-return error;
- Spearman rank correlation;
- Pearson correlation;
- top-k average realized return;
- top-k hit rate;
- simple simulated portfolio growth;
- drawdown-aware evaluation.

## Ranking Requirement

The model should support ranking trade candidates, not only predicting exact returns.

## Open Decisions

- First horizon or horizons.
- Point prediction versus direct ranking objective.
- How to represent uncertainty or confidence.
- How to approximate transaction costs and stop-loss behavior during evaluation.
- How outputs map to the eventual frontend ranking.