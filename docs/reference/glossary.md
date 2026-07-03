# Glossary

## Active Trading Universe

The production candidate set: tickers the deployed app should keep updated and eventually rank as possible trades.

## Available Universe

A curated ticker catalog, stored as a YAML artifact in the repository.

## Bronze

Source-oriented database layer that stores normalized provider data plus project metadata such as `provider`, `request_id`, and `fetched_at`.

## Candidate Ranking

The future model output that orders active trade candidates by expected opportunity over the target horizon.

## Inference-Ready

A ticker with the recent feature data required for production prediction.

## Label Horizon

The future period over which a target is calculated, such as 5 or 10 trading days.

## Onboarded

In the current bronze-only workflow, a ticker is onboarded once any bronze daily price row exists for it.

## Training-Eligible

A ticker with enough historical feature and label data to be included safely in model training and evaluation.

## Training Universe

The broader set of tickers that may be used for model development. It may be larger than the active trading universe.