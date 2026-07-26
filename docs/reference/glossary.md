# Glossary

## Active Trading Universe

The production candidate set: tickers the deployed app should keep updated and eventually rank as possible trades.

## Available Universe

A curated ticker catalog, stored as a YAML artifact in the repository.

## Bronze

Source-oriented database layer that stores normalized provider data plus project metadata such as `provider`, `request_id`, and `fetched_at`.

## Candidate Ranking

The future model output that orders active trade candidates by expected opportunity over the target horizon.

## Embargo

Removal of an additional number of observed signal dates near the end of a train or validation split after purging. An embargo creates extra temporal separation from the following split.

## Inference-Ready

A ticker with the recent feature data required for production prediction.

## Label Horizon

The future period over which a target is calculated, such as 5 or 10 trading days.

## Onboarded

In the current bronze-only workflow, a ticker is onboarded once any bronze daily price row exists for it.

## Panel

A dataset containing observations for multiple entities across time. In Swingtrader, a temporal panel contains ticker observations indexed by `provider`, `ticker`, and `trading_date`.

## Purge

Removal of a sample whose target-resolution date lies beyond the end of its proposed temporal split. Purging prevents information used to determine the target from crossing a train, validation, or test boundary.

## Training-Eligible

A ticker with enough historical feature and label data to be included safely in model training and evaluation.

## Training Universe

The broader set of tickers that may be used for model development. It may be larger than the active trading universe.