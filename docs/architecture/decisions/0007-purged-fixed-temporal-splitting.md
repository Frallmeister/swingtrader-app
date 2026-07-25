# ADR 0007: Purged Fixed Temporal Splitting

- **Status:** Accepted
- **Date:** 2026-07-25

## Context

The canonical temporal bundle records each signal date and its actual `target_end_date`, but it intentionally has no split assignment. A multi-ticker panel cannot safely use random row splitting or ticker-local cutoffs: all instruments must be evaluated against the same market periods, and a label must not consume observations outside the split where its signal is assigned.

The term *embargo* is ambiguous. It can mean removing observations after an evaluation interval, delaying the next split, or adding a gap before a later split. The initial workflow has one forward-only train/validation/locked-test holdout, not repeated cross-validation with training data after an evaluation period.

## Decision

Keep canonical dataset construction unsplit under `modeling.datasets`. Implement fixed split policy and execution downstream under `modeling.experiments`.

Train, validation, and test ranges are inclusive calendar ranges shared by every ticker. A row is first a candidate according to its signal date. It is assigned only when its actual `target_end_date` is no later than that candidate split's end date. Rows in gaps or outside the declared ranges remain in split metadata with an explicit exclusion reason.

Define `embargo_sessions` as an additional pre-boundary gap. After target-end purging, remove the final N distinct global observed signal dates from train and validation. Every ticker with a surviving row on one of those dates is removed. The embargo is not calculated independently per ticker, does not move validation or test starts, and is not applied after the locked test period.

The splitter returns copied canonical sample metadata with assignment and exclusion columns, deterministic diagnostics, and positional indices into the original bundle. Its scikit-learn-style iterator yields only train and validation indices; locked test indices require explicit access.

## Consequences

- A training signal whose target resolves in validation or later is purged even when its signal date lies in the training range.
- A validation signal whose target resolves after validation is likewise purged.
- Different ticker histories do not create different boundaries or ticker-local embargo dates.
- The canonical feature and target frames are not duplicated or mutated by split assignment.
- Diagnostics report candidate, assigned, purged, and embargoed rows together with dates, tickers, and binary-class prevalence.
- Expanding-window walk-forward folds remain a later extension of the same containment rules.
