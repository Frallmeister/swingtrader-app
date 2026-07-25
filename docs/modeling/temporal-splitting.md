# Temporal Splitting

The experiment layer applies one fixed chronological train, validation, and locked-test policy to a canonical `TemporalDatasetBundle`. It does not recompute features or targets and does not mutate the unsplit bundle.

## Containment Rule

Each declared range is inclusive and shared by every ticker. Assignment has two stages:

1. the signal date must lie inside the range;
2. the row's actual `target_end_date` must be no later than that range's end.

For example:

```text
training ends:       2021-12-31
validation begins:   2022-01-01
signal date:         2021-12-29
target_end_date:     2022-01-05
```

The row is a training candidate by signal date but is purged because its target observation window is not contained in training. The same rule removes validation rows whose targets resolve after validation, including inside the locked test period. A target ending exactly on the split end is retained.

Purging uses the per-row metadata produced by the canonical dataset. It never converts a session horizon into an assumed number of calendar days.

## Embargo Semantics

`embargo_sessions=0` disables embargo. A positive value removes an additional number of signal dates before each later split:

1. purge boundary-crossing rows;
2. find the remaining distinct signal dates in train or validation across the complete panel;
3. remove the final N dates and every surviving ticker row on those dates.

This is a global pre-boundary gap. It is not calculated per ticker, does not shift the declared validation or test ranges, and is not applied to the end of the locked test period. Dates already emptied by purging do not count toward the embargo, so N always represents N additional observed signal dates.

## Usage

```python
from swingtrader.modeling.experiments import FixedTemporalSplitter

splitter = FixedTemporalSplitter(experiment_spec.split)
split_result = splitter.assign(bundle)

train_index = split_result.indices("train")
validation_index = split_result.indices("validation")
locked_test_index = split_result.indices("test")

X_train = bundle.features.iloc[train_index]
y_train = bundle.targets.iloc[train_index]
```

For scikit-learn-style model selection, the splitter yields one train/validation pair and deliberately omits test:

```python
for train_index, validation_index in splitter.split(bundle):
    ...
```

Access the test indices only for final locked evaluation.

## Assignment Metadata

`split_result.samples` retains every canonical sample and adds:

- `split`: `train`, `validation`, or `test` for assigned rows;
- `split_exclusion_reason`: `outside_declared_ranges`, `target_end_after_split_end`, or `embargo` for excluded rows.

Exactly one of these columns is populated for each row. This preserves an auditable account of samples outside the selected periods and samples removed for leakage control.

## Diagnostics

The deterministic split manifest records:

- source, assigned, outside-range, purged, and embargoed row counts;
- candidate and assigned rows per split;
- unique signal dates and tickers per split;
- observed signal-date and target-end-date ranges;
- binary-class prevalence when the selected classification target is Boolean or encoded as 0/1;
- the dataset-manifest and split-specification digests.

Use coverage diagnostics when choosing split dates. Each post-purge split must retain enough dates, tickers, and positive labels for its intended role. The splitter validates semantics but does not choose statistically suitable dates automatically.
