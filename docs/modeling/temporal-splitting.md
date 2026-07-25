# Temporal Splitting

The experiment layer applies one fixed chronological train, validation, and locked-test policy to a canonical `TemporalDatasetBundle`. It does not recompute features or targets and does not mutate the unsplit bundle.

## Containment Rule

Each declared range is inclusive and shared by every ticker. Assignment has two stages:

1. the signal date must lie inside the range;
2. the row's actual `target_end_date` must be no later than that range's end.

The signal date determines the candidate split, while the complete target window determines whether the candidate is retained:

Signals and target end dates are states; the final hexagon is the assignment decision:

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 30}}}%%
flowchart TB
    train["Train through<br/>2021-12-31"] --> validation["Validation from<br/>2022-01-01"]

    subgraph retained["Sample A"]
        direction LR
        a_signal([Signal<br/>2021-12-29]) --> a_target([Target ends<br/>2021-12-31]) --> a_result{{Retain in train}}
    end

    subgraph purged["Sample B"]
        direction LR
        b_signal([Signal<br/>2021-12-29]) --> b_target([Target ends<br/>2022-01-05]) --> b_result{{Purge from train}}
    end

    a_target -.-> train
    b_target -.-> validation

    classDef boundary fill:#1565c0,stroke:#0d3d75,color:#ffffff
    classDef state fill:#455a64,stroke:#263238,color:#ffffff
    classDef retainedState fill:#2e7d32,stroke:#17451c,color:#ffffff
    classDef purgedState fill:#a12727,stroke:#5f1515,color:#ffffff
    class train,validation boundary
    class a_signal,a_target,b_signal,b_target state
    class a_result retainedState
    class b_result purgedState
```

Both rows are training candidates by signal date. Sample A is retained because its target ends on the inclusive train boundary; Sample B is purged because its target window crosses that boundary. The same rule removes validation rows whose targets resolve after validation, including inside the locked test period.

Purging uses the per-row metadata produced by the canonical dataset. It never converts a session horizon into an assumed number of calendar days.

## Embargo Semantics

`embargo_sessions=0` disables embargo. A positive value applies an additional global pre-boundary gap after purging:

Rounded boxes are operations, the cylinder is the surviving panel, and the final hexagon is the
resulting boundary state:

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 26, "rankSpacing": 32}}}%%
flowchart TB
    candidates[(Train or validation candidates)]
    purge([Purge target windows crossing the split end])
    panel[(Surviving panel dates<br/>Dec 27: AAA, BBB<br/>Dec 28: AAA, CCC<br/>Dec 29: AAA, BBB, CCC<br/>Dec 30: AAA, BBB)]
    choose([For N = 2, select Dec 29 and Dec 30])
    remove([Remove all surviving rows on those dates])
    later{{Later split keeps its declared start}}

    candidates --> purge --> panel --> choose --> remove --> later

    classDef action fill:#9a4d00,stroke:#5d2e00,color:#ffffff
    classDef artifact fill:#2e7d32,stroke:#17451c,color:#ffffff
    classDef state fill:#6a1b9a,stroke:#3c0f58,color:#ffffff
    class candidates,panel artifact
    class purge,choose,remove action
    class later state
```

The dates are selected once across the complete panel, not separately for each ticker. The embargo does not shift validation or test ranges and is not applied after the locked test period. Dates already emptied by purging do not count, so `N` always means `N` additional observed signal dates.

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
