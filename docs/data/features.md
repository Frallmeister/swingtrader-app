# Features

Feature generation currently supports in-memory historical return features for exploratory analysis and baseline modeling. Persistent feature tables and versioned feature pipelines are still future work.

## Intended Role

Feature code should transform bronze market and macro data into model-ready records. It should be deterministic and rerunnable.

Current feature transformations operate on pandas dataframes and preserve the input row alignment. Source observations should include `provider`, `ticker`, and `trading_date` identifiers, plus the input columns required by each feature family.

## Return Features

The implemented return feature helper is `swingtrader.data.features.returns.add_return_features`. It adds trailing percentage-return columns named `return_{horizon}d`, where each horizon is a positive integer number of trading rows.

For example, `horizons=(1, 5, 10)` produces `return_1d`, `return_5d`, and `return_10d` from `adjusted_close` values. Calculations are grouped by `provider` and `ticker`, so one ticker's history cannot leak into another ticker's features. Warm-up rows without enough history remain missing.

## Future Feature Ideas

- moving averages;
- volatility measures;
- volume features;
- opening gap features such as next open versus previous close;
- later macro and market-context joins.

## Design Constraints

- Feature code reads from bronze data, not directly from yfinance.
- Warmup periods should be represented explicitly.
- Features should avoid point-in-time leakage.
- Labels should be generated separately from input features.
- If feature persistence is introduced, it should support later train, validation, and test splits.