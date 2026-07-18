# Features

Feature generation currently supports in-memory historical return and trend features for exploratory analysis and baseline modeling. Persistent feature tables and versioned feature pipelines are still future work.

## Intended Role

Feature code should transform bronze market and macro data into model-ready records. It should be deterministic and rerunnable.

Current feature transformations operate on pandas dataframes and preserve the input row alignment. Source observations must include `provider`, `ticker`, and `trading_date` identifiers either all as columns or all as named index levels, plus the input columns required by each feature family. Identifier fields must not be split between columns and index levels.

Feature functions follow three contracts:

- numerical helpers operate on one `pd.Series` and return one index-aligned `pd.Series`;
- feature generators accept a source `pd.DataFrame` and return a `pd.DataFrame` containing only newly calculated model feature columns;
- feature-family orchestrators such as `add_return_features` and `add_trend_features` return a copy of the input dataframe with feature columns added.

## Return Features

The return feature generator is `swingtrader.data.features.returns.return_features`. It returns only trailing percentage-return columns named `return_{horizon}d`, where each horizon is a positive integer number of trading rows.

The orchestrator `swingtrader.data.features.returns.add_return_features` validates the input once, copies it, calls `return_features` without repeated validation, and appends the returned feature columns.

For example, `horizons=(1, 5, 10)` produces `return_1d`, `return_5d`, and `return_10d` from `adjusted_close` values. Calculations are grouped by `provider` and `ticker`, so one ticker's history cannot leak into another ticker's features. Within each provider/ticker group, input rows must be strictly ordered by `trading_date`; warm-up rows without enough history remain missing.

## Trend Features

The trend feature generators return dataframes containing only new feature columns from `adjusted_close` or previously generated PPO values. The orchestrator `swingtrader.data.features.trends.add_trend_features` validates the source prices once, copies them, calls the generators without repeated validation, and appends the generated columns while preserving input row alignment.

With the default settings, the orchestrator adds:

- `sma_fast_to_sma_slow`, the fast SMA divided by the slow SMA minus one;
- `ema_fast_to_ema_slow`, the fast EMA divided by the slow EMA minus one;
- `ema_fast_to_sma_fast`, the fast EMA divided by the fast SMA minus one;
- `ppo`, the fast/slow EMA oscillator as a ratio;
- `ppo_signal`, an EMA signal line over `ppo`;
- `ppo_histogram`, the difference between `ppo` and `ppo_signal`;
- `ppo_percentile`, the point-in-time percentile rank of `ppo` within prior valid PPO observations for the same provider/ticker group.

The dataframe-returning trend generators are:

- `moving_average_features`, which returns `sma_fast_to_sma_slow`, `ema_fast_to_ema_slow`, and `ema_fast_to_sma_fast`;
- `ppo_features`, which returns `ppo`, `ppo_signal`, and `ppo_histogram`;
- `ppo_percentile_features`, which returns `ppo_percentile`.

The default fast/slow moving-average lengths are 20 and 50 rows. The default PPO lengths are 12, 26, and 9 rows, and `ppo_percentile_features` requires 100 prior valid PPO observations by default when used through `add_trend_features`. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each rolling, exponential, or expanding-history calculation has enough observations.

The lower-level numerical helpers `sma`, `ema`, `ppo`, `ppo_signal`, `ppo_histogram`, and `ppo_percentile` operate on one series and return one index-aligned series. They do not perform dataframe-level feature validation or choose model feature-column names. SMA and EMA validate their local `length` parameter and reject visibly unordered temporal indexes, but they do not sort input values.

## Future Feature Ideas

- volatility measures;
- volume features;
- opening gap features such as next open versus previous close;
- later macro and market-context joins.

## Design Constraints

- Feature code reads from bronze data, not directly from yfinance.
- Feature inputs must be point-in-time ordered by `trading_date` within each provider/ticker group.
- Warmup periods should be represented explicitly.
- Features should avoid point-in-time leakage.
- Labels should be generated separately from input features.
- If feature persistence is introduced, it should support later train, validation, and test splits.