# Features

Feature generation currently supports in-memory historical return and trend features for exploratory analysis and baseline modeling. Persistent feature tables and versioned feature pipelines are still future work.

## Intended Role

Feature code should transform bronze market and macro data into model-ready records. It should be deterministic and rerunnable.

Current feature transformations operate on pandas dataframes and preserve the input row alignment. Source observations must include `provider`, `ticker`, and `trading_date` identifiers either all as columns or all as named index levels, plus the input columns required by each feature family. Identifier fields must not be split between columns and index levels.

## Return Features

The implemented return feature helper is `swingtrader.data.features.returns.add_return_features`. It adds trailing percentage-return columns named `return_{horizon}d`, where each horizon is a positive integer number of trading rows.

For example, `horizons=(1, 5, 10)` produces `return_1d`, `return_5d`, and `return_10d` from `adjusted_close` values. Calculations are grouped by `provider` and `ticker`, so one ticker's history cannot leak into another ticker's features. Within each provider/ticker group, input rows must be strictly ordered by `trading_date`; warm-up rows without enough history remain missing.

## Trend Features

The implemented trend feature helper is `swingtrader.data.features.trends.add_trend_features`. It adds moving-average and oscillator columns from `adjusted_close` values while preserving input row alignment.

With the default settings, the helper adds:

- `sma_fast_to_sma_slow`, the fast SMA divided by the slow SMA minus one;
- `ema_fast_to_ema_slow`, the fast EMA divided by the slow EMA minus one;
- `ema_fast_to_sma_fast`, the fast EMA divided by the fast SMA minus one;
- `ppo`, the fast/slow EMA oscillator as a ratio;
- `ppo_signal`, an EMA signal line over `ppo`;
- `ppo_histogram`, the difference between `ppo` and `ppo_signal`.

The default fast/slow moving-average lengths are 20 and 50 rows. The default PPO lengths are 12, 26, and 9 rows. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each rolling or exponential window has enough observations. The lower-level helpers `sma`, `ema`, `ppo`, `ppo_signal`, and `ppo_histogram` are available when notebooks or experiments need individual indicators.

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