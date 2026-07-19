# Features

Feature generation currently supports in-memory historical return, trend, momentum, and volatility features for exploratory analysis and baseline modeling. Persistent feature tables and versioned feature pipelines are still future work.

## Intended Role

Feature code should transform bronze market and macro data into model-ready records. It should be deterministic and rerunnable.

Current feature transformations operate on pandas dataframes and preserve the input row alignment. Market-price data used by the feature layer must have a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, in that exact order. The identifiers must not also appear as ordinary columns, and feature functions never reset, set, or sort the index for the caller. Each feature family additionally requires the value columns it consumes, such as `adjusted_close`.

A valid call looks like:

```python
prices = prices.set_index(["provider", "ticker", "trading_date"]).sort_index()

features = add_trend_features(prices)
```

External consumers that need identifiers as columns, for example database writes, CSV export, APIs, or plotting, convert explicitly at their own boundary:

```python
records = features.reset_index()
```

Feature functions follow two contracts:

- public numerical indicators operate per ticker and return either one index-aligned `pd.Series` or, for naturally multi-output indicators, one index-aligned `pd.DataFrame`. Most indicators take a single ordered `pd.Series`; indicators that need several price columns at once, such as the volatility indicators consuming `high`, `low`, and `close`, take a `pd.DataFrame` instead;
- application feature orchestrators such as `add_return_features`, `add_trend_features`, `add_momentum_features`, and `add_volatility_features` return a copy of the input dataframe with final model feature columns added.

## Return Features

The return feature orchestrator is `swingtrader.data.features.returns.add_return_features`. It validates the input once, copies it, calculates trailing percentage-return columns named `return_{horizon}d`, and appends them to the copied dataframe. Each horizon is a positive integer number of trading rows.

For example, `horizons=(1, 5, 10)` produces `return_1d`, `return_5d`, and `return_10d` from `adjusted_close` values. Calculations are grouped by `provider` and `ticker`, so one ticker's history cannot leak into another ticker's features. Within each provider/ticker group, input rows must be strictly ordered by `trading_date`; warm-up rows without enough history remain missing.

## Trend Features

The trend feature orchestrator is `swingtrader.data.features.trends.add_trend_features`. It validates the source prices once, copies them, calculates the final trend model features from `adjusted_close`, and appends those columns while preserving input row alignment.

With the default settings, the orchestrator adds:

- `sma_fast_to_sma_slow`, the fast SMA divided by the slow SMA minus one;
- `ema_fast_to_ema_slow`, the fast EMA divided by the slow EMA minus one;
- `ema_fast_to_sma_fast`, the fast EMA divided by the fast SMA minus one.

The public numerical trend indicators are:

- `sma`, which has one natural output and returns a series;
- `ema`, which has one natural output and returns a series.

Each indicator accepts either one ordered series for a single ticker or a multi-ticker series that carries the canonical `provider`, `ticker`, and `trading_date` index levels. A standalone single-ticker series does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved. A partial or wrongly ordered MultiIndex, such as `["ticker", "trading_date"]`, is rejected.

The default fast/slow moving-average lengths are 20 and 50 rows. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each rolling or exponential calculation has enough observations. Intermediate moving-average values such as `sma_fast`, `sma_slow`, `ema_fast`, and `ema_slow` are local calculations and are not persisted as feature columns. The trend module is intended to later host directional indicators such as `adx`, `plus_di`, and `minus_di`.

SMA and EMA validate their local parameters. A standalone single-ticker series is rejected only when its datetime or period index is visibly unordered. A multi-ticker series must satisfy the canonical market-price index contract, and the calculation stays within each provider/ticker group. They do not perform dataframe-level column validation and do not sort input values.

## Momentum Features

The momentum feature orchestrator is `swingtrader.data.features.momentum.add_momentum_features`. It validates the source prices once, copies them, calculates the final momentum model features from `adjusted_close`, and appends those columns while preserving input row alignment.

With the default settings, the orchestrator adds:

- `ppo`, the fast/slow EMA oscillator as a ratio;
- `ppo_signal`, an EMA signal line over `ppo`;
- `ppo_histogram`, the difference between `ppo` and `ppo_signal`;
- `ppo_percentile`, the point-in-time percentile rank of `ppo` within prior valid PPO observations for the same provider/ticker group;
- `rsi`, Wilder's Relative Strength Index calculated from `adjusted_close`;
- `rsi_percent_b`, the position of the `rsi` line within its own Bollinger bands.

The public numerical momentum indicators are:

- `ppo`, which has three natural outputs and returns a dataframe with `ppo`, `ppo_signal`, and `ppo_histogram` columns;
- `ppo_percentile`, which has one natural output and returns a series;
- `rsi`, which has one natural output and returns a bounded `[0, 100]` oscillator series;
- `macd`, which has three natural outputs and returns a dataframe with `macd`, `macd_signal`, and `macd_histogram` columns expressed in the input price units.

Each indicator accepts either one ordered series for a single ticker or a multi-ticker series that carries the canonical `provider`, `ticker`, and `trading_date` index levels. A standalone single-ticker series does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved. A partial or wrongly ordered MultiIndex, such as `["ticker", "trading_date"]`, is rejected.

The default PPO lengths are 12, 26, and 9 rows, and `add_momentum_features` requires 100 prior valid PPO observations before `ppo_percentile` is populated by default. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each exponential or expanding-history calculation has enough observations. The momentum module is intended to later host additional oscillators such as rate-of-change.

PPO and PPO percentile validate their local parameters. A standalone single-ticker series is rejected only when its datetime or period index is visibly unordered. A multi-ticker series must satisfy the canonical market-price index contract, and the calculation stays within each provider/ticker group. They do not perform dataframe-level column validation and do not sort input values. PPO signal and histogram are part of the cohesive `ppo` output rather than separate public functions.

`macd` shares the PPO length validation and grouping semantics but returns the raw fast-minus-slow EMA difference in the input price units instead of a scaled ratio. It is not included in `add_momentum_features`; it is exposed as a standalone indicator so future consumers, such as the frontend application, can compute MACD, signal, and histogram values directly. The default lengths are 12, 26, and 9 rows.

`rsi` operates on a single ordered series, so the caller chooses the source, such as close, adjusted close, or an OHLC average. It is a bounded `[0, 100]` oscillator built from the average gain and average loss over `length` rows, each smoothed with Wilder's moving average, and calculated as `100 * avg_gain / (avg_gain + avg_loss)`. A window with no losses returns 100 and a window with no gains returns 0, while a fully flat window has neither gains nor losses and is left missing. The Wilder smoothing is the recursive form seeded from the first change rather than the canonical definition that seeds from the simple average of the first `length` changes, so early values differ slightly before converging, matching the ATR behavior in the volatility module. The first `length` rows of each series remain missing until the window is full.

Inside `add_momentum_features`, `rsi` is calculated from `adjusted_close` so its gains and losses are not distorted by split and dividend discontinuities in the raw close, matching the return, trend, and volatility families. `rsi_percent_b` then reuses the volatility module's `bollinger_percent_b` on the `rsi` line, locating momentum within its own recent range as a scale-invariant feature. The default RSI length is 14 rows, calibratable through `rsi_length` on `add_momentum_features` and `length` on `rsi`, and the RSI Bollinger bands default to 20 rows with 2 standard deviations, calibratable through `rsi_bollinger_length` and `rsi_bollinger_num_std`.

## Volatility Features

The volatility feature orchestrator is `swingtrader.data.features.volatility.add_volatility_features`. It validates the source prices once, copies them, calculates the final volatility model features from `high`, `low`, `close`, and `adjusted_close`, and appends them while preserving input row alignment.

With the default settings, the orchestrator adds:

- `atr_percent`, the Average True Range expressed as a percentage of the closing price;
- `bollinger_bandwidth`, the width between the upper and lower Bollinger bands relative to the middle band, calculated from `adjusted_close`;
- `bollinger_percent_b`, the position of `adjusted_close` within its Bollinger bands.

The public numerical volatility indicators are:

- `true_range`, which returns a series with the greatest of the current high-low range, the absolute gap between the current high and the previous close, and the absolute gap between the current low and the previous close;
- `atr`, which returns a series with Wilder's smoothed moving average of `true_range` in the input price units;
- `atr_percent`, which returns a series with `atr` divided by the current close and scaled to percentage points;
- `bollinger_bands`, which returns a dataframe with `bollinger_middle`, `bollinger_upper`, and `bollinger_lower` columns in the input units;
- `bollinger_bandwidth`, which returns a series with the band width relative to the middle band;
- `bollinger_percent_b`, which returns a series with the position within the bands.

The indicators split into two input shapes. `true_range`, `atr`, and `atr_percent` consume several price columns, so each accepts a dataframe with `high`, `low`, and `close` columns. The Bollinger indicators instead operate on a single ordered series, such as `adjusted_close` or any other signal, which keeps them reusable: the momentum module applies `bollinger_percent_b` to its RSI signal to expose the `rsi_percent_b` feature. A standalone single-ticker input does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved.

Inside `add_volatility_features`, ATR is calculated from raw `high`, `low`, and `close` because True Range needs the intraday extremes together, whereas the Bollinger features are calculated from `adjusted_close`. Using the adjusted series keeps the rolling mean and standard deviation from being distorted by the split and dividend discontinuities in the raw close, and matches the return, trend, and momentum families.

The default ATR length is 14 rows and is calibratable through the `atr_length` argument on `add_volatility_features` and the `length` argument on `atr` and `atr_percent`. True Range uses the previous close taken within each provider/ticker group, so the first row of each ticker falls back to its high-low range. ATR then applies Wilder's smoothing, leaving the first `length - 1` rows of each ticker missing until the window is full.

Wilder's smoothing here is the recursive exponential form seeded from the first True Range value, not the canonical definition that seeds the first ATR with the simple average of the first `length` True Ranges. The two forms converge quickly as more observations accrue, but early ATR (and `atr_percent`) values differ slightly from a canonical implementation.

The default Bollinger length is 20 rows with 2 standard deviations, both calibratable through the `bollinger_length` and `bollinger_num_std` arguments on `add_volatility_features` and the `length` and `num_std` arguments on the Bollinger indicators. The middle band is the simple moving average, and the outer bands sit `num_std` rolling standard deviations away, leaving the first `length - 1` rows of each series missing until the window is full. The rolling standard deviation is the population standard deviation (`ddof=0`), matching John Bollinger's original definition and most charting platforms; implementations that use the sample standard deviation (`ddof=1`) produce slightly wider bands for the same `length`.

Raw `true_range`, `atr`, and `bollinger_bands` are expressed in the input price units and are not comparable across tickers, so `add_volatility_features` only appends the scale-invariant `atr_percent`, `bollinger_bandwidth`, and `bollinger_percent_b` columns. `true_range`, `atr`, and `bollinger_bands` are exposed as standalone indicators, analogous to `macd`, so consumers such as exploratory analysis and the frontend application can obtain absolute price-unit values directly. The volatility module is intended to later host additional range and dispersion measures.

## Future Feature Ideas

- volume features;
- opening gap features such as next open versus previous close;
- later macro and market-context joins.

## Design Constraints

- Feature code reads from bronze data, not directly from yfinance.
- Feature inputs must use a unique, sorted `provider`/`ticker`/`trading_date` MultiIndex.
- Warmup periods should be represented explicitly.
- Features should avoid point-in-time leakage.
- Labels should be generated separately from input features.
- If feature persistence is introduced, it should support later train, validation, and test splits.