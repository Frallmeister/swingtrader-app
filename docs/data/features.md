# Features

Feature generation currently supports in-memory historical return, trend, momentum, volatility, price-action, volume, and market-structure features for exploratory analysis, screening, backtest analysis, and baseline modeling. Persistent feature tables and versioned feature pipelines are still future work.

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

Feature code is organized into two layers with a clear responsibility boundary:

- **Indicators** calculate reusable technical quantities and live in `swingtrader.indicators`. Indicators know nothing about the model feature set, so they can also be reused by notebooks, tests, future API endpoints, charting, stock screening, and backtest or trade-record analysis.
- **Features** transform raw data and indicators into model inputs and live in `swingtrader.data.features`. A feature builder decides which source columns an indicator should use, how indicators are combined and normalized, how historical context is represented, and what the model-facing column is named.

In short: indicators calculate reusable technical quantities, and features transform raw data and indicators into model inputs.

### Adjustment-Consistent Model Prices

Model features that compare prices across sessions use a common price representation. The feature layer multiplies each requested raw OHLC value by the row-wise factor `adjusted_close / close`, placing open, high, low, and close on the adjusted-close scale. The transformed close is set directly to `adjusted_close`.

This preserves same-session candle geometry while preventing stock splits and dividend adjustments from appearing as overnight gaps, directional movement, volatility, momentum, or market-structure changes. Source volume is not transformed. Features that combine price and volume, such as rolling VWAP and MFI, therefore use adjustment-consistent prices together with the provider's volume series. Turnover defines its own model-facing quantity as `adjusted_close * volume`.

The indicator layer remains source-agnostic. Standalone indicators calculate from whichever series or price frame the caller supplies, which permits raw-price charting and analysis. Canonical feature orchestrators own the adjustment decision so persisted model columns do not accidentally mix incompatible price histories. `zigzag_features` also accepts an explicitly selected `high`/`low`/`close` frame without `adjusted_close` for direct use; canonical model inputs include `adjusted_close` and use the adjustment-consistent path.

Feature functions follow two contracts:

- public numerical indicators in `swingtrader.indicators` operate per ticker and return either one index-aligned `pd.Series` or, for naturally multi-output indicators, one index-aligned `pd.DataFrame`. Most indicators take a single ordered `pd.Series`; indicators that need several price columns at once, such as the volatility indicators consuming `high`, `low`, and `close`, take a `pd.DataFrame` instead. Every indicator supports two input forms: a single ordered instrument (only required to be chronologically ordered), or a canonical multi-instrument market frame with a `provider`/`ticker`/`trading_date` MultiIndex, in which case calculations are isolated per group and the input index and row order are preserved;
- application feature orchestrators such as `add_return_features`, `add_trend_features`, `add_momentum_features`, `add_volatility_features`, `add_price_action_features`, `add_volume_features`, and `add_market_structure_features` return a copy of the input dataframe with final model feature columns added. They are importable from `swingtrader.data.features`, and `swingtrader.data.features.pipeline.add_default_features` runs the standard families in a fixed order.

Every public `add_*_features` family orchestrator rejects input columns that have the same names as its generated features. This prevents applying a family twice, or combining families with an accidental naming collision, from silently replacing existing data.

## Return Features

The return feature orchestrator is `swingtrader.data.features.returns.add_return_features`. It validates the input once, copies it, calculates trailing percentage-return columns named `return_{horizon}d`, and appends them to the copied dataframe. Each horizon is a positive integer number of trading rows.

For example, `horizons=(1, 5, 10)` produces `return_1d`, `return_5d`, and `return_10d` from `adjusted_close` values. Calculations are grouped by `provider` and `ticker`, so one ticker's history cannot leak into another ticker's features. Within each provider/ticker group, input rows must be strictly ordered by `trading_date`; warm-up rows without enough history remain missing.

## Trend Features

The trend feature orchestrator is `swingtrader.data.features.trend.add_trend_features`. It validates the source prices once, copies them, calculates the final trend model features, and appends those columns while preserving input row alignment.

With the default settings, the orchestrator adds:

- `ema_fast_to_ema_mid`, the fast EMA divided by the mid EMA minus one;
- `ema_mid_to_ema_slow`, the mid EMA divided by the slow EMA minus one;
- `ema_mid_to_sma_mid`, the mid EMA divided by the mid SMA minus one;
- `close_to_ema_fast`, the adjusted close divided by the fast EMA minus one;
- `close_to_ema_mid`, the adjusted close divided by the mid EMA minus one;
- `close_to_ema_slow`, the adjusted close divided by the slow EMA minus one;
- `adx`, Wilder's Average Directional Index measuring trend strength;
- `plus_di`, the positive directional indicator measuring upward directional movement;
- `minus_di`, the negative directional indicator measuring downward directional movement;
- `vwap_distance`, adjusted close divided by adjustment-consistent rolling VWAP minus one;
- `vwap_distance_percent_b`, the position of `vwap_distance` within its own Bollinger bands.

The public numerical trend indicators, importable from `swingtrader.indicators`, are:

- `sma`, which has one natural output and returns a series;
- `ema`, which has one natural output and returns a series;
- `adx`, which has three natural outputs and returns a dataframe with `adx`, `plus_di`, and `minus_di` columns;
- `rolling_vwap`, which consumes `high`, `low`, `close`, and `volume` and returns the trailing volume-weighted average typical price.

Each indicator accepts either one ordered single-instrument input or a canonical multi-instrument input carrying the `provider`, `ticker`, and `trading_date` index levels. `sma` and `ema` consume a `Series`, while `adx` and `rolling_vwap` consume a `DataFrame` containing their required columns. A standalone single-instrument input does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present, the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's. The original index and row order are preserved. A partial or wrongly ordered MultiIndex, such as `["ticker", "trading_date"]`, is rejected.

The default fast/mid/slow moving-average lengths are 10, 20, and 50 rows and must be strictly ascending. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each rolling or exponential calculation has enough observations. Intermediate moving-average values such as `sma_mid`, `ema_fast`, `ema_mid`, and `ema_slow` are local calculations and are not persisted as feature columns.

SMA and EMA validate their local parameters. A standalone single-ticker series is rejected only when its datetime or period index is visibly unordered. A multi-ticker series must satisfy the canonical market-price index contract, and the calculation stays within each provider/ticker group. They do not perform dataframe-level column validation and do not sort input values.

`adx` implements Wilder's directional-movement system and, unlike the moving-average ratios, consumes several price columns at once, so it takes a dataframe with `high`, `low`, and `close` columns rather than a single series. It returns the `adx`, `plus_di`, and `minus_di` columns together as one cohesive output, analogous to `macd` and `bollinger_bands`. The positive and negative directional indicators measure the share of smoothed True Range attributable to upward and downward directional movement over `length` rows, and ADX is Wilder's smoothed moving average of the directional index `DX` over the same `length`. All three are bounded to `[0, 100]`: `plus_di` and `minus_di` gauge direction while `adx` gauges strength regardless of direction.

Inside `add_trend_features`, ADX, `plus_di`, and `minus_di` are calculated from `high`, `low`, and `close` placed on the adjusted-close scale. The directional-movement system still receives the intraday extremes together, but corporate-action adjustments cannot appear as movement between sessions. The smoothing length is the conventional 14 rows by default and is calibratable through the `adx_length` argument on `add_trend_features` and the `length` argument on `adx`. Both the directional indicators and ADX use Wilder's recursive smoothing seeded from the first observation rather than the canonical definition that seeds from the simple average of the first `length` observations, so early values differ slightly before converging, matching the ATR and RSI behavior elsewhere. Because ADX smooths `DX` a second time, its warm-up spans roughly `2 * length` rows before values become populated.

Rolling VWAP uses each row's typical price, `(high + low + close) / 3`, and weights it by that row's volume. Over a trailing window it is calculated as the rolling sum of typical price times volume divided by the rolling sum of volume. This is a moving VWAP over daily observations, not an intraday session VWAP that resets at each market open.

Inside `add_trend_features`, `vwap_distance` is calculated as `adjusted_close / rolling_vwap - 1`. Rolling VWAP uses adjustment-consistent `high`, `low`, and `close` together with source `volume`. Positive values mean that adjusted close is above the recent volume-weighted price level, negative values mean that it is below, and zero means that it is equal to VWAP.

`vwap_distance_percent_b` applies `bollinger_percent_b` to the VWAP-distance series itself. A value of 0 marks the lower Bollinger band, 1 marks the upper band, and values outside `[0, 1]` lie beyond the bands. This provides historical context for the distance: the same absolute displacement can be ordinary for one ticker or unusually stretched for another.

The default rolling VWAP length is 20 rows. The Bollinger transformation over VWAP distance also defaults to 20 rows and 2 population standard deviations. These values are calibratable through `vwap_length`, `vwap_bollinger_length`, and `vwap_bollinger_num_std`. Warm-up observations remain missing until the underlying VWAP and Bollinger windows have enough valid history.

## Momentum Features

The momentum feature orchestrator is `swingtrader.data.features.momentum.add_momentum_features`. It validates the source prices once, copies them, calculates PPO and RSI from `adjusted_close`, and calculates stochastic, MFI, and LazyBear squeeze features from adjustment-consistent `high`, `low`, and `close`. MFI combines that price frame with source `volume`. The resulting columns preserve input row alignment.

With the default settings, the orchestrator adds:

- `ppo`, the fast/slow EMA oscillator as a ratio;
- `ppo_signal`, an EMA signal line over `ppo`;
- `ppo_histogram`, the difference between `ppo` and `ppo_signal`;
- `ppo_percentile`, the point-in-time percentile rank of `ppo` within prior valid PPO observations for the same provider/ticker group;
- `rsi`, Wilder's Relative Strength Index calculated from `adjusted_close`;
- `rsi_percent_b`, the position of the `rsi` line within its own Bollinger bands;
- `stochastic_k`, the smoothed stochastic %K locating the close within its recent high/low range;
- `stochastic_d`, a further simple moving average of `stochastic_k`;
- `mfi`, the Money Flow Index, a volume-weighted momentum oscillator calculated from `high`, `low`, `close`, and `volume`;
- `mfi_percent_b`, the position of the `mfi` line within its own Bollinger bands.

The orchestrator also appends the LazyBear squeeze momentum features:

- `squeeze_on`, true while the Bollinger Bands sit inside the Keltner Channels (a low-volatility squeeze);
- `squeeze_off`, true while the Bollinger Bands sit outside the Keltner Channels;
- `squeeze_released`, true on the first row after a squeeze that is no longer squeezed, marking the bar the squeeze fires;
- `squeeze_width_ratio`, the Bollinger-band width relative to the Keltner-channel width;
- `squeeze_momentum_atr`, the squeeze momentum histogram normalised by ATR;
- `squeeze_momentum_atr_change`, the row-over-row change in `squeeze_momentum_atr`;
- `squeeze_duration`, the number of consecutive rows the current squeeze has been on;
- `squeeze_release_duration`, the length of the squeeze that fired, recorded on the release row.

The public numerical momentum indicators, importable from `swingtrader.indicators`, are:

- `ppo`, which has three natural outputs and returns a dataframe with `ppo`, `ppo_signal`, and `ppo_histogram` columns;
- `rsi`, which has one natural output and returns a bounded `[0, 100]` oscillator series;
- `stochastic_oscillator`, which has two natural outputs and returns a dataframe with `stochastic_k` and `stochastic_d` columns bounded to `[0, 100]`, and which consumes a dataframe with `high`, `low`, and `close` columns rather than a single series;
- `mfi`, which has one natural output and returns a bounded `[0, 100]` oscillator series, and which consumes a dataframe with `high`, `low`, `close`, and `volume` columns rather than a single series;
- `macd`, which has three natural outputs and returns a dataframe with `macd`, `macd_signal`, and `macd_histogram` columns expressed in the input price units;
- `lazybear_squeeze_momentum`, which consumes a dataframe with `high`, `low`, and `close` columns and returns a dataframe with the squeeze state and momentum columns, computing True Range and ATR internally.

Each indicator accepts either one ordered series or DataFrame, as appropriate for that indicator for a single ticker or a multi-ticker series that carries the canonical `provider`, `ticker`, and `trading_date` index levels. A standalone single-ticker series does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved. A partial or wrongly ordered MultiIndex, such as `["ticker", "trading_date"]`, is rejected.

The default PPO lengths are 12, 26, and 9 rows, and `add_momentum_features` requires 100 prior valid PPO observations before `ppo_percentile` is populated by default. Calculations are grouped by `provider` and `ticker`, and warm-up rows remain missing until each exponential or expanding-history calculation has enough observations. The momentum module is intended to later host additional oscillators such as rate-of-change.

`ppo_percentile` is a public model-oriented feature transform rather than a technical indicator. It is available from `swingtrader.data.features.momentum`. It calculates the expanding, point-in-time percentile rank of a PPO series while preserving provider/ticker isolation.

PPO and PPO percentile validate their local parameters. A standalone single-ticker series is rejected only when its datetime or period index is visibly unordered. A multi-ticker series must satisfy the canonical market-price index contract, and the calculation stays within each provider/ticker group. They do not perform dataframe-level column validation and do not sort input values. PPO signal and histogram are part of the cohesive `ppo` output rather than separate public functions.

`macd` shares the PPO length validation and grouping semantics but returns the raw fast-minus-slow EMA difference in the input price units instead of a scaled ratio. It is not included in `add_momentum_features`; it is exposed as a standalone indicator so future consumers, such as the frontend application, can compute MACD, signal, and histogram values directly. The default lengths are 12, 26, and 9 rows.

`rsi` operates on a single ordered series, so the caller chooses the source, such as close, adjusted close, or an OHLC average. It is a bounded `[0, 100]` oscillator built from the average gain and average loss over `length` rows, each smoothed with Wilder's moving average, and calculated as `100 * avg_gain / (avg_gain + avg_loss)`. A window with no losses returns 100 and a window with no gains returns 0, while a fully flat window has neither gains nor losses and is left missing. The Wilder smoothing is the recursive form seeded from the first change rather than the canonical definition that seeds from the simple average of the first `length` changes, so early values differ slightly before converging, matching the ATR behavior in the volatility module. The first `length` rows of each series remain missing until the window is full.

Inside `add_momentum_features`, `rsi` is calculated from `adjusted_close` so its gains and losses are not distorted by split and dividend discontinuities in the raw close, matching the return, trend, and volatility families. `rsi_percent_b` then reuses the `bollinger_percent_b` indicator on the `rsi` line, locating momentum within its own recent range as a scale-invariant feature. The standalone `rsi` indicator defaults to the conventional 14 rows through its `length` argument, while `add_momentum_features` deliberately defaults `rsi_length` to 21 rows so the persisted model feature uses a smoother, slower oscillator. The RSI Bollinger bands default to 20 rows with 2 standard deviations, calibratable through `rsi_bollinger_length` and `rsi_bollinger_num_std`.

`stochastic_oscillator`, unlike the other momentum indicators, consumes several price columns at once, so it takes a dataframe with `high`, `low`, and `close` columns rather than a single series, analogous to `adx` in the trend module. The raw %K locates the close within its recent range as `100 * (close - lowest_low) / (highest_high - lowest_low)` over `k_length` rows, where `lowest_low` and `highest_high` are the rolling minimum low and maximum high over the same window. `stochastic_k` is that raw %K smoothed with a simple moving average over `k_smoothing` rows, and `stochastic_d` is a further simple moving average of `stochastic_k` over `d_length` rows. Passing `k_smoothing=1` yields the fast stochastic, while the conventional 14/3/3 defaults yield the slow stochastic. Both series are bounded to `[0, 100]`, a window whose highest high equals its lowest low has no range and is left missing, and the warm-up rows of each series remain missing until every rolling window is full.

Inside `add_momentum_features`, `stochastic_oscillator` is calculated from adjustment-consistent `high`, `low`, and `close`. It receives the intraday extremes and close together without treating corporate-action adjustments as momentum. The 14/3/3 slow-stochastic defaults are calibratable through the `stochastic_k_length`, `stochastic_k_smoothing`, and `stochastic_d_length` arguments on `add_momentum_features` and the `k_length`, `k_smoothing`, and `d_length` arguments on `stochastic_oscillator`.

`mfi`, like `stochastic_oscillator`, consumes several price columns at once, so it takes a dataframe with `high`, `low`, `close`, and `volume` columns rather than a single series. Each row's typical price is `(high + low + close) / 3` and its raw money flow is the typical price times `volume`. A row's money flow counts as positive when its typical price rose from the prior row and negative when it fell; a row whose typical price is unchanged contributes to neither. MFI is `100 * positive_flow / (positive_flow + negative_flow)` over the trailing `length` rows, so it is often described as a volume-weighted RSI: a window with no negative flow returns 100, a window with no positive flow returns 0, and a window whose typical price never changes has neither positive nor negative flow and is left missing. The first `length` rows of each series remain missing until the trailing window is full, because the first row has no prior typical price to compare against, so MFI warms up one row later than a plain rolling window.

Inside `add_momentum_features`, `mfi` is calculated from adjustment-consistent `high`, `low`, and `close` together with source `volume` because the money flow needs the intraday extremes, the close, and the traded volume together, matching the stochastic, ADX, and ATR calculations. `mfi_percent_b` then reuses the `bollinger_percent_b` indicator on the `mfi` line, locating it within its own recent range as a scale-invariant feature, mirroring `rsi_percent_b`. The default length is the conventional 14 rows, calibratable through the `mfi_length` argument on `add_momentum_features` and the `length` argument on `mfi`, and the MFI Bollinger bands default to 20 rows with 2 standard deviations, calibratable through `mfi_bollinger_length` and `mfi_bollinger_num_std`.

`lazybear_squeeze_momentum` is a pandas port of the open-source [Squeeze Momentum Indicator [LazyBear]](https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/), itself a derivative of John Carter's TTM Squeeze. Like `stochastic_oscillator` and `mfi` it consumes several price columns at once, so it takes a dataframe with `high`, `low`, and `close` columns; it computes True Range and ATR internally rather than requiring the caller to precompute them. The squeeze compares two volatility envelopes built from `close`: the Bollinger Bands are the `bb_length`-row simple moving average plus and minus `bb_mult` population standard deviations, and the Keltner Channels are the `kc_length`-row simple moving average plus and minus `kc_mult` times the `kc_length`-row average True Range. A squeeze is on (`squeeze_on`) while both Bollinger Bands sit inside the Keltner Channels, signalling low volatility and a coiled market, and off (`squeeze_off`) once they expand back outside them.

The original LazyBear script multiplied the Bollinger standard deviation by the Keltner multiplier (`kc_mult`); this port uses `bb_mult`, matching LazyBear's own September 2014 fix and the standard 2.0 Bollinger multiplier. The population standard deviation (`ddof=0`) matches the volatility module and most charting platforms. The momentum histogram is a rolling linear regression, over `kc_length` rows, of `close` detrended against the midpoint of its recent high/low range and its moving average; `squeeze_momentum` is the raw histogram in price units, and `squeeze_momentum_atr` divides it by the `atr_length`-row ATR to give a scale-invariant momentum measure comparable across tickers.

Alongside the core state and momentum, the indicator derives modeling-oriented columns not present in the original script: `squeeze_released` flags the first row after a squeeze that is no longer squeezed, `squeeze_width_ratio` is the Bollinger-band width relative to the Keltner-channel width, `squeeze_duration` counts consecutive squeezed rows, `squeeze_release_duration` records the length of the squeeze that just fired, and `squeeze_momentum_atr_change` is the row-over-row change in `squeeze_momentum_atr`. The `bb_length`, `bb_mult`, `kc_length`, `kc_mult`, and `atr_length` defaults (20, 2.0, 20, 1.5, and 14) are calibratable through the matching arguments on `lazybear_squeeze_momentum` and the `squeeze_`-prefixed arguments on `add_momentum_features`. Warm-up rows remain missing until every rolling and smoothing window is full. Inside `add_momentum_features`, the squeeze calculation receives adjustment-consistent `high`, `low`, and `close`. The absolute price-unit `squeeze_momentum` line is dropped so the persisted `squeeze_momentum_atr` feature stays comparable across tickers.

## Volatility Features

The volatility feature orchestrator is `swingtrader.data.features.volatility.add_volatility_features`. It validates the source prices once, copies them, calculates the final volatility model features from `high`, `low`, `close`, and `adjusted_close`, and appends them while preserving input row alignment.

With the default settings, the orchestrator adds:

- `adr_percent`, the Average Daily Range expressed as a percentage of the closing price;
- `atr_percent`, the Average True Range expressed as a percentage of the closing price;
- `bollinger_bandwidth`, the width between the upper and lower Bollinger bands relative to the middle band, calculated from `adjusted_close`;
- `bollinger_percent_b`, the position of `adjusted_close` within its Bollinger bands.

The public numerical volatility indicators, importable from `swingtrader.indicators`, are:

- `true_range`, which returns a series with the greatest of the current high-low range, the absolute gap between the current high and the previous close, and the absolute gap between the current low and the previous close;
- `atr`, which returns a series with Wilder's smoothed moving average of `true_range` in the input price units;
- `atr_percent`, which returns a series with `atr` divided by the current close and scaled to percentage points;
- `adr`, which returns a dataframe with an `adr` column (the simple moving average of the daily high-low range in the input price units) and an `adr_percent` column (that range as a percentage of the current close);
- `bollinger_bands`, which returns a dataframe with `bollinger_middle`, `bollinger_upper`, and `bollinger_lower` columns in the input units;
- `bollinger_bandwidth`, which returns a series with the band width relative to the middle band;
- `bollinger_percent_b`, which returns a series with the position within the bands.

The indicators split into two input shapes. `true_range`, `atr`, `atr_percent`, and `adr` consume several price columns, so each accepts a dataframe with `high`, `low`, and `close` columns. The Bollinger indicators instead operate on a single ordered series, such as `adjusted_close` or any other signal, which keeps them reusable: the momentum module applies `bollinger_percent_b` to its RSI signal to expose the `rsi_percent_b` feature. A standalone single-ticker input does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is applied independently within each provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved.

Inside `add_volatility_features`, ADR and ATR are calculated from adjustment-consistent `high`, `low`, and `close`, while the Bollinger features are calculated directly from `adjusted_close`. This keeps both intraday ranges and cross-session True Range on the same economic price history as the return, trend, and momentum families.

The default ATR length is 14 rows and is calibratable through the `atr_length` argument on `add_volatility_features` and the `length` argument on `atr` and `atr_percent`. True Range uses the previous close taken within each provider/ticker group, so the first row of each ticker falls back to its high-low range. ATR then applies Wilder's smoothing, leaving the first `length - 1` rows of each ticker missing until the window is full.

Wilder's smoothing here is the recursive exponential form seeded from the first True Range value, not the canonical definition that seeds the first ATR with the simple average of the first `length` True Ranges. The two forms converge quickly as more observations accrue, but early ATR (and `atr_percent`) values differ slightly from a canonical implementation.

The default ADR length is 20 rows and is calibratable through the `adr_length` argument on `add_volatility_features` and the `length` argument on `adr`. ADR is the simple moving average of each session's high-low range, so unlike ATR it ignores gaps between the previous close and the current session and measures typical intraday movement only. `adr_percent` divides adjustment-consistent ADR by adjusted close to give a scale-invariant measure, leaving the first `length - 1` rows of each ticker missing until the rolling window is full.

The default Bollinger length is 20 rows with 2 standard deviations, both calibratable through the `bollinger_length` and `bollinger_num_std` arguments on `add_volatility_features` and the `length` and `num_std` arguments on the Bollinger indicators. The middle band is the simple moving average, and the outer bands sit `num_std` rolling standard deviations away, leaving the first `length - 1` rows of each series missing until the window is full. The rolling standard deviation is the population standard deviation (`ddof=0`), matching John Bollinger's original definition and most charting platforms; implementations that use the sample standard deviation (`ddof=1`) produce slightly wider bands for the same `length`.

Absolute `true_range`, `atr`, `bollinger_bands`, and the price-unit `adr` column are expressed in the supplied input price units and are not comparable across tickers, so `add_volatility_features` only appends the scale-invariant `adr_percent`, `atr_percent`, `bollinger_bandwidth`, and `bollinger_percent_b` columns. `true_range`, `atr`, `adr`, and `bollinger_bands` are exposed as standalone indicators, analogous to `macd`, so consumers such as exploratory analysis and the frontend application can obtain absolute price-unit values directly. The volatility module is intended to later host additional range and dispersion measures.

## Price Action Features

The price-action feature orchestrator is `swingtrader.data.features.price_action.add_price_action_features`. It combines continuous descriptions of the current candle, short-horizon range context, local patterns, directional candle runs, and interactions with prior rolling price extremes. This preserves body, wick, close-location, gap, range, containment, engulfing, rejection, directional-persistence, breakout, and failed-break information without adding a large catalogue of thresholded textbook patterns.

With the default settings, the orchestrator adds:

- `candle_signed_body_fraction`, `(close - open) / (high - low)`, where positive values represent bullish bodies and negative values bearish bodies;
- `candle_upper_wick_fraction`, the distance from the higher of open and close to the high, divided by the high-low range;
- `candle_lower_wick_fraction`, the distance from the low to the lower of open and close, divided by the high-low range;
- `candle_close_location`, `(close - low) / (high - low)`, where 0 is the session low and 1 is the session high;
- `candle_range_atr`, the current True Range divided by the ATR available at the end of the previous row;
- `candle_gap_atr`, the signed opening gap from the previous close divided by that same prior ATR;
- `range_percentile_20`, the fraction of the preceding 20 high-low ranges that are less than or equal to the current high-low range;
- `candle_inside_bar`, true when the current high-low range is contained by the previous range;
- `candle_outside_bar`, true when the current high-low range contains the previous range;
- `candle_engulfing_strength`, the signed amount by which an opposite-direction real body exceeds the previous real body, divided by prior ATR;
- `candle_lower_rejection_strength`, the lower wick divided by prior ATR and weighted by the close's position toward the candle high;
- `candle_upper_rejection_strength`, the upper wick divided by prior ATR and weighted by the close's position toward the candle low;
- `candle_consecutive_inside_bars`, the number of consecutive inside bars ending on the current row;
- `candle_direction_run`, the signed number of consecutive candles with the same open-to-close direction, positive for bullish runs and negative for bearish runs;
- `candle_direction_run_return`, the cumulative close-to-close return from immediately before the active directional run through the current close;
- `candle_direction_run_body_atr`, the cumulative sum of each candle's signed real body divided by the ATR known before that candle;
- `candle_close_to_prior_high_atr_20` and `candle_close_to_prior_low_atr_20`, signed close distances from the preceding 20-row high and low, divided by prior ATR;
- `candle_breakout_high_strength_20` and `candle_breakout_low_strength_20`, positive close penetration beyond the corresponding prior level, divided by prior ATR;
- `candle_failed_breakout_high_strength_20` and `candle_failed_breakout_low_strength_20`, positive intraday excursions beyond a prior level when the close finishes back inside the prior range.

The public numerical candlestick indicators, importable from `swingtrader.indicators`, are:

- `candle_geometry`, which returns `signed_body_fraction`, `upper_wick_fraction`, `lower_wick_fraction`, and `close_location`;
- `candle_range_context`, which returns `range_atr`, `gap_atr`, and `range_percentile`;
- `candle_patterns`, which returns the containment, engulfing, rejection, and inside-bar streak outputs;
- `candle_direction_runs`, which returns signed run length, cumulative close-to-close return, and cumulative ATR-normalized real-body magnitude;
- `rolling_level_interactions`, which returns prior rolling high and low levels together with ATR-normalised close distance, accepted-breakout strength, and failed-break strength.

The candlestick indicators support either one chronologically ordered instrument or the canonical multi-instrument index, and calculations are isolated within each provider/ticker group. Geometry, range context, local patterns, and directional runs consume OHLC data, while rolling level interactions require high, low, and close. Zero-range candles cannot produce normalized geometry and therefore leave the four geometry outputs missing rather than producing infinities. Other outputs follow their own history and denominator requirements.

`candle_range_context` uses the ATR ending on the previous row for both current True Range and the opening gap. The current event therefore cannot increase its own denominator. Its range percentile is also point-in-time safe: it compares the current high-low range with the preceding `range_percentile_length` rows and excludes the current row from the reference sample. The feature column includes the configured history length in its name, so a length of 10 produces `range_percentile_10`.

`candle_patterns` compares each candle only with information available on that row or earlier. Inside and outside bars allow equality at one boundary, while an unchanged high-low range is neither. The first comparison row and rows with incomplete candle pairs remain missing. `candle_consecutive_inside_bars` resets to zero when the current candle is not inside the previous candle. An engulfing signal requires an opposite-direction real body that strictly exceeds and contains the previous real body; non-engulfing rows receive zero strength. Engulfing and rejection magnitudes use ATR from the previous row, so the current candle cannot inflate its own normalization denominator.

`candle_direction_runs` treats `close > open` as bullish and `close < open` as bearish. A doji resets the run and produces zero outputs, while missing open or close values remain missing and also break the sequence. Run return is measured from the close immediately before the active run, so it includes both gaps and intraday movement. The body-magnitude feature accumulates each signed real body divided by ATR from the preceding row. It remains missing when the full active run cannot be normalized, rather than silently dropping unavailable early contributions.

`rolling_level_interactions` calculates each level from only the preceding `breakout_length` rows, excluding the current candle. A breakout strength is positive when the close finishes beyond the prior high or low. A failed-break strength is positive when the intraday high or low crosses the level but the close finishes back inside; evaluable non-events are zero. Distances and strengths use ATR from the previous row, and the feature names include the configured level length.

Inside `add_price_action_features`, all four OHLC columns are placed on the `adjusted_close` scale with the row-wise factor `adjusted_close / close`. This leaves same-row geometry ratios unchanged but removes artificial cross-session gaps and True Range spikes caused by splits and dividend adjustments. The standalone indicators remain source-agnostic and operate on whichever OHLC representation the caller supplies.

The default ATR length is 14 rows, while the range-percentile and rolling-level histories both default to 20 preceding rows. Warm-up periods are kept as missing values where a calculation requires prior ATR, a previous candle, or a complete rolling history.

## Volume Features

The volume feature orchestrator is `swingtrader.data.features.volume.add_volume_features`. It validates and copies the canonical market-price dataframe, calculates `turnover_zscore`, and appends the feature while preserving the input index and row order.

Turnover is calculated as `adjusted_close * volume`. The public `turnover` indicator returns either this raw value or `log1p(turnover)` when `log=True`. Using adjusted close avoids artificial historical turnover jumps caused by corporate-action adjustments, while the logarithmic transform compresses the strong right skew commonly present in turnover.

`turnover_zscore` measures how unusual the current turnover is relative to the same instrument's recent history:

`(current turnover - prior median) / prior population standard deviation`.

The reference statistics use only the preceding `length - 1` observations; the current row is excluded. A `length` of 252 therefore compares the current row with the previous 251 trading rows. The first `length - 1` rows for each provider/ticker group remain missing, and a reference window with zero standard deviation also produces a missing value.

The feature builder defaults to `turnover_zscore_length=252` and `turnover_zscore_log=True`. Raw-turnover normalization remains available by setting `turnover_zscore_log=False`. The standalone indicator instead defaults
to `log=False`, so callers explicitly opt into the transform outside the model-feature pipeline.

The feature is point-in-time safe: no future observations are used, and the current turnover does not influence its own reference median or standard
deviation. Calculations are isolated within each provider/ticker group, so one instrument's turnover history cannot leak into another's.


## Market Structure

Market-structure indicators describe the local geometry of a price series rather than its smoothing or oscillator dynamics. They identify swing highs and swing lows and summarise the alternating swings between them.

The market-structure feature orchestrator is `swingtrader.data.features.market_structure.add_market_structure_features`. It validates the source prices once, rejects any input that already carries the generated feature names, copies the input, calculates the point-in-time Zig Zag feature block from adjustment-consistent `high`, `low`, and `close`, and appends those columns while preserving input row alignment. `swingtrader.data.features.zigzag_features` returns just the feature block for callers, such as an API endpoint, that do not need every family. When `adjusted_close` is omitted, `zigzag_features` treats the supplied `high`, `low`, and `close` as an explicitly selected price representation.

With the default settings, the orchestrator adds:

- `zigzag_last_direction`, `1` when the latest confirmed endpoint is a swing high and `-1` when it is a swing low, missing before the first endpoint is confirmed;
- `zigzag_last_swing_return`, the latest endpoint divided by the preceding endpoint minus one;
- `zigzag_last_swing_bars`, the number of observations between the latest two pivot rows;
- `zigzag_swing_return_per_bar`, the geometric mean return per observation over the latest retained swing;
- `zigzag_bars_since_pivot`, the number of observations from the latest pivot row to the current row; because the feature is emitted on confirmation, its first populated value is at least `pivot_legs // 2`;
- `zigzag_retracement`, direction-normalised movement away from the latest pivot, calculated as `-(close - last) / (last - previous)`, where zero is the latest pivot price, one is the preceding pivot price, positive values are retracements, and negative values extend the latest swing;
- `market_structure_high_change`, the logarithmic price change between the latest two confirmed swing highs;
- `market_structure_low_change`, the logarithmic price change between the latest two confirmed swing lows;
- `market_structure_high_rate`, `market_structure_high_change` divided by the number of observations between the two swing-high positions;
- `market_structure_low_rate`, `market_structure_low_change` divided by the number of observations between the two swing-low positions;
- `market_structure_high_consistency`, Kendall's tau-b between chronological order and the prices of the latest confirmed swing highs;
- `market_structure_low_consistency`, Kendall's tau-b between chronological order and the prices of the latest confirmed swing lows;
- `market_structure_leg_balance`, the median magnitude of recent completed upward legs minus the median magnitude of recent completed downward legs, divided by their sum;
- `market_structure_efficiency`, signed net log displacement divided by total absolute log path length over recent completed legs;
- `market_structure_close_to_prior_high_atr` and `market_structure_close_to_prior_low_atr`, signed close distances from the latest confirmed swing high and low, divided by prior ATR;
- `market_structure_breakout_high_strength` and `market_structure_breakout_low_strength`, positive close penetration beyond the corresponding confirmed swing level;
- `market_structure_failed_breakout_high_strength` and `market_structure_failed_breakout_low_strength`, positive intraday excursions beyond a confirmed swing level when the close finishes back on the other side.

The high and low changes preserve both direction and magnitude. Positive values mean that the corresponding structural boundary moved upward, while negative values mean that it moved downward. Taken together, their signs distinguish higher-high/higher-low, lower-high/lower-low, broadening, and contracting structures without adding a redundant categorical state. The rate features retain the same direction but also distinguish an equal structural displacement completed over a few bars from one completed over a much longer period. Drift and width are documented analytical interpretations of the high/low pair rather than additional default columns: `(high_change + low_change) / 2` describes common structural drift and `(high_change - low_change) / 2` describes expansion or contraction.

The consistency features measure whether several same-direction pivots have progressed monotonically rather than relying only on the latest pair. Kendall's tau compares every selected pair while ignoring displacement magnitude: `1` means every later pivot is higher, `-1` means every later pivot is lower, and values between those extremes represent mixed ordering. The tau-b denominator adjusts for equal pivot prices. Consistency is missing when fewer than the configured number of same-direction pivots exist or when all selected prices are equal. The default is the latest four swing highs and four swing lows, calibratable through `zigzag_consistency_pivots` on `add_market_structure_features` and `consistency_pivots` on `zigzag_features`.

The completed-movement features use logarithmic changes between adjacent retained pivots. `market_structure_leg_balance` compares the median absolute magnitude of low-to-high legs with the median absolute magnitude of high-to-low legs using `(up_median - down_median) / (up_median + down_median)`. Positive values mean upward impulses are typically larger, negative values mean downward legs dominate, and zero means their typical magnitudes are equal. `market_structure_efficiency` divides signed net log displacement by the sum of absolute log-leg magnitudes. Values near `1` or `-1` indicate efficient directional progression, while values near zero indicate substantial structural movement with little net progress. Both features use the same Zig Zag scale as the displacement and consistency features.

The dynamics window defaults to the latest six completed legs, giving three upward and three downward observations. It is calibratable through `zigzag_dynamics_legs` on `add_market_structure_features` and `dynamics_legs` on `zigzag_features`, and must be an even integer of at least two. Both outputs remain missing until the complete window exists. Only movement between confirmed retained pivots is included: price movement after the latest confirmed endpoint, including the current interpolated or active leg, is excluded.

The swing-level interaction features reuse the latest high and low that were confirmed and knowable on each row. Accepted and failed breaks follow the same definitions as the rolling price-action levels and are normalized by ATR from the previous row. They remain missing until the corresponding swing level and prior ATR exist; evaluable rows without a break receive zero strength. The ATR length defaults to 14 and is configurable through `zigzag_atr_length` on `add_market_structure_features` and `atr_length` on `zigzag_features`.

Unlike the retrospective `zigzag` indicator, these features are point-in-time: a pivot updates the output only on and after its confirmation row, and an intermediate endpoint stays visible until a later, more extreme, confirmed endpoint replaces it. Temporary confirmed endpoints can therefore affect the state that was knowable at the time even if they later disappear from the final retrospective Zig Zag. Appending future rows never changes an already-emitted value. The structural changes and rates remain missing until two confirmed pivots of the corresponding direction exist. Their bar counts use the historical pivot positions rather than the later confirmation rows. The `zigzag_deviation` and `zigzag_pivot_legs` arguments default to 5.0 percent and 10 bars and are forwarded to the underlying Zig Zag calculation.

!!! warning "Do not use retrospective Zig Zag outputs as model features"

    The columns returned by the public `zigzag()` indicator — `zigzag_price`, `zigzag_direction`, `zigzag_return`, and `zigzag_bars` — are retrospective analytical  outputs. Final pivots are written on their historical extreme rows only after later observations confirm and potentially revise the Zig Zag sequence. These columns therefore contain future information relative to their row dates and must not be used as row-aligned machine-learning predictors.

    Use the columns produced by `zigzag_features()` or `add_market_structure_features()` for model training. Those outputs are point-in-time safe: confirmed pivot information first appears on the confirmation row, and appending future observations does not change previously emitted values.

The public numerical market-structure indicators, importable from `swingtrader.indicators`, are:

- `pivot_points_high_low`, which consumes a dataframe with `high` and `low` columns (or, when `kind="balanced"`, `open`, `high`, `low`, and `close`) and returns a dataframe of pivot flags together with either ordinal ranks or normalised strengths;
- `zigzag`, which consumes a dataframe with `high` and `low` columns and returns the retained alternating swing highs and lows with their signed returns and bar counts.

Like the other multi-column indicators they accept either one ordered single-instrument input or a canonical multi-instrument input carrying the `provider`, `ticker`, and `trading_date` index levels. A standalone single-ticker input does not require the three-level MultiIndex; it only has to be chronologically ordered. When the canonical index levels are present the calculation is isolated per provider/ticker group, so one ticker's history cannot leak into another's, and the original index and row order are preserved.

### Pivot Points

A row is a **pivot high** (swing high) when its selected high value is the most extreme within a window made up of the candidate row, the preceding `high_left` rows, and the following `high_right` rows; a row is a **pivot low** (swing low) when its selected low value is the most extreme within the corresponding `low_left`/`low_right` window. Equal extreme values share the best rank and are therefore all marked as pivots. The default window is 10 rows on each side, calibratable through the `high_left`, `high_right`, `low_left`, and `low_right` arguments, which must each be a positive integer.

`kind` selects the price representation used for ranking. `"high_low"`, the default, ranks the raw `high` and `low` values. `"balanced"` pulls each extreme toward the candle body using `balanced_high = (2 * high + max(open, close)) / 3` and `balanced_low = (2 * low + min(open, close)) / 3`, so a long wick counts for less than a decisive close near the extreme.

`rank_output` selects what is returned alongside the nullable Boolean `pivot_high` and `pivot_low` flags. `"rank"`, the default, returns `pivot_high_rank` and `pivot_low_rank`, ordinal ranks beginning at one, where a pivot is any row of rank one. `"strength"` returns `pivot_high_strength` and `pivot_low_strength` instead, rescaling those ranks to the interval from zero to one, where one is the strongest possible pivot candidate. Rows without a complete surrounding window remain missing.

Because each pivot is evaluated from observations on both sides of the candidate row, the outputs are aligned with the candidate row but are only knowable `high_right` (or `low_right`) rows later. `pivot_points_high_low` is therefore a lookahead-aware standalone indicator: its outputs must be shifted to their confirmation rows before being used as point-in-time model features. It is exposed for exploratory analysis and future API or frontend charting rather than included in a feature orchestrator.

### Zig Zag

`zigzag` filters confirmed local extrema into an alternating sequence of swing highs and lows that each meet a minimum percentage reversal. It first detects confirmed pivot candidates from the raw `high` and `low`: `pivot_legs` is the total confirmation width, matching the TradingView input, and is divided by two with floor division to require that many observations on both sides of each candidate. A pivot high must be strictly greater than every value to its left and greater than or equal to every value to its right, with the inverse comparisons for a pivot low, so the first value in a run of equal extrema is retained.

Candidate pivots are processed chronologically. An opposite-direction pivot is retained only when its reversal from the last retained pivot is at least `deviation` percent, while a same-direction candidate replaces the current endpoint when it is more extreme, so the retained sequence always alternates between highs and lows. The result contains `zigzag_price` (the raw high or low of each retained pivot), `zigzag_direction` (`1` for a high, `-1` for a low, `0` elsewhere), `zigzag_return` (`current_price / previous_price - 1` on retained pivots), and `zigzag_bars` (observations between consecutive retained pivots). `deviation` and `pivot_legs` default to 5.0 percent and 10 bars; `deviation` must be a non-negative finite number and `pivot_legs` an integer of at least two.

Like `pivot_points_high_low`, `zigzag` is retrospective and lookahead-aware: each pivot is first knowable `pivot_legs // 2` observations later, and the latest retained endpoint can still be replaced by a later, more extreme same-direction pivot. The point-in-time `add_market_structure_features` block above wraps the identical pivot logic but delays every update to its confirmation row, so it can be used as a leakage-free model feature while `zigzag` itself remains a charting- and analysis-oriented indicator.

## Default Feature Pipeline

`swingtrader.data.features.pipeline.add_default_features` runs the standard feature families in a fixed order: returns, then trend, then momentum, then volatility, then price action, then volume, then market structure. Each family receives the dataframe produced by the previous step, so the result is identical to calling `add_return_features`, `add_trend_features`, `add_momentum_features`, `add_volatility_features`, `add_price_action_features`, `add_volume_features`, and `add_market_structure_features` in that sequence with their default arguments. It provides a single entry point for producing the full default feature set while leaving the individual builders available for callers that need custom arguments or a subset of families.

## Future Feature Ideas

- later macro and market-context joins.

## Design Constraints

- Feature code reads from bronze data, not directly from yfinance.
- Feature inputs must use a unique, sorted `provider`/`ticker`/`trading_date` MultiIndex.
- Cross-session model price features must use the adjustment-consistent OHLC contract.
- Warmup periods should be represented explicitly.
- Features should avoid point-in-time leakage.
- Labels should be generated separately from input features.
- If feature persistence is introduced, it should support later train, validation, and test splits.