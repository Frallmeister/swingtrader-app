# Indicators

## Responsibility

The indicators area holds reusable technical-indicator calculations such as moving averages, directional movement, volatility measures, MACD and PPO, oscillators, volume indicators, and squeeze momentum. Indicators calculate reusable technical quantities; features transform raw data and indicators into model inputs. Because indicators know nothing about the model feature set, they can be reused by feature builders, notebooks, tests, and future API and frontend charting functionality.

## Design principles

Each public indicator supports two input forms: a single ordered instrument, which only has to be chronologically ordered, or a canonical multi-instrument market frame with a unique, sorted `MultiIndex` of levels `provider`, `ticker`, and `trading_date`, in that exact order. When the canonical index is present, calculations are applied independently within each provider/ticker group so one ticker's history cannot leak into another's, and the input index and row order are preserved.

Indicators return either one index-aligned `pd.Series` or, for naturally multi-output indicators, one index-aligned `pd.DataFrame`. They must not fill warm-up periods silently; incomplete windows remain missing until the relevant rolling and smoothing windows are full.

## Package boundaries

Indicators must not import from `swingtrader.data.features` or any other higher-level model, storage, or presentation code. Reuse between indicator modules is allowed. Shared, domain-neutral numerical helpers belong in `swingtrader.core.numerical`, and the canonical market-frame contract helpers belong in `swingtrader.data.market_frame`.

## Further documentation

See [Features](../../../docs/data/features.md) for the indicator and feature design notes, and [API reference](../../../docs/reference/api.md) for the generated indicator API.
