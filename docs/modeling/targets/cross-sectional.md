# Cross-Sectional Return Targets

The `cross_sectional_return_targets:1` target set compares each stock's future adjusted-close return with the other valid stocks in the same provider and prediction date. It is an independent target variant: it is generated on its own from canonical prices, derives the forward returns it needs internally, and does not include forward-return classification or triple-barrier outputs. It defines a distinct learning objective and does not supersede the other target variants.

## Purpose

These targets support stock-selection research where the main question is relative rather than absolute:

> Which stocks are likely to outperform the other stocks available on the same date?

They are an alternative to absolute forward returns and path-dependent triple-barrier labels, not a replacement. A stock can rank highly during a falling market and still have a negative return.

## Cross-Section Definition

Each cross-section contains rows sharing:

- `provider`;
- `trading_date`.

Missing or non-finite forward returns are excluded from the benchmark and ranking for that horizon. Their derived cross-sectional targets remain missing. This target set requires at least 20 valid stocks per cross-section; smaller cross-sections remain missing.

The forward-return endpoint must also match the provider's shared trading-date calendar. A stock that skipped the expected endpoint session is excluded from that horizon's comparison rather than being ranked over a different calendar window.

The canonical index currently has no separate market or universe identifier. The provider/date grouping therefore assumes that the supplied modeling frame represents one comparable universe. Stockholm and US securities should not be mixed in the same provider-scoped cross-section until a market or universe identifier is available.

## Market-Relative Forward Return

For stock \(i\), date \(t\), and horizon \(h\), let \(r_{i,t,h}\) be the adjusted-close forward return over horizon \(h\). The builder derives these forward returns internally from `adjusted_close`, so it accepts the same canonical price frame as the other target builders. The equal-weight market return is:

\[
\bar r_{t,h}=\frac{1}{N_{t,h}}\sum_i r_{i,t,h}.
\]

The market-relative target is the gross-return ratio minus one:

\[
y^{\mathrm{relative}}_{i,t,h}
=\frac{1+r_{i,t,h}}{1+\bar r_{t,h}}-1.
\]

The output column is:

```text
market_relative_forward_return_{horizon}d
```

A value of zero means the stock matched the equal-weight cross-section. Positive values indicate outperformance and negative values indicate underperformance.

## Future-Return Percentile

Valid returns are ranked within the same provider/date cross-section using average ranks for ties. With rank \(q_i\) and valid count \(N\), the midpoint percentile is:

\[
p_i=\frac{q_i-0.5}{N}.
\]

The output column is:

```text
forward_return_{horizon}d_cross_sectional_percentile
```

The midpoint convention avoids exact zero or one values. Tied returns receive the same percentile.

## Ordinal Relevance Grade

This target set maps each percentile to 16 ordered grades:

\[
g_i=\min\left(15,\left\lfloor 16p_i\right\rfloor\right).
\]

The output column is:

```text
forward_return_{horizon}d_relevance_grade
```

The nullable `Int8` values range from `0` for the weakest future-return region to `15` for the strongest. These grades are intended as convenient future relevance labels for later ranking experiments; this target set itself does not introduce a ranking model or ranking-specific task contract.

## Horizons and Primary Task

The configured horizons are 5, 10, and 15 observed sessions. The primary supervised task is regression on:

```text
forward_return_5d_cross_sectional_percentile
```

This keeps the current `classification`/`regression` task contract intact. A future learning-to-rank implementation can consume the relevance-grade columns while grouping rows by prediction date.

## Temporal Semantics

All cross-sectional targets depend on future prices and must never be used as features. Their maximum horizon remains 15 sessions. The five-session primary task resolves at the same fixed future session as `forward_return_5d`, so the existing temporal dataset and purging logic continue to apply.

## Limitations

- The benchmark is equal weighted and includes the stock itself.
- Cross-sectional composition depends on which valid rows are present for the date and horizon.
- Percentiles discard the magnitude of return differences.
- Relevance grades discretize the percentile further.
- A high relative grade does not imply a positive absolute return or a tradable setup.
