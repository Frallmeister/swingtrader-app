# Indicator Screening

The first replay screener uses indicators rather than model features. A screen is composed from continuous operands and simple arithmetic.

## Operands

An operand is either a raw market column or one output of a configured indicator. Indicator parameters are the public parameters exposed by the Python function signature. For a multi-output indicator, one calculation produces all outputs; the screen selects the particular output it needs.

Examples:

```text
close
EMA(source=close, length=10).ema
MACD(source=close, lengths=(10, 20, 9)).macd_histogram
Bollinger Bands(source=close, length=20).bollinger_upper
```

## Expressions

Version one supports an identity value or two operands combined with division, subtraction, addition, or multiplication. This keeps comparisons continuous:

```text
close / EMA(10)
EMA(10) / EMA(20)
ATR(14) / close
close / Donchian upper(20)
```

Each expression can use the latest, maximum, minimum, or mean value over a configurable lookback of up to 252 sessions. Rules support greater-than, greater-than-or-equal, less-than, less-than-or-equal, equality, and between comparisons.

## Saved presets

A preset stores the complete screen configuration: indicator parameters, selected outputs, arithmetic, lookbacks, thresholds, sort order, and owned/pending-stock exclusions. Each executed screen also stores a configuration snapshot and its resulting ticker list so later preset edits do not rewrite replay history.
