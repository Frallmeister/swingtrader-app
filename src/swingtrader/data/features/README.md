# Features

## Responsibility

The features area is reserved for reusable transformations that convert point-in-time input data into model-ready explanatory variables. Feature code should make historical market data useful for modeling without changing the meaning of the source observations or mixing in target, model, or presentation concerns.

Feature generation currently includes in-memory return features. The current directory should not yet be treated as a persistent feature pipeline.

## Design principles

Feature code should operate only on data available at or before each observation timestamp. It must preserve ticker and trading-date alignment, avoid lookahead leakage, and avoid cross-ticker contamination unless a future feature explicitly defines a safe market-level aggregate.

Current feature inputs must provide `provider`, `ticker`, and `trading_date` consistently as columns or named index levels. Observations are expected to be strictly ordered by `trading_date` within each provider/ticker group.

Rolling-window features should handle warm-up periods explicitly instead of silently filling incomplete history. Outputs should be predictable, testable, and suitable for both exploratory analysis and later training workflows. Reusable feature logic belongs in package modules with tests, not only in notebooks.

Likely future feature categories include returns and momentum, trend, volatility and range, volume, technical indicators, and candlestick geometry. Those categories are directional examples, not implemented functions or committed formulas.

## Current status

Initial feature work calculates features in memory for EDA and baseline modeling. Persistent feature tables, feature versioning, and feature-store-like infrastructure are future design decisions. They are not required by the current package contract and are not implemented.

## Package boundaries

Feature code should read source-oriented data from bronze or other approved point-in-time inputs. It should remain separate from provider download code, bronze persistence, target and label generation, model training, production inference, prediction storage, and dashboard presentation.

Experimental ideas, formula sketches, and external references should live in GitHub issues, research notebooks, or dedicated research documentation rather than in this package README.

## Further documentation

See [Features](../../../../docs/data/features.md) for the project-level feature design notes.
