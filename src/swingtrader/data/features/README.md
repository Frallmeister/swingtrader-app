# Features

## Responsibility

The features area contains reusable transformations that convert point-in-time input data into model-ready explanatory variables. Feature code should make historical market data useful for modeling, screening, APIs, backtests, and later trade-record analysis without changing the meaning of source observations or mixing in target, model, storage, or presentation concerns.

Reusable technical calculations live in `swingtrader.indicators`. Indicators calculate domain quantities; feature builders decide which source columns to use, how quantities are adjusted or normalized, and what the model-facing columns are named. Feature families must import indicator calculations from `swingtrader.indicators` rather than reimplementing them or importing calculations from sibling feature families.

Feature generation currently includes seven in-memory families: returns, trend, momentum, volatility, price action, volume, and market structure.

`contracts.py` defines the immutable feature-set contract types, `catalog.py` contains concrete named and versioned feature-set definitions, and `pipeline.py` executes a supplied feature-set specification in its declared block order. This directory should not yet be treated as a persistent feature pipeline.

## Design principles

Feature code must operate only on information available at or before each observation timestamp. It preserves ticker and trading-date alignment, avoids lookahead leakage and cross-ticker contamination, and leaves incomplete rolling windows missing rather than silently filling them.

Inputs use a unique, sorted `MultiIndex` with levels `provider`, `ticker`, and `trading_date`, in that exact order, plus the value columns consumed by the family. Identifiers must not also appear as ordinary columns. External consumers that need column-oriented records convert explicitly with `features.reset_index()` at their own boundary.

Outputs should remain predictable, testable, and reusable. The same feature dataframe may later support model training, stock-screen filters, API responses, backtest diagnostics, and statistical analysis of recorded trades; those consumers should not require feature formulas to be duplicated in application-specific code.

## Current status

Features are currently calculated in memory for EDA and baseline modeling. Versioned feature-set contracts are implemented, while persistent feature tables and feature-store-like infrastructure remain future design decisions.

## Package boundaries

Feature code reads source-oriented data from bronze or other approved point-in-time inputs. It remains separate from provider downloads, bronze persistence, target and label generation, model training, production inference, prediction storage, and dashboard presentation.

Experimental formulas and external references belong in GitHub issues, research notebooks, or dedicated research documentation rather than this package README.

## Further documentation

See [Features](../../../../docs/data/features.md) for the project-level feature design notes.
