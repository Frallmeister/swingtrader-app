# API Reference

This page documents selected implemented modules. Planned modules are intentionally omitted until code exists.

## Core Database

::: swingtrader.core.db

## Data Database

::: swingtrader.data.db

## Core Configuration

::: swingtrader.core.config

## Core Logging

::: swingtrader.core.logging_config

## Core Numerical Helpers

::: swingtrader.core.numerical

## Market Data Ingestion

::: swingtrader.data.ingestion.market_data

## Market Data Settings

::: swingtrader.data.ingestion.market_data_settings

## Daily Market Data Job

::: swingtrader.data.jobs.update_market_data

## Bronze Onboarding

::: swingtrader.data.ingestion.onboarding

## Ticker Eligibility

::: swingtrader.data.eligibility

## Market-Frame Contract

::: swingtrader.data.market_frame

## Indicators

Reusable technical indicators. Indicators calculate reusable technical
quantities; features transform raw data and indicators into model inputs. Each
public indicator supports two input forms: a single ordered instrument, or a
canonical multi-instrument market frame indexed by `provider`, `ticker`, and
`trading_date`. Standalone indicators are intended to be reusable by future
API endpoints, frontend charting and screening, backtests, and trade-record
analysis.

### Moving Averages

::: swingtrader.indicators.moving_averages

### Directional Movement

::: swingtrader.indicators.directional_movement

### Candlestick Indicators

::: swingtrader.indicators.candlesticks

### Volatility Indicators

::: swingtrader.indicators.volatility

### MACD and PPO

::: swingtrader.indicators.macd

### Oscillators

::: swingtrader.indicators.oscillators

### Volume Indicators

::: swingtrader.indicators.volume

### Squeeze Momentum

::: swingtrader.indicators.squeeze_momentum

### Market Structure

::: swingtrader.indicators.market_structure

## Return Features

::: swingtrader.data.features.returns

## Trend Features

::: swingtrader.data.features.trend

## Momentum Features

::: swingtrader.data.features.momentum

## Volatility Features

::: swingtrader.data.features.volatility

## Price Action Features

::: swingtrader.data.features.price_action

## Volume Features

::: swingtrader.data.features.volume

## Market Structure Features

::: swingtrader.data.features.market_structure

## Default Feature Pipeline

::: swingtrader.data.features.pipeline

## Versioned Feature Sets

### Contract Types

::: swingtrader.data.features.contracts

### Catalog

::: swingtrader.data.features.catalog

## Modeling Dataset Labels

::: swingtrader.modeling.datasets.labels

## Universe Selection

::: swingtrader.data.ingestion.universe_selection

## Bronze Writer

::: swingtrader.data.bronze.writer

## Bronze Loaders

::: swingtrader.data.bronze.loaders

## Bronze Queries

::: swingtrader.data.bronze.queries

## Yfinance Client

::: swingtrader.data.clients.yfinance