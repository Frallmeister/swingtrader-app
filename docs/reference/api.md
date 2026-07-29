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

## Interactive Entry Labeling

::: swingtrader.modeling.labeling

## Modeling Targets

### Contracts

::: swingtrader.modeling.datasets.contracts

### Catalog

::: swingtrader.modeling.datasets.catalog

### Target Builders

::: swingtrader.modeling.datasets.labels

### ATR Barrier Targets

::: swingtrader.modeling.datasets.barriers

## Temporal Modeling Datasets

### Specifications

::: swingtrader.modeling.datasets.specifications

### Construction and Bundle Contracts

::: swingtrader.modeling.datasets.temporal

### Tabular Adapter

::: swingtrader.modeling.datasets.tabular

## Modeling Experiments

### Contracts

::: swingtrader.modeling.experiments.contracts

### Purged Temporal Splitting

::: swingtrader.modeling.experiments.splitting

### Train-Only Temporal Cross-Validation

::: swingtrader.modeling.experiments.cross_validation

### MLflow Tracking

::: swingtrader.modeling.experiments.tracking

## Baseline Training and Evaluation

### Prediction and Evaluation Contracts

::: swingtrader.modeling.training.contracts

### Baseline Models

::: swingtrader.modeling.training.baselines

### Evaluation

::: swingtrader.modeling.training.evaluation

### Reporting

::: swingtrader.modeling.training.reporting

### Reusable Harness

::: swingtrader.modeling.training.harness

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