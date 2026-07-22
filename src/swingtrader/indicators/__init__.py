"""Reusable technical indicators.

This package calculates reusable technical quantities such as moving averages,
directional movement, candlestick geometry, volatility bands, oscillators, and
squeeze momentum.
Indicators are independent of model-specific feature naming and interpretation
and are intended to be reused by feature builders, notebooks, tests, and future
API or frontend charting code.

Every public indicator supports two input forms:

- a single ordered instrument (a ``Series`` or ``DataFrame`` that only has to be
  chronologically ordered);
- a canonical multi-instrument market frame with a ``MultiIndex`` of
  ``provider``, ``ticker``, and ``trading_date``, in which case calculations are
  isolated per ``provider`` and ``ticker`` and the input index and row order are
  preserved.

Indicators return new index-aligned outputs and never mutate their input objects.
"""

from swingtrader.indicators.candlesticks import candle_geometry, candle_range_context
from swingtrader.indicators.directional_movement import adx
from swingtrader.indicators.macd import macd, ppo
from swingtrader.indicators.market_structure import pivot_points_high_low, zigzag
from swingtrader.indicators.moving_averages import ema, rolling_vwap, sma
from swingtrader.indicators.oscillators import rsi, stochastic_oscillator
from swingtrader.indicators.squeeze_momentum import lazybear_squeeze_momentum
from swingtrader.indicators.volatility import (
    adr,
    atr,
    atr_percent,
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_percent_b,
    true_range,
)
from swingtrader.indicators.volume import mfi, turnover, turnover_zscore

__all__ = [
    "adr",
    "adx",
    "atr",
    "atr_percent",
    "bollinger_bands",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "candle_geometry",
    "candle_range_context",
    "ema",
    "lazybear_squeeze_momentum",
    "macd",
    "mfi",
    "pivot_points_high_low",
    "ppo",
    "rolling_vwap",
    "rsi",
    "sma",
    "stochastic_oscillator",
    "true_range",
    "turnover",
    "turnover_zscore",
    "zigzag",
]
