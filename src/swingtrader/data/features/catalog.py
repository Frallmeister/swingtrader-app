"""Concrete, versioned feature-set definitions.

This module is the catalog of reproducible feature sets built from the
contract types in :mod:`swingtrader.data.features.feature_sets`. It holds
configured instances such as :data:`DEFAULT_FEATURE_SET`; add new named and
versioned sets here rather than in the module that defines the spec types.
"""

from __future__ import annotations

from swingtrader.data.features.feature_sets import (
    FeatureBlockSpec,
    FeatureSetSpec,
    HistoryRequirement,
)
from swingtrader.data.features.market_structure import add_market_structure_features
from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.data.features.volume import add_volume_features

DEFAULT_FEATURE_SET = FeatureSetSpec(
    name="ohlcv_v1_candidates",
    version="1",
    blocks=(
        FeatureBlockSpec(
            name="returns",
            builder=add_return_features,
            parameters={"horizons": (1, 5, 10, 20)},
            output_columns=(
                "return_1d",
                "return_5d",
                "return_10d",
                "return_20d",
            ),
            required_columns=frozenset({"adjusted_close"}),
        ),
        FeatureBlockSpec(
            name="trend",
            builder=add_trend_features,
            parameters={
                "ma_lengths": (10, 20, 50),
                "adx_length": 14,
                "vwap_length": 20,
                "vwap_bollinger_length": 20,
                "vwap_bollinger_num_std": 2.0,
            },
            output_columns=(
                "ema_fast_to_ema_mid",
                "ema_mid_to_ema_slow",
                "ema_mid_to_sma_mid",
                "close_to_ema_fast",
                "close_to_ema_mid",
                "close_to_ema_slow",
                "adx",
                "plus_di",
                "minus_di",
                "vwap_distance",
                "vwap_distance_percent_b",
            ),
            required_columns=frozenset({"high", "low", "close", "volume", "adjusted_close"}),
            history_requirement=HistoryRequirement.EXPANDING,
        ),
        FeatureBlockSpec(
            name="momentum",
            builder=add_momentum_features,
            parameters={
                "ppo_lengths": (12, 26, 9),
                "ppo_percentile_min_history": 100,
                "rsi_length": 21,
                "rsi_bollinger_length": 20,
                "rsi_bollinger_num_std": 2.0,
                "stochastic_k_length": 14,
                "stochastic_k_smoothing": 3,
                "stochastic_d_length": 3,
                "mfi_length": 14,
                "mfi_bollinger_length": 20,
                "mfi_bollinger_num_std": 2.0,
                "squeeze_bb_length": 20,
                "squeeze_bb_mult": 2.0,
                "squeeze_kc_length": 20,
                "squeeze_kc_mult": 1.5,
                "squeeze_atr_length": 14,
            },
            output_columns=(
                "ppo",
                "ppo_signal",
                "ppo_histogram",
                "ppo_percentile",
                "rsi",
                "rsi_percent_b",
                "stochastic_k",
                "stochastic_d",
                "mfi",
                "mfi_percent_b",
                "squeeze_on",
                "squeeze_off",
                "squeeze_released",
                "squeeze_width_ratio",
                "squeeze_momentum_atr",
                "squeeze_momentum_atr_change",
                "squeeze_duration",
                "squeeze_release_duration",
            ),
            required_columns=frozenset({"high", "low", "close", "adjusted_close", "volume"}),
            history_requirement=HistoryRequirement.EXPANDING,
        ),
        FeatureBlockSpec(
            name="volatility",
            builder=add_volatility_features,
            parameters={
                "adr_length": 20,
                "atr_length": 14,
                "bollinger_length": 20,
                "bollinger_num_std": 2.0,
            },
            output_columns=(
                "adr_percent",
                "atr_percent",
                "bollinger_bandwidth",
                "bollinger_percent_b",
            ),
            required_columns=frozenset({"high", "low", "close", "adjusted_close"}),
            history_requirement=HistoryRequirement.EXPANDING,
        ),
        FeatureBlockSpec(
            name="price_action",
            builder=add_price_action_features,
            parameters={
                "atr_length": 14,
                "range_percentile_length": 20,
                "breakout_length": 20,
            },
            output_columns=(
                "candle_signed_body_fraction",
                "candle_upper_wick_fraction",
                "candle_lower_wick_fraction",
                "candle_close_location",
                "candle_range_atr",
                "candle_gap_atr",
                "range_percentile_20",
                "candle_inside_bar",
                "candle_outside_bar",
                "candle_engulfing_strength",
                "candle_lower_rejection_strength",
                "candle_upper_rejection_strength",
                "candle_consecutive_inside_bars",
                "candle_direction_run",
                "candle_direction_run_return",
                "candle_direction_run_body_atr",
                "candle_close_to_prior_high_atr_20",
                "candle_close_to_prior_low_atr_20",
                "candle_breakout_high_strength_20",
                "candle_breakout_low_strength_20",
                "candle_failed_breakout_high_strength_20",
                "candle_failed_breakout_low_strength_20",
            ),
            required_columns=frozenset({"open", "high", "low", "close", "adjusted_close"}),
            history_requirement=HistoryRequirement.EXPANDING,
        ),
        FeatureBlockSpec(
            name="volume",
            builder=add_volume_features,
            parameters={
                "turnover_zscore_length": 252,
                "turnover_zscore_log": True,
            },
            output_columns=("turnover_zscore",),
            required_columns=frozenset({"close", "volume"}),
        ),
        FeatureBlockSpec(
            name="market_structure",
            builder=add_market_structure_features,
            parameters={
                "zigzag_deviation": 5.0,
                "zigzag_pivot_legs": 10,
                "zigzag_consistency_pivots": 4,
                "zigzag_dynamics_legs": 6,
                "zigzag_atr_length": 14,
            },
            output_columns=(
                "zigzag_last_direction",
                "zigzag_last_swing_return",
                "zigzag_last_swing_bars",
                "zigzag_swing_return_per_bar",
                "zigzag_bars_since_pivot",
                "zigzag_retracement",
                "market_structure_high_change",
                "market_structure_low_change",
                "market_structure_high_rate",
                "market_structure_low_rate",
                "market_structure_high_consistency",
                "market_structure_low_consistency",
                "market_structure_leg_balance",
                "market_structure_efficiency",
                "market_structure_close_to_prior_high_atr",
                "market_structure_close_to_prior_low_atr",
                "market_structure_breakout_high_strength",
                "market_structure_breakout_low_strength",
                "market_structure_failed_breakout_high_strength",
                "market_structure_failed_breakout_low_strength",
            ),
            required_columns=frozenset({"high", "low", "close"}),
            history_requirement=HistoryRequirement.PATH_DEPENDENT,
        ),
    ),
)
