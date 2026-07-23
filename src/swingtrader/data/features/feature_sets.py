"""Versioned contracts for reproducible model feature sets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import pandas as pd

from swingtrader.data.features.market_structure import add_market_structure_features
from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.data.features.volume import add_volume_features

type FeatureParameter = bool | int | float | str | tuple[object, ...]
type FeatureBuilder = Callable[..., pd.DataFrame]


class HistoryRequirement(StrEnum):
    """Describe how much historical state a feature block may depend on."""

    BOUNDED = "bounded"
    EXPANDING = "expanding"
    PATH_DEPENDENT = "path_dependent"


@dataclass(frozen=True, slots=True)
class FeatureBlockSpec:
    """Declare one executable feature-family block and its stable schema."""

    name: str
    builder: FeatureBuilder = field(repr=False, compare=False)
    parameters: Mapping[str, FeatureParameter] = field(default_factory=dict)
    output_columns: tuple[str, ...] = ()
    required_columns: frozenset[str] = frozenset()
    history_requirement: HistoryRequirement = HistoryRequirement.BOUNDED

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature block name must not be empty.")

        output_columns = tuple(self.output_columns)
        object.__setattr__(self, "output_columns", output_columns)

        if not output_columns:
            raise ValueError(f"Feature block {self.name!r} must declare output columns.")
        if len(output_columns) != len(set(output_columns)):
            raise ValueError(f"Feature block {self.name!r} contains duplicate output columns.")

        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "required_columns", frozenset(self.required_columns))

    @property
    def builder_path(self) -> str:
        """Return the import path of the configured builder."""
        return f"{self.builder.__module__}.{self.builder.__qualname__}"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply this block with its declared parameters."""
        return self.builder(data, **self.parameters)

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable block description."""
        return {
            "name": self.name,
            "builder": self.builder_path,
            "parameters": {
                key: _json_value(value) for key, value in sorted(self.parameters.items())
            },
            "output_columns": list(self.output_columns),
            "required_columns": sorted(self.required_columns),
            "history_requirement": self.history_requirement.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    """Declare an ordered, versioned collection of feature blocks."""

    name: str
    version: str
    blocks: tuple[FeatureBlockSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature set name must not be empty.")
        if not self.version:
            raise ValueError("Feature set version must not be empty.")
        blocks = tuple(self.blocks)
        object.__setattr__(self, "blocks", blocks)

        if not blocks:
            raise ValueError("A feature set must contain at least one block.")

        block_names = tuple(block.name for block in blocks)

        if len(block_names) != len(set(block_names)):
            raise ValueError("Feature block names must be unique within a feature set.")

        output_columns = self.feature_columns
        if len(output_columns) != len(set(output_columns)):
            raise ValueError("Feature output columns must be unique across a feature set.")

    @property
    def identifier(self) -> str:
        """Return the stable feature-set name and version identifier."""
        return f"{self.name}:{self.version}"

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return all declared feature columns in execution order."""
        return tuple(column for block in self.blocks for column in block.output_columns)

    @property
    def required_columns(self) -> frozenset[str]:
        """Return the union of source columns required by all blocks."""
        return frozenset(column for block in self.blocks for column in block.required_columns)

    def select(
        self,
        *block_names: str,
        name: str,
        version: str,
    ) -> FeatureSetSpec:
        """Return a newly identified subset in the original block order."""
        requested = set(block_names)
        if not requested:
            raise ValueError("At least one feature block name is required.")

        available = {block.name for block in self.blocks}
        unknown = requested.difference(available)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown feature block names: {names}.")

        return FeatureSetSpec(
            name=name,
            version=version,
            blocks=tuple(block for block in self.blocks if block.name in requested),
        )

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable feature-set manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "feature_columns": list(self.feature_columns),
            "required_columns": sorted(self.required_columns),
            "blocks": [block.to_manifest() for block in self.blocks],
        }


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


def _json_value(value: FeatureParameter) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
