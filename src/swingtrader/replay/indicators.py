"""Indicator catalogue and on-demand calculation for charts and screening."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args, get_origin, get_type_hints

import pandas as pd

import swingtrader.indicators as public_indicators


@dataclass(frozen=True)
class IndicatorParameter:
    name: str
    label: str
    kind: str
    default: Any
    required: bool
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class IndicatorOutput:
    id: str
    label: str
    chart_style: Literal["line", "histogram", "marker"]
    pane: Literal["price", "separate"]


@dataclass(frozen=True)
class IndicatorDefinition:
    id: str
    label: str
    input_kind: Literal["series", "frame"]
    default_source: str | None
    parameters: tuple[IndicatorParameter, ...]
    outputs: tuple[IndicatorOutput, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SERIES_INPUTS = {
    "sma",
    "ema",
    "rolling_fraction_above_ema",
    "rsi",
    "macd",
    "ppo",
    "bollinger_bands",
    "bollinger_bandwidth",
    "bollinger_percent_b",
}

REQUIRED_DEFAULTS: dict[tuple[str, str], Any] = {
    ("sma", "length"): 20,
    ("ema", "length"): 20,
    ("rolling_vwap", "length"): 20,
}

OUTPUT_IDS: dict[str, tuple[str, ...]] = {
    "adr": ("adr", "adr_percent"),
    "adx": ("adx", "plus_di", "minus_di"),
    "atr": ("atr",),
    "atr_percent": ("atr_percent",),
    "bollinger_bands": ("bollinger_middle", "bollinger_upper", "bollinger_lower"),
    "bollinger_bandwidth": ("bollinger_bandwidth",),
    "bollinger_percent_b": ("bollinger_percent_b",),
    "candle_direction_runs": (
        "direction_run",
        "direction_run_return",
        "direction_run_body_atr",
    ),
    "candle_geometry": (
        "signed_body_fraction",
        "upper_wick_fraction",
        "lower_wick_fraction",
        "close_location",
    ),
    "candle_patterns": (
        "inside_bar",
        "outside_bar",
        "engulfing_strength",
        "lower_rejection_strength",
        "upper_rejection_strength",
        "consecutive_inside_bars",
    ),
    "candle_range_context": ("range_atr", "gap_atr", "range_percentile"),
    "donchian_channel": ("donchian_upper", "donchian_lower"),
    "ema": ("ema",),
    "lazybear_squeeze_momentum": (
        "squeeze_on",
        "squeeze_off",
        "squeeze_released",
        "squeeze_width_ratio",
        "squeeze_momentum",
        "squeeze_momentum_atr",
        "squeeze_momentum_atr_change",
        "squeeze_duration",
        "squeeze_release_duration",
    ),
    "macd": ("macd", "macd_signal", "macd_histogram"),
    "mfi": ("mfi",),
    "pivot_points_high_low": ("pivot_high", "pivot_low", "pivot_high_rank", "pivot_low_rank"),
    "ppo": ("ppo", "ppo_signal", "ppo_histogram"),
    "rolling_bullish_candle_fraction": ("rolling_bullish_candle_fraction",),
    "rolling_fraction_above_ema": ("rolling_fraction_above_ema",),
    "rolling_level_interactions": (
        "prior_high",
        "prior_low",
        "close_to_upper_atr",
        "close_to_lower_atr",
        "breakout_high_strength",
        "breakout_low_strength",
        "failed_break_high_strength",
        "failed_break_low_strength",
    ),
    "rolling_vwap": ("rolling_vwap",),
    "rsi": ("rsi",),
    "sma": ("sma",),
    "stochastic_oscillator": ("stochastic_k", "stochastic_d"),
    "true_range": ("true_range",),
    "turnover": ("turnover",),
    "turnover_zscore": ("turnover_zscore",),
    "zigzag": ("zigzag_price", "zigzag_direction", "zigzag_return", "zigzag_bars"),
}

PRICE_OUTPUTS = {
    "adr",
    "atr",
    "bollinger_middle",
    "bollinger_upper",
    "bollinger_lower",
    "donchian_upper",
    "donchian_lower",
    "ema",
    "prior_high",
    "prior_low",
    "rolling_vwap",
    "sma",
    "true_range",
    "zigzag_price",
}
MARKER_OUTPUTS = {"pivot_high", "pivot_low"}
HISTOGRAM_OUTPUTS = {
    "inside_bar",
    "outside_bar",
    "squeeze_on",
    "squeeze_off",
    "squeeze_released",
    "macd_histogram",
    "ppo_histogram",
    "zigzag_direction",
}


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def _parameter_kind(annotation: Any, default: Any) -> tuple[str, tuple[Any, ...]]:
    origin = get_origin(annotation)
    if origin is Literal:
        return "choice", tuple(get_args(annotation))
    if origin is tuple or isinstance(default, tuple):
        return "integer_tuple", ()
    if annotation is bool or isinstance(default, bool):
        return "boolean", ()
    if annotation is int or isinstance(default, int):
        return "integer", ()
    if annotation is float or isinstance(default, float):
        return "number", ()
    return "text", ()


def _parameters(indicator_id: str) -> tuple[IndicatorParameter, ...]:
    function = getattr(public_indicators, indicator_id)
    type_hints = get_type_hints(function)
    items: list[IndicatorParameter] = []
    for parameter in list(inspect.signature(function).parameters.values())[1:]:
        required = parameter.default is inspect.Parameter.empty
        default = REQUIRED_DEFAULTS.get((indicator_id, parameter.name))
        if not required:
            default = parameter.default
        kind, choices = _parameter_kind(
            type_hints.get(parameter.name, parameter.annotation), default
        )
        items.append(
            IndicatorParameter(
                name=parameter.name,
                label=_title(parameter.name),
                kind=kind,
                default=default,
                required=required,
                choices=choices,
            )
        )
    return tuple(items)


def _outputs(indicator_id: str) -> tuple[IndicatorOutput, ...]:
    outputs: list[IndicatorOutput] = []
    for output_id in OUTPUT_IDS[indicator_id]:
        style: Literal["line", "histogram", "marker"] = "line"
        if output_id in MARKER_OUTPUTS:
            style = "marker"
        elif output_id in HISTOGRAM_OUTPUTS:
            style = "histogram"
        outputs.append(
            IndicatorOutput(
                id=output_id,
                label=_title(output_id),
                chart_style=style,
                pane="price" if output_id in PRICE_OUTPUTS else "separate",
            )
        )
    return tuple(outputs)


INDICATOR_REGISTRY: dict[str, IndicatorDefinition] = {
    indicator_id: IndicatorDefinition(
        id=indicator_id,
        label=_title(indicator_id),
        input_kind="series" if indicator_id in SERIES_INPUTS else "frame",
        default_source="close" if indicator_id in SERIES_INPUTS else None,
        parameters=_parameters(indicator_id),
        outputs=_outputs(indicator_id),
    )
    for indicator_id in public_indicators.__all__
}


def list_indicator_definitions() -> list[dict[str, Any]]:
    """Return the complete public indicator catalogue for the frontend."""
    return [INDICATOR_REGISTRY[key].to_dict() for key in sorted(INDICATOR_REGISTRY)]


def calculate_indicator(
    prices: pd.DataFrame,
    *,
    indicator_id: str,
    parameters: dict[str, Any] | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Calculate one configured indicator and normalize every public output to a dataframe."""
    try:
        definition = INDICATOR_REGISTRY[indicator_id]
    except KeyError as error:
        raise ValueError(f"Unknown indicator: {indicator_id}") from error

    supplied = dict(parameters or {})
    allowed = {parameter.name for parameter in definition.parameters}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown parameters for {indicator_id}: {', '.join(unknown)}")

    resolved: dict[str, Any] = {}
    for parameter in definition.parameters:
        value = supplied.get(parameter.name, parameter.default)
        if value is None and parameter.required:
            raise ValueError(f"Missing required parameter: {parameter.name}")
        if parameter.kind == "integer_tuple" and isinstance(value, list):
            value = tuple(value)
        resolved[parameter.name] = value

    function = getattr(public_indicators, indicator_id)
    if definition.input_kind == "series":
        selected_source = source or definition.default_source
        if selected_source not in prices.columns:
            raise ValueError(f"Missing source column: {selected_source}")
        result = function(prices[selected_source], **resolved)
    else:
        result = function(prices, **resolved)

    if isinstance(result, pd.Series):
        result = result.to_frame(definition.outputs[0].id)
    else:
        result = result.copy()
        actual_columns = list(result.columns)
        expected_columns = [output.id for output in definition.outputs]
        if len(actual_columns) != len(expected_columns):
            raise ValueError(
                f"Indicator {indicator_id} returned {len(actual_columns)} outputs; "
                f"the registry defines {len(expected_columns)}."
            )
        result.columns = expected_columns
    return result
