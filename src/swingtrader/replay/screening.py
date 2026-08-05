"""Continuous indicator screening over data visible at the replay date."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.replay.indicators import calculate_indicator


class ScreeningService:
    """Evaluate saved or ad-hoc screens without a model or feature layer."""

    def run(
        self,
        prices: pd.DataFrame,
        configuration: dict[str, Any],
        *,
        excluded_tickers: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if prices.empty:
            return []
        cache: dict[str, pd.DataFrame] = {}
        result_values: dict[str, pd.Series] = {}
        mask: pd.Series | None = None

        rules = configuration.get("rules", [])
        for index, rule in enumerate(rules):
            values = self._expression(prices, rule["expression"], cache)
            latest = self._latest(values, rule["expression"])
            result_values[f"rule_{index}"] = latest
            current_mask = self._compare(latest, rule)
            mask = current_mask if mask is None else mask & current_mask

        if mask is None:
            tickers = prices.index.get_level_values("ticker").unique()
            mask = pd.Series(True, index=tickers)
        if excluded_tickers:
            mask.loc[mask.index.intersection(excluded_tickers)] = False

        sort_values: list[tuple[dict[str, Any], pd.Series]] = []
        for sort in configuration.get("sort", []):
            values = self._expression(prices, sort["expression"], cache)
            latest = self._latest(values, sort["expression"])
            sort_values.append((sort, latest))

        rows: list[dict[str, Any]] = []
        for ticker in mask.index[mask.fillna(False)]:
            row: dict[str, Any] = {"ticker": str(ticker), "values": {}}
            for key, values in result_values.items():
                row["values"][key] = _json_number(values.get(ticker))
            for index, (_, values) in enumerate(sort_values):
                row["values"][f"sort_{index}"] = _json_number(values.get(ticker))
            rows.append(row)

        for index in reversed(range(len(sort_values))):
            sort, _ = sort_values[index]
            reverse = sort.get("direction", "desc") == "desc"
            rows.sort(
                key=lambda row: _sort_key(row["values"].get(f"sort_{index}"), reverse),
                reverse=reverse,
            )
        return rows

    def _expression(
        self,
        prices: pd.DataFrame,
        expression: dict[str, Any],
        cache: dict[str, pd.DataFrame],
    ) -> pd.Series:
        left = self._operand(prices, expression["left"], cache)
        operation = expression.get("operation", "identity")
        if operation == "identity":
            return left
        right_config = expression.get("right")
        if right_config is None:
            raise ValueError(f"Operation {operation} requires a right operand")
        right = self._operand(prices, right_config, cache)
        if operation == "divide":
            return safe_divide(left.astype(float), right.astype(float))
        if operation == "subtract":
            return left - right
        if operation == "add":
            return left + right
        if operation == "multiply":
            return left * right
        raise ValueError(f"Unsupported screening operation: {operation}")

    def _operand(
        self,
        prices: pd.DataFrame,
        operand: dict[str, Any],
        cache: dict[str, pd.DataFrame],
    ) -> pd.Series:
        if operand["kind"] == "column":
            column = operand["column"]
            if column not in prices.columns:
                raise ValueError(f"Unknown market-data column: {column}")
            return prices[column]
        if operand["kind"] != "indicator":
            raise ValueError(f"Unsupported operand kind: {operand['kind']}")
        config = {
            "indicator_id": operand["indicator_id"],
            "parameters": operand.get("parameters", {}),
            "source": operand.get("source"),
        }
        cache_key = json.dumps(config, sort_keys=True)
        if cache_key not in cache:
            cache[cache_key] = calculate_indicator(prices, **config)
        output = operand["output"]
        if output not in cache[cache_key].columns:
            raise ValueError(f"Unknown output {output!r} for {operand['indicator_id']}")
        return cache[cache_key][output]

    @staticmethod
    def _latest(values: pd.Series, expression: dict[str, Any]) -> pd.Series:
        lookback = int(expression.get("lookback_sessions", 1))
        aggregation = expression.get("aggregation", "latest")

        def aggregate(group: pd.Series) -> float | bool | None:
            window = group.tail(lookback).dropna()
            if window.empty:
                return None
            if aggregation == "latest":
                return window.iloc[-1]
            if aggregation == "maximum":
                return window.max()
            if aggregation == "minimum":
                return window.min()
            if aggregation == "mean":
                return window.mean()
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        return values.groupby(level="ticker", sort=False).apply(aggregate)

    @staticmethod
    def _compare(values: pd.Series, rule: dict[str, Any]) -> pd.Series:
        comparison = rule["comparison"]
        threshold = rule.get("value")
        if comparison == "gt":
            return values > threshold
        if comparison == "gte":
            return values >= threshold
        if comparison == "lt":
            return values < threshold
        if comparison == "lte":
            return values <= threshold
        if comparison == "between":
            return values.between(rule["minimum"], rule["maximum"], inclusive="both")
        if comparison == "eq":
            return values == threshold
        raise ValueError(f"Unsupported comparison: {comparison}")


def _json_number(value: Any) -> float | int | bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _sort_key(value: Any, reverse: bool) -> tuple[bool, float]:
    if value is None:
        return (not reverse, 0.0)
    return (reverse, float(value))
