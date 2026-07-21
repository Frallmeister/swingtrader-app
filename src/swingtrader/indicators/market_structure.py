"""ADD MODULE DOCSTRING HERE."""

from typing import Literal

import pandas as pd

from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import validate_length


def pivot_points_high_low(
    data: pd.DataFrame,
    *,
    high_left: int = 10,
    high_right: int = 10,
    low_left: int = 10,
    low_right: int = 10,
    kind: Literal["high_low", "balanced"] = "high_low",
    normalize_rank: bool = False,
) -> pd.DataFrame:
    """ADD DOCSTRING HERE."""
    validate_length(high_left)
    validate_length(high_right)
    validate_length(low_left)
    validate_length(low_right)
    validate_required_columns(data, required_columns={"open", "high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _pivot_points_high_low(
            group,
            high_left=high_left,
            high_right=high_right,
            low_left=low_left,
            low_right=low_right,
            kind=kind,
            normalize_rank=normalize_rank,
        ),
    )


def _pivot_points_high_low(
    data: pd.DataFrame,
    *,
    high_left: int = 10,
    high_right: int = 10,
    low_left: int = 10,
    low_right: int = 10,
    kind: Literal["high_low", "balanced"] = "high_low",
    normalize_rank: bool = False,
) -> pd.DataFrame:
    """ADD DOCSTRING HERE."""
    if kind == "high_low":
        high = data["high"]
        low = data["low"]
    elif kind == "balanced":
        high = (2 * data["high"] + data[["close", "open"]].max(axis=1)) / 3
        low = (2 * data["low"] + data[["close", "open"]].min(axis=1)) / 3
    else:
        raise ValueError(f"kind must be either 'high_low' or 'balanced'; got {kind!r}.")

    pivot_high_rank = _centered_rank(
        high,
        left=high_left,
        right=high_right,
        ascending=False,
    ).rename("pivot_high_rank")

    pivot_low_rank = _centered_rank(
        low,
        left=low_left,
        right=low_right,
        ascending=True,
    ).rename("pivot_low_rank")

    pivot_high = (
        pivot_high_rank.eq(1).where(pivot_high_rank.notna()).astype("boolean").rename("pivot_high")
    )

    pivot_low = (
        pivot_low_rank.eq(1).where(pivot_low_rank.notna()).astype("boolean").rename("pivot_low")
    )

    if normalize_rank:
        pivot_high_rank = (1.0 - (pivot_high_rank - 1.0) / (high_left + high_right)).rename(
            "pivot_high_strength"
        )

        pivot_low_rank = (1.0 - (pivot_low_rank - 1.0) / (low_left + low_right)).rename(
            "pivot_low_strength"
        )

    return pd.concat([pivot_high, pivot_low, pivot_high_rank, pivot_low_rank], axis=1)


def _centered_rank(
    values: pd.Series,
    *,
    left: int,
    right: int,
    ascending: bool,
) -> pd.Series:
    """ADD DOCSTRING HERE."""
    rank = pd.Series(1.0, index=values.index)
    complete_window = values.notna()

    for distance in range(1, left + 1):
        neighbour = values.shift(distance)

        if ascending:
            rank += neighbour.lt(values)
        else:
            rank += neighbour.gt(values)

        complete_window &= neighbour.notna()

    for distance in range(1, right + 1):
        neighbour = values.shift(-distance)

        if ascending:
            rank += neighbour.lt(values)
        else:
            rank += neighbour.gt(values)

        complete_window &= neighbour.notna()

    return rank.where(complete_window)
