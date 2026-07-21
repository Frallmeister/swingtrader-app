"""Market-structure indicators for ordered price histories.

This module identifies local structural properties of price series. Pivot highs
and lows are evaluated within configurable windows containing observations on
both sides of each candidate row. The outputs are aligned with the candidate
row, although a pivot cannot be confirmed until the configured number of
right-side observations has become available.

Public functions accept either one ordered instrument dataframe or a canonical
multi-instrument dataframe carrying the ``provider``, ``ticker``, and
``trading_date`` index levels. Multi-instrument calculations are isolated by
provider and ticker so that observations from one instrument cannot affect
another.
"""

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
    rank_output: Literal["rank", "strength"] = "rank",
) -> pd.DataFrame:
    """Identify pivot highs and lows in ordered price histories.

    A row is a pivot high when its selected high value has rank one among the
    candidate row, the preceding ``high_left`` rows, and the following
    ``high_right`` rows. A row is a pivot low when its selected low value has
    rank one among the corresponding low window. Equal extreme values share
    rank one and are therefore all marked as pivots.

    When ``kind="high_low"``, pivot highs are calculated from ``high`` and
    pivot lows from ``low``. When ``kind="balanced"``, the values are adjusted
    toward the candle body using::

        balanced_high = (2 * high + max(open, close)) / 3
        balanced_low = (2 * low + min(open, close)) / 3

    The result is aligned with the candidate row. Because each calculation uses
    observations to the right of that row, a reported pivot high is only known
    after ``high_right`` later observations and a reported pivot low only after
    ``low_right`` later observations. The outputs must therefore be shifted to
    their confirmation rows before being used as point-in-time model features.

    Parameters
    ----------
    data
        Ordered price observations for one instrument, or canonical
        multi-instrument observations indexed by ``provider``, ``ticker``, and
        ``trading_date``. ``kind="high_low"`` requires ``high`` and ``low``.
        ``kind="balanced"`` additionally requires ``open`` and ``close``.
    high_left
        Number of observations preceding each candidate high. Must be a
        positive integer.
    high_right
        Number of observations following each candidate high. Must be a
        positive integer.
    low_left
        Number of observations preceding each candidate low. Must be a
        positive integer.
    low_right
        Number of observations following each candidate low. Must be a
        positive integer.
    kind
        Price representation used for ranking. ``"high_low"`` uses the raw
        high and low values. ``"balanced"`` adjusts each extreme toward the
        candle body.
    rank_output
        Representation returned alongside the pivot flags. ``"rank"`` returns
        ordinal ranks beginning at one. ``"strength"`` converts those ranks to
        the interval from zero to one, where one is the strongest possible
        pivot candidate.

    Returns
    -------
    pandas.DataFrame
        A dataframe with nullable Boolean columns ``pivot_high`` and
        ``pivot_low``. When ``rank_output="rank"``, it also contains
        ``pivot_high_rank`` and ``pivot_low_rank``. When
        ``rank_output="strength"``, those columns are replaced by
        ``pivot_high_strength`` and ``pivot_low_strength``. Rows without a
        complete corresponding window are missing.

    Raises
    ------
    ValueError
        If a distance is not a positive integer, ``kind`` or ``rank_output`` is
        unsupported, required price columns are missing, the canonical
        multi-instrument index is invalid, or observations are not ordered
        chronologically.
    """
    validate_length(high_left)
    validate_length(high_right)
    validate_length(low_left)
    validate_length(low_right)

    if kind == "high_low":
        required_columns = {"high", "low"}
    elif kind == "balanced":
        required_columns = {"open", "high", "low", "close"}
    else:
        raise ValueError(f"kind must be either 'high_low' or 'balanced'; got {kind!r}.")
    validate_required_columns(data, required_columns=required_columns)
    if rank_output not in ("rank", "strength"):
        raise ValueError(f"rank_output must be either 'rank' or 'strength'; got {rank_output!r}")

    return apply_by_ticker(
        data,
        lambda group: _pivot_points_high_low(
            group,
            high_left=high_left,
            high_right=high_right,
            low_left=low_left,
            low_right=low_right,
            kind=kind,
            rank_output=rank_output,
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
    rank_output: Literal["rank", "strength"] = "rank",
) -> pd.DataFrame:
    """Calculate pivot outputs for one ordered instrument.

    The caller is responsible for validating parameters, required columns, and
    input structure. Results are aligned with the candidate rows rather than
    the later rows on which the pivots become confirmed.

    Parameters
    ----------
    data
        Ordered price observations for one instrument.
    high_left
        Number of observations preceding each candidate high.
    high_right
        Number of observations following each candidate high.
    low_left
        Number of observations preceding each candidate low.
    low_right
        Number of observations following each candidate low.
    kind
        Price representation used for the high and low calculations.
    rank_output
        Whether to return ordinal ranks or normalized pivot strengths.

    Returns
    -------
    pandas.DataFrame
        Pivot flags and the requested rank representation, preserving the input
        index.
    """
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

    if rank_output == "strength":
        high_output = (1.0 - (pivot_high_rank - 1.0) / (high_left + high_right)).rename(
            "pivot_high_strength"
        )

        low_output = (1.0 - (pivot_low_rank - 1.0) / (low_left + low_right)).rename(
            "pivot_low_strength"
        )
    else:
        high_output = pivot_high_rank
        low_output = pivot_low_rank

    return pd.concat([pivot_high, pivot_low, high_output, low_output], axis=1)


def _centered_rank(
    values: pd.Series,
    *,
    left: int,
    right: int,
    ascending: bool,
) -> pd.Series:
    """Rank each value within an asymmetric window centered on its row.

    The rank is one plus the number of surrounding values that are strictly
    more extreme than the candidate. With ``ascending=True``, smaller values
    receive better ranks. With ``ascending=False``, larger values receive
    better ranks. Equal values therefore share the best applicable rank,
    matching minimum-rank tie semantics.

    Parameters
    ----------
    values
        Ordered values for one instrument.
    left
        Number of preceding values included in each window.
    right
        Number of following values included in each window.
    ascending
        Whether smaller values should receive lower ranks.

    Returns
    -------
    pandas.Series
        Floating-point ranks aligned with ``values``. Rows without a complete
        window, including rows whose window contains missing values, are
        missing.
    """
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
