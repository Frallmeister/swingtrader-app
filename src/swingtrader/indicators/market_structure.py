"""Market-structure indicators for ordered price histories.

This module identifies local structural properties of price series. Pivot highs
and lows are evaluated within configurable windows containing observations on
both sides of each candidate row. Zig Zag filters confirmed local extrema into
an alternating sequence that meets a minimum percentage reversal. Indicator
outputs are aligned with historical pivot rows, although a pivot cannot be
confirmed until the configured number of right-side observations is available.

Public functions accept either one ordered instrument dataframe or a canonical
multi-instrument dataframe carrying the ``provider``, ``ticker``, and
``trading_date`` index levels. Multi-instrument calculations are isolated by
provider and ticker so that observations from one instrument cannot affect
another.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import validate_length


@dataclass(frozen=True, slots=True)
class _ZigZagPivot:
    """Represent one Zig Zag pivot candidate or retained turning point.

    Attributes:
        position: Zero-based row position within one instrument's ordered price
            history.
        price: Pivot price, taken from ``high`` for swing highs and ``low`` for
            swing lows.
        direction: Pivot direction, where ``1`` denotes a swing high and ``-1``
            denotes a swing low.
    """

    position: int
    price: float
    direction: int


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
        raise ValueError(f"rank_output must be either 'rank' or 'strength'; got {rank_output!r}.")

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


def zigzag(
    data: pd.DataFrame,
    *,
    deviation: float = 5.0,
    pivot_legs: int = 10,
) -> pd.DataFrame:
    """Identify significant alternating swing highs and lows.

    Zig Zag first identifies confirmed pivot candidates from raw ``high`` and
    ``low`` values. ``pivot_legs`` is the total confirmation width, matching the
    TradingView input: the value is divided by two using floor division and the
    resulting number of observations is required on both sides of each candidate.

    A pivot high must be strictly greater than every value to its left and greater
    than or equal to every value to its right. A pivot low uses the inverse
    comparisons. These asymmetric tie rules retain the first value in a run of
    equal extrema.

    Candidate pivots are processed chronologically. An opposite-direction pivot
    is retained only when its reversal from the last retained pivot is at least
    ``deviation`` percent. A same-direction candidate replaces the latest pivot
    when it is more extreme. The retained sequence therefore alternates between
    highs and lows.

    The result contains:

    - ``zigzag_price``: raw ``high`` for retained highs and raw ``low`` for
      retained lows;
    - ``zigzag_direction``: ``1`` for a high, ``-1`` for a low, and ``0`` on
      non-pivot rows;
    - ``zigzag_return``: ``current_price / previous_price - 1`` on retained
      pivot rows;
    - ``zigzag_bars``: number of input observations between consecutive retained
      pivots.

    The first retained pivot has missing return and bar-count values. The output
    is retrospective and aligned with historical pivot rows. Each pivot is first
    knowable ``pivot_legs // 2`` observations later, and the latest retained
    endpoint can be replaced by a later, more extreme same-direction pivot.

    Parameters
    ----------
    data
        Ordered price observations containing ``high`` and ``low`` columns.
    deviation
        Minimum reversal percentage relative to the last retained pivot. A value
        of ``5.0`` means five percent. Zero disables the percentage filter while
        preserving pivot confirmation and direction alternation.
    pivot_legs
        Total number of bars used for pivot confirmation. Must be an integer of
        at least two. Odd values are rounded down before assigning equal left and
        right legs.

    Returns
    -------
    pandas.DataFrame
        Retrospective Zig Zag pivots and retained-swing measurements, aligned
        with the input index.

    Warnings
    -------
    This indicator is retrospective. Its pivot-aligned outputs depend on later
    observations and must not be used directly as row-aligned machine-learning
    features. Use ``zigzag_features`` or ``add_market_structure_features`` for
    point-in-time-safe predictor columns.
    """
    validate_required_columns(data, required_columns={"high", "low"})
    deviation_ratio, legs = _validate_zigzag_parameters(
        deviation=deviation,
        pivot_legs=pivot_legs,
    )

    return apply_by_ticker(
        data,
        lambda group: _zigzag(
            group,
            deviation_ratio=deviation_ratio,
            legs=legs,
        ),
    )


def _pivot_points_high_low(
    data: pd.DataFrame,
    *,
    high_left: int,
    high_right: int,
    low_left: int,
    low_right: int,
    kind: Literal["high_low", "balanced"],
    rank_output: Literal["rank", "strength"],
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


def _zigzag(
    data: pd.DataFrame,
    *,
    deviation_ratio: float,
    legs: int,
) -> pd.DataFrame:
    """Calculate retrospective Zig Zag pivots for one ordered instrument.

    Local pivot-high and pivot-low candidates are detected from the configured
    number of bars on each side, then processed chronologically into an
    alternating sequence of retained pivots. More extreme candidates replace
    the current endpoint, while opposite-direction candidates are accepted only
    when they satisfy the minimum reversal deviation.

    The returned dataframe is aligned to the input rows. Pivot prices, signed
    returns, and bar counts are populated only on retained pivot rows.
    """
    high_candidates, low_candidates = _zigzag_candidate_prices(data, legs=legs)
    high_candidate_values = high_candidates.to_numpy(dtype="float64")
    low_candidate_values = low_candidates.to_numpy(dtype="float64")
    pivots: list[_ZigZagPivot] = []

    for position in range(len(data)):
        _update_zigzag_from_candidates(
            pivots,
            position=position,
            high_candidates=high_candidate_values,
            low_candidates=low_candidate_values,
            deviation_ratio=deviation_ratio,
        )

    prices = [math.nan] * len(data)
    directions = [0] * len(data)
    returns = [math.nan] * len(data)
    bars = [math.nan] * len(data)

    for pivot in pivots:
        prices[pivot.position] = pivot.price
        directions[pivot.position] = pivot.direction

    for previous, current in zip(pivots, pivots[1:], strict=False):
        returns[current.position] = current.price / previous.price - 1.0
        bars[current.position] = float(current.position - previous.position)

    return pd.DataFrame(
        {
            "zigzag_price": pd.Series(prices, index=data.index, dtype="float64"),
            "zigzag_direction": pd.Series(
                directions,
                index=data.index,
                dtype="int8",
            ),
            "zigzag_return": pd.Series(returns, index=data.index, dtype="float64"),
            "zigzag_bars": pd.Series(bars, index=data.index, dtype="float64"),
        },
        index=data.index,
    )


def _confirmed_zigzag_state(
    data: pd.DataFrame,
    *,
    deviation: float,
    pivot_legs: int,
    consistency_pivots: int = 4,
) -> pd.DataFrame:
    """Return point-in-time Zig Zag state for one ordered instrument.

    This private lower-layer helper is used by the market-structure feature
    family. Candidate pivots update the state only on their confirmation rows,
    so the returned history does not backfill final pivots into periods where
    they were not yet known.
    """
    deviation_ratio, legs = _validate_zigzag_parameters(
        deviation=deviation,
        pivot_legs=pivot_legs,
    )
    _validate_consistency_pivots(consistency_pivots)
    high_candidates, low_candidates = _zigzag_candidate_prices(data, legs=legs)
    high_candidate_values = high_candidates.to_numpy(dtype="float64")
    low_candidate_values = low_candidates.to_numpy(dtype="float64")

    last_price = [math.nan] * len(data)
    previous_price = [math.nan] * len(data)
    last_direction = [math.nan] * len(data)
    last_position = [math.nan] * len(data)
    previous_position = [math.nan] * len(data)
    last_high_price = [math.nan] * len(data)
    previous_high_price = [math.nan] * len(data)
    last_high_position = [math.nan] * len(data)
    previous_high_position = [math.nan] * len(data)
    last_low_price = [math.nan] * len(data)
    previous_low_price = [math.nan] * len(data)
    last_low_position = [math.nan] * len(data)
    previous_low_position = [math.nan] * len(data)
    high_consistency = [math.nan] * len(data)
    low_consistency = [math.nan] * len(data)
    pivots: list[_ZigZagPivot] = []

    for current_position in range(len(data)):
        candidate_position = current_position - legs
        if candidate_position >= 0:
            _update_zigzag_from_candidates(
                pivots,
                position=candidate_position,
                high_candidates=high_candidate_values,
                low_candidates=low_candidate_values,
                deviation_ratio=deviation_ratio,
            )

        if not pivots:
            continue

        latest = pivots[-1]
        last_price[current_position] = latest.price
        last_direction[current_position] = latest.direction
        last_position[current_position] = float(latest.position)

        if len(pivots) >= 2:
            previous = pivots[-2]
            previous_price[current_position] = previous.price
            previous_position[current_position] = float(previous.position)

        latest_high, earlier_high = _last_two_zigzag_pivots(pivots, direction=1)
        if latest_high is not None:
            last_high_price[current_position] = latest_high.price
            last_high_position[current_position] = float(latest_high.position)
        if earlier_high is not None:
            previous_high_price[current_position] = earlier_high.price
            previous_high_position[current_position] = float(earlier_high.position)

        latest_low, earlier_low = _last_two_zigzag_pivots(pivots, direction=-1)
        if latest_low is not None:
            last_low_price[current_position] = latest_low.price
            last_low_position[current_position] = float(latest_low.position)
        if earlier_low is not None:
            previous_low_price[current_position] = earlier_low.price
            previous_low_position[current_position] = float(earlier_low.position)

        high_consistency[current_position] = _zigzag_pivot_consistency(
            pivots,
            direction=1,
            count=consistency_pivots,
        )
        low_consistency[current_position] = _zigzag_pivot_consistency(
            pivots,
            direction=-1,
            count=consistency_pivots,
        )

    return pd.DataFrame(
        {
            "_zigzag_last_price": last_price,
            "_zigzag_previous_price": previous_price,
            "_zigzag_last_direction": pd.Series(
                last_direction,
                index=data.index,
                dtype="float64",
            ),
            "_zigzag_last_position": last_position,
            "_zigzag_previous_position": previous_position,
            "_zigzag_last_high_price": last_high_price,
            "_zigzag_previous_high_price": previous_high_price,
            "_zigzag_last_high_position": last_high_position,
            "_zigzag_previous_high_position": previous_high_position,
            "_zigzag_last_low_price": last_low_price,
            "_zigzag_previous_low_price": previous_low_price,
            "_zigzag_last_low_position": last_low_position,
            "_zigzag_previous_low_position": previous_low_position,
            "_zigzag_high_consistency": high_consistency,
            "_zigzag_low_consistency": low_consistency,
        },
        index=data.index,
    )


def _zigzag_pivot_consistency(
    pivots: list[_ZigZagPivot],
    *,
    direction: int,
    count: int,
) -> float:
    """Calculate Kendall's tau-b for recent pivots in one direction.

    Pivot order is strictly chronological, so the first ranked variable has no
    ties. Price ties are handled by Kendall's tau-b. The result is
    missing until ``count`` pivots are available and when every selected pivot
    price is equal.
    """
    prices: list[float] = []

    for pivot in reversed(pivots):
        if pivot.direction == direction:
            prices.append(pivot.price)

        if len(prices) == count:
            break

    if len(prices) < count:
        return math.nan

    prices.reverse()

    result = kendalltau(
        range(count),
        prices,
        variant="b",
    )

    return float(result.statistic)


def _last_two_zigzag_pivots(
    pivots: list[_ZigZagPivot],
    *,
    direction: int,
) -> tuple[_ZigZagPivot | None, _ZigZagPivot | None]:
    """Return the latest two retained pivots in one direction."""
    matching = [pivot for pivot in reversed(pivots[-4:]) if pivot.direction == direction]
    latest = matching[0] if matching else None
    previous = matching[1] if len(matching) >= 2 else None
    return latest, previous


def _zigzag_candidate_prices(
    data: pd.DataFrame,
    *,
    legs: int,
) -> tuple[pd.Series, pd.Series]:
    """Return high and low prices only where a row is a pivot candidate."""
    high = data["high"]
    low = data["low"]
    pivot_high = high.notna()
    pivot_low = low.notna()

    for distance in range(1, legs + 1):
        pivot_high &= high.gt(high.shift(distance)) & high.ge(high.shift(-distance))
        pivot_low &= low.lt(low.shift(distance)) & low.le(low.shift(-distance))

    return high.where(pivot_high), low.where(pivot_low)


def _update_zigzag_from_candidates(
    pivots: list[_ZigZagPivot],
    *,
    position: int,
    high_candidates: np.ndarray,
    low_candidates: np.ndarray,
    deviation_ratio: float,
) -> None:
    """Update the retained Zig Zag sequence from one row of pivot candidates.

    The high candidate is evaluated before the low candidate. Processing stops
    after the first candidate that appends a new pivot or replaces the current
    endpoint, ensuring that at most one structural Zig Zag update is applied for
    the row.

    Args:
        pivots: Mutable sequence of retained Zig Zag pivots for one instrument.
        position: Zero-based row position of the candidate values.
        high_candidates: Row-aligned candidate prices for pivot highs, with
            ``NaN`` on rows that are not high candidates.
        low_candidates: Row-aligned candidate prices for pivot lows, with
            ``NaN`` on rows that are not low candidates.
        deviation_ratio: Minimum absolute relative price change required to
            accept an opposite-direction reversal.
    """
    high_price = high_candidates[position]
    if pd.notna(high_price) and _update_zigzag(
        pivots,
        _ZigZagPivot(position=position, price=float(high_price), direction=1),
        deviation_ratio=deviation_ratio,
    ):
        return

    low_price = low_candidates[position]
    if pd.notna(low_price):
        _update_zigzag(
            pivots,
            _ZigZagPivot(position=position, price=float(low_price), direction=-1),
            deviation_ratio=deviation_ratio,
        )


def _update_zigzag(
    pivots: list[_ZigZagPivot],
    candidate: _ZigZagPivot,
    *,
    deviation_ratio: float,
) -> bool:
    """Update the retained Zig Zag sequence from one pivot candidate.

    A more extreme candidate in the current direction replaces the latest
    endpoint, while an opposite-direction candidate is appended only when it
    satisfies the minimum reversal deviation.

    Returns:
        ``True`` if the retained pivot sequence changed, otherwise ``False``.
    """
    if not pivots:
        pivots.append(candidate)
        return True

    latest = pivots[-1]

    if candidate.direction == latest.direction:
        more_extreme = (
            candidate.price > latest.price
            if candidate.direction == 1
            else candidate.price < latest.price
        )
        if more_extreme:
            pivots[-1] = candidate
            return True
        return False

    reversal = abs(candidate.price / latest.price - 1.0)

    if reversal >= deviation_ratio:
        pivots.append(candidate)
        return True

    return False


def _validate_zigzag_parameters(
    *,
    deviation: float,
    pivot_legs: int,
) -> tuple[float, int]:
    """Validate Zig Zag parameters and return ratio plus per-side legs."""
    if (
        isinstance(deviation, bool)
        or not isinstance(deviation, int | float)
        or not math.isfinite(deviation)
        or deviation < 0
    ):
        raise ValueError(f"deviation must be a non-negative finite number; got {deviation!r}.")

    if isinstance(pivot_legs, bool) or not isinstance(pivot_legs, int) or pivot_legs < 2:
        raise ValueError(
            f"pivot_legs must be an integer greater than or equal to 2; got {pivot_legs!r}."
        )

    return deviation / 100.0, pivot_legs // 2


def _validate_consistency_pivots(consistency_pivots: int) -> None:
    """Validate the number of same-direction pivots used for consistency."""
    if (
        isinstance(consistency_pivots, bool)
        or not isinstance(consistency_pivots, int)
        or consistency_pivots < 2
    ):
        raise ValueError(
            "consistency_pivots must be an integer greater than or equal to 2; "
            f"got {consistency_pivots!r}."
        )


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
