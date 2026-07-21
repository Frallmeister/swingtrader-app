import pandas as pd
import pytest

from swingtrader.indicators.market_structure import pivot_points_high_low


def test_pivot_points_high_low_returns_expected_pivots_and_ranks() -> None:
    prices = _prices()

    result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
    )

    expected = pd.DataFrame(
        {
            "pivot_high": pd.array(
                [pd.NA, True, False, True, False, True, pd.NA],
                dtype="boolean",
            ),
            "pivot_low": pd.array(
                [pd.NA, True, False, True, False, True, pd.NA],
                dtype="boolean",
            ),
            "pivot_high_rank": [
                float("nan"),
                1.0,
                3.0,
                1.0,
                2.0,
                1.0,
                float("nan"),
            ],
            "pivot_low_rank": [
                float("nan"),
                1.0,
                3.0,
                1.0,
                2.0,
                1.0,
                float("nan"),
            ],
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_pivot_points_high_low_returns_expected_strengths() -> None:
    prices = _prices()

    result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
        rank_output="strength",
    )

    expected = pd.DataFrame(
        {
            "pivot_high": pd.array(
                [pd.NA, True, False, True, False, True, pd.NA],
                dtype="boolean",
            ),
            "pivot_low": pd.array(
                [pd.NA, True, False, True, False, True, pd.NA],
                dtype="boolean",
            ),
            "pivot_high_strength": [
                float("nan"),
                1.0,
                0.0,
                1.0,
                0.5,
                1.0,
                float("nan"),
            ],
            "pivot_low_strength": [
                float("nan"),
                1.0,
                0.0,
                1.0,
                0.5,
                1.0,
                float("nan"),
            ],
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_pivot_points_high_low_uses_balanced_candle_values() -> None:
    prices = pd.DataFrame(
        {
            "open": [10.0, 0.0, 10.0],
            "high": [10.0, 11.0, 10.0],
            "low": [0.0, 0.0, 0.0],
            "close": [10.0, 0.0, 10.0],
        }
    )

    high_low_result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
        kind="high_low",
    )
    balanced_result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
        kind="balanced",
    )

    assert high_low_result.loc[1, "pivot_high"]
    assert high_low_result.loc[1, "pivot_high_rank"] == 1.0

    assert not balanced_result.loc[1, "pivot_high"]
    assert balanced_result.loc[1, "pivot_high_rank"] == 3.0


def test_pivot_points_high_low_supports_asymmetric_distances() -> None:
    prices = pd.DataFrame(
        {
            "open": [1.0, 4.0, 3.0, 5.0, 1.0],
            "high": [1.0, 5.0, 4.0, 6.0, 2.0],
            "low": [0.0, 3.0, 2.0, 4.0, 0.0],
            "close": [0.5, 4.0, 3.0, 5.0, 1.0],
        }
    )

    short_right = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
    )
    long_right = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=2,
        low_left=1,
        low_right=2,
    )

    # With one right-side candle, the high at index 1 is the largest value in
    # [1, 5, 4]. Including one additional right-side candle adds the value 6,
    # changing the candidate from rank 1 to rank 2.
    assert short_right.loc[1, "pivot_high"]
    assert short_right.loc[1, "pivot_high_rank"] == 1.0

    assert not long_right.loc[1, "pivot_high"]
    assert long_right.loc[1, "pivot_high_rank"] == 2.0


def test_pivot_points_high_low_assigns_tied_extrema_rank_one() -> None:
    prices = pd.DataFrame(
        {
            "open": [3.0, 4.0, 4.0, 2.0],
            "high": [4.0, 5.0, 5.0, 3.0],
            "low": [2.0, 1.0, 1.0, 2.0],
            "close": [3.0, 4.0, 4.0, 2.0],
        }
    )

    result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
    )

    assert result.loc[1, "pivot_high"]
    assert result.loc[2, "pivot_high"]
    assert result.loc[1, "pivot_high_rank"] == 1.0
    assert result.loc[2, "pivot_high_rank"] == 1.0

    assert result.loc[1, "pivot_low"]
    assert result.loc[2, "pivot_low"]
    assert result.loc[1, "pivot_low_rank"] == 1.0
    assert result.loc[2, "pivot_low_rank"] == 1.0


def test_pivot_points_high_low_marks_incomplete_windows_as_missing() -> None:
    prices = _prices()

    result = pivot_points_high_low(
        prices,
        high_left=2,
        high_right=1,
        low_left=1,
        low_right=2,
    )

    assert result["pivot_high"].iloc[:2].isna().all()
    assert pd.isna(result["pivot_high"].iloc[-1])
    assert result["pivot_high_rank"].iloc[:2].isna().all()
    assert pd.isna(result["pivot_high_rank"].iloc[-1])

    assert pd.isna(result["pivot_low"].iloc[0])
    assert result["pivot_low"].iloc[-2:].isna().all()
    assert pd.isna(result["pivot_low_rank"].iloc[0])
    assert result["pivot_low_rank"].iloc[-2:].isna().all()


def test_pivot_points_high_low_preserves_index_and_row_order() -> None:
    prices = _prices().set_axis(pd.Index([6, 1, 5, 0, 4, 2, 3]))

    result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
    )

    pd.testing.assert_index_equal(result.index, prices.index)


def test_pivot_points_high_low_groups_by_provider_and_ticker() -> None:
    prices = _multi_ticker_prices()

    result = pivot_points_high_low(
        prices,
        high_left=1,
        high_right=1,
        low_left=1,
        low_right=1,
    )

    pd.testing.assert_index_equal(result.index, prices.index)

    for ticker in ["AAA.ST", "BBB.ST"]:
        ticker_prices = prices.loc[("yfinance", ticker)]
        expected = pivot_points_high_low(
            ticker_prices,
            high_left=1,
            high_right=1,
            low_left=1,
            low_right=1,
        )

        pd.testing.assert_frame_equal(
            result.loc[("yfinance", ticker)],
            expected,
        )


@pytest.mark.parametrize(
    "parameter",
    ["high_left", "high_right", "low_left", "low_right"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_pivot_points_high_low_rejects_invalid_distances(
    parameter: str,
    value: object,
) -> None:
    kwargs = {
        "high_left": 1,
        "high_right": 1,
        "low_left": 1,
        "low_right": 1,
        parameter: value,
    }

    with pytest.raises(ValueError, match="positive integer"):
        pivot_points_high_low(
            _prices(),
            **kwargs,  # type: ignore[arg-type]
        )


def test_pivot_points_high_low_rejects_invalid_kind() -> None:
    with pytest.raises(
        ValueError,
        match="kind must be either 'high_low' or 'balanced'",
    ):
        pivot_points_high_low(
            _prices(),
            high_left=1,
            high_right=1,
            low_left=1,
            low_right=1,
            kind="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("column", ["high", "low"])
def test_pivot_points_high_low_requires_price_columns_when_kind_is_hl(column: str) -> None:
    prices = _prices().drop(columns=column)

    with pytest.raises(ValueError, match="Missing required columns"):
        pivot_points_high_low(
            prices,
            high_left=1,
            high_right=1,
            low_left=1,
            low_right=1,
            kind="high_low",
        )


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_pivot_points_high_low_requires_price_columns_when_kind_is_balanced(column: str) -> None:
    prices = _prices().drop(columns=column)

    with pytest.raises(ValueError, match="Missing required columns"):
        pivot_points_high_low(
            prices,
            high_left=1,
            high_right=1,
            low_left=1,
            low_right=1,
            kind="balanced",
        )


def test_pivot_points_high_low_rejects_invalid_rank_output() -> None:
    with pytest.raises(
        ValueError,
        match="rank_output must be either 'rank' or 'strength'",
    ):
        pivot_points_high_low(
            _prices(),
            high_left=1,
            high_right=1,
            low_left=1,
            low_right=1,
            rank_output="invalid",  # type: ignore[arg-type]
        )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [0.5, 2.0, 1.5, 4.0, 3.0, 3.0, 1.5],
            "high": [1.0, 3.0, 2.0, 5.0, 4.0, 4.0, 2.0],
            "low": [5.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0],
            "close": [0.8, 2.5, 1.8, 4.5, 3.5, 3.5, 1.8],
        }
    )


def _multi_ticker_prices() -> pd.DataFrame:
    aaa = _prices().copy()
    bbb = pd.DataFrame(
        {
            "open": [10.0] * len(aaa),
            "high": [11.0] * len(aaa),
            "low": [9.0] * len(aaa),
            "close": [10.0] * len(aaa),
        }
    )

    dates = pd.date_range("2026-01-01", periods=len(aaa), freq="D")

    aaa.index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * len(aaa),
            ["AAA.ST"] * len(aaa),
            dates,
        ],
        names=["provider", "ticker", "trading_date"],
    )
    bbb.index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * len(bbb),
            ["BBB.ST"] * len(bbb),
            dates,
        ],
        names=["provider", "ticker", "trading_date"],
    )

    return pd.concat([aaa, bbb])
