from datetime import date, timedelta

import pandas as pd
import pytest

from swingtrader.modeling.datasets.labels import (
    V1_RETURN_THRESHOLD,
    generate_v1_labels,
)


def test_generate_v1_labels_calculates_known_forward_returns() -> None:
    prices = _price_frame(
        adjusted_closes=[
            100,
            101,
            102,
            103,
            104,
            110,
            111,
            112,
            113,
            114,
            120,
            121,
            122,
            123,
            124,
            130,
        ]
    )

    labels = generate_v1_labels(prices)

    first_row = labels.iloc[0]
    assert first_row["forward_return_5d"] == pytest.approx(0.10)
    assert first_row["forward_return_10d"] == pytest.approx(0.20)
    assert first_row["forward_return_15d"] == pytest.approx(0.30)


def test_generate_v1_labels_calculates_tickers_independently() -> None:
    prices = pd.concat(
        [
            _price_frame(ticker="AAA.ST", adjusted_closes=[100, 100, 100, 100, 100, 120]),
            _price_frame(ticker="BBB.ST", adjusted_closes=[200, 200, 200, 200, 200, 180]),
        ],
        ignore_index=True,
    )

    labels = generate_v1_labels(prices)

    aaa_first = labels.loc[labels["ticker"].eq("AAA.ST")].iloc[0]
    bbb_first = labels.loc[labels["ticker"].eq("BBB.ST")].iloc[0]
    assert aaa_first["forward_return_5d"] == pytest.approx(0.20)
    assert bbb_first["forward_return_5d"] == pytest.approx(-0.10)


def test_generate_v1_labels_calculates_providers_independently() -> None:
    prices = pd.concat(
        [
            _price_frame(provider="yfinance", adjusted_closes=[100, 100, 100, 100, 100, 110]),
            _price_frame(provider="other", adjusted_closes=[100, 100, 100, 100, 100, 90]),
        ],
        ignore_index=True,
    )

    labels = generate_v1_labels(prices)

    yfinance_first = labels.loc[labels["provider"].eq("yfinance")].iloc[0]
    other_first = labels.loc[labels["provider"].eq("other")].iloc[0]
    assert yfinance_first["forward_return_5d"] == pytest.approx(0.10)
    assert other_first["forward_return_5d"] == pytest.approx(-0.10)


def test_generate_v1_labels_handles_unsorted_input_and_preserves_row_order() -> None:
    prices = _price_frame(adjusted_closes=[100, 101, 102, 103, 104, 110])
    unsorted_prices = prices.iloc[[5, 0, 2, 1, 4, 3]].reset_index(drop=True)

    labels = generate_v1_labels(unsorted_prices)

    assert list(labels["trading_date"]) == list(unsorted_prices["trading_date"])
    first_observation = labels.loc[labels["trading_date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]
    assert first_observation["forward_return_5d"] == pytest.approx(0.10)


def test_generate_v1_labels_does_not_mutate_input() -> None:
    prices = _price_frame(adjusted_closes=[100, 101, 102, 103, 104, 110])
    original_prices = prices.copy(deep=True)

    generate_v1_labels(prices)

    pd.testing.assert_frame_equal(prices, original_prices)


def test_generate_v1_labels_leaves_tail_outcomes_missing_for_each_horizon() -> None:
    prices = _price_frame(adjusted_closes=range(1, 17))

    labels = generate_v1_labels(prices)

    assert labels["forward_return_5d"].tail(5).isna().all()
    assert labels["forward_return_10d"].tail(10).isna().all()
    assert labels["forward_return_15d"].tail(15).isna().all()


def test_generate_v1_labels_leaves_missing_current_adjusted_close_outcome_missing() -> None:
    prices = _price_frame(adjusted_closes=[pd.NA, 101, 102, 103, 104, 110])

    labels = generate_v1_labels(prices)

    assert pd.isna(labels.iloc[0]["forward_return_5d"])
    assert labels.iloc[0]["target_significant_up_5d"] is pd.NA


def test_generate_v1_labels_leaves_zero_current_adjusted_close_outcome_missing() -> None:
    prices = _price_frame(adjusted_closes=[0, 101, 102, 103, 104, 110])

    labels = generate_v1_labels(prices)

    assert pd.isna(labels.iloc[0]["forward_return_5d"])
    assert labels.iloc[0]["target_significant_up_5d"] is pd.NA


def test_generate_v1_labels_leaves_zero_future_adjusted_close_outcome_missing() -> None:
    prices = _price_frame(adjusted_closes=[100, 101, 102, 103, 104, 0])

    labels = generate_v1_labels(prices)

    assert pd.isna(labels.iloc[0]["forward_return_5d"])
    assert labels.iloc[0]["target_significant_up_5d"] is pd.NA


def test_generate_v1_labels_leaves_missing_future_adjusted_close_outcome_missing() -> None:
    prices = _price_frame(adjusted_closes=[100, 101, 102, 103, 104, pd.NA])

    labels = generate_v1_labels(prices)

    assert pd.isna(labels.iloc[0]["forward_return_5d"])
    assert labels.iloc[0]["target_significant_up_5d"] is pd.NA


def test_generate_v1_labels_uses_strict_v1_threshold_for_nullable_target() -> None:
    exact_threshold_price = 100 * (1 + V1_RETURN_THRESHOLD)
    prices = _price_frame(
        adjusted_closes=[
            100,
            100,
            100,
            100,
            100,
            exact_threshold_price,
            100 * (1 + V1_RETURN_THRESHOLD + 0.0001),
            100,
            100,
            100,
            100,
            100,
        ]
    )

    labels = generate_v1_labels(prices)

    assert pytest.approx(0.01311017, abs=0.00000001) == V1_RETURN_THRESHOLD
    assert not labels.iloc[0]["target_significant_up_5d"]
    assert labels.iloc[1]["target_significant_up_5d"]
    assert not labels.iloc[6]["target_significant_up_5d"]
    assert labels.iloc[-1]["target_significant_up_5d"] is pd.NA


def test_generate_v1_labels_returns_nullable_boolean_target_dtype() -> None:
    labels = generate_v1_labels(_price_frame(adjusted_closes=[100, 100, 100, 100, 100, 110]))

    assert str(labels["target_significant_up_5d"].dtype) == "boolean"


def test_generate_v1_labels_rejects_duplicate_observations() -> None:
    prices = pd.concat(
        [
            _price_frame(adjusted_closes=[100]),
            _price_frame(adjusted_closes=[101]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate provider/ticker/trading_date"):
        generate_v1_labels(prices)


def test_generate_v1_labels_returns_empty_frame_with_stable_label_columns() -> None:
    prices = pd.DataFrame(
        columns=["provider", "ticker", "trading_date", "adjusted_close", "volume"]
    )

    labels = generate_v1_labels(prices)

    assert labels.empty
    assert list(labels.columns) == [
        "provider",
        "ticker",
        "trading_date",
        "adjusted_close",
        "volume",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_15d",
        "target_significant_up_5d",
    ]
    assert pd.api.types.is_float_dtype(labels["forward_return_5d"])
    assert str(labels["target_significant_up_5d"].dtype) == "boolean"


def test_generate_v1_labels_rejects_missing_required_columns() -> None:
    prices = pd.DataFrame(columns=["provider", "ticker", "trading_date"])

    with pytest.raises(ValueError, match="missing required columns: adjusted_close"):
        generate_v1_labels(prices)


def _price_frame(
    *,
    adjusted_closes: list[object] | range,
    provider: str = "yfinance",
    ticker: str = "AAA.ST",
) -> pd.DataFrame:
    start_date = date(2026, 1, 1)
    return pd.DataFrame(
        {
            "provider": provider,
            "ticker": ticker,
            "trading_date": [
                pd.Timestamp(start_date + timedelta(days=index))
                for index in range(len(adjusted_closes))
            ],
            "adjusted_close": list(adjusted_closes),
        }
    )
