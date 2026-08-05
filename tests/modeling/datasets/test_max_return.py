from datetime import date, timedelta

import pandas as pd
import pytest

from swingtrader.modeling.datasets import MAX_RETURN_TARGET_SET
from swingtrader.modeling.datasets.max_return import add_future_max_return_targets


def test_add_future_max_return_targets_calculates_known_max_returns() -> None:
    prices = _price_frame(
        opens=[10, 20, 30, 40, 50, 60],
        highs=[15, 25, 35, 45, 55, 65],
    )

    targets = add_future_max_return_targets(prices, horizons=(2, 3))

    first_row = targets.iloc[0]
    # Entry at next open (20); best high over next 2 sessions is 35.
    assert first_row["future_max_return_2d"] == pytest.approx(35 / 20 - 1)
    # Best high over next 3 sessions is 45.
    assert first_row["future_max_return_3d"] == pytest.approx(45 / 20 - 1)


def test_add_future_max_return_targets_calculates_tickers_independently() -> None:
    prices = pd.concat(
        [
            _price_frame(ticker="AAA.ST", opens=[10, 20, 30], highs=[10, 25, 35]),
            _price_frame(ticker="BBB.ST", opens=[100, 200, 300], highs=[100, 180, 260]),
        ]
    ).sort_index()

    targets = add_future_max_return_targets(prices, horizons=(2,))

    aaa_first = targets.xs(("yfinance", "AAA.ST"), level=["provider", "ticker"]).iloc[0]
    bbb_first = targets.xs(("yfinance", "BBB.ST"), level=["provider", "ticker"]).iloc[0]
    assert aaa_first["future_max_return_2d"] == pytest.approx(35 / 20 - 1)
    assert bbb_first["future_max_return_2d"] == pytest.approx(260 / 200 - 1)


def test_add_future_max_return_targets_calculates_providers_independently() -> None:
    prices = pd.concat(
        [
            _price_frame(provider="yfinance", opens=[10, 20, 30], highs=[10, 25, 35]),
            _price_frame(provider="other", opens=[10, 20, 30], highs=[10, 22, 24]),
        ]
    ).sort_index()

    targets = add_future_max_return_targets(prices, horizons=(2,))

    yfinance_first = targets.xs(("yfinance", "AAA.ST"), level=["provider", "ticker"]).iloc[0]
    other_first = targets.xs(("other", "AAA.ST"), level=["provider", "ticker"]).iloc[0]
    assert yfinance_first["future_max_return_2d"] == pytest.approx(35 / 20 - 1)
    assert other_first["future_max_return_2d"] == pytest.approx(24 / 20 - 1)


def test_add_future_max_return_targets_leaves_tail_outcomes_missing_for_each_horizon() -> None:
    prices = _price_frame(opens=range(1, 17), highs=range(1, 17))

    targets = add_future_max_return_targets(prices, horizons=(2, 3))

    # A horizon-h target needs the next open plus h future highs.
    assert targets["future_max_return_2d"].tail(2).isna().all()
    assert targets["future_max_return_3d"].tail(3).isna().all()


def test_add_future_max_return_targets_leaves_missing_high_in_window_outcome_missing() -> None:
    prices = _price_frame(
        opens=[10, 20, 30, 40],
        highs=[15, 25, pd.NA, 45],
    )

    targets = add_future_max_return_targets(prices, horizons=(2,))

    assert pd.isna(targets.iloc[0]["future_max_return_2d"])


def test_add_future_max_return_targets_leaves_non_positive_entry_open_outcome_missing() -> None:
    prices = _price_frame(
        opens=[10, 0, 30, 40],
        highs=[15, 25, 35, 45],
    )

    targets = add_future_max_return_targets(prices, horizons=(2,))

    assert pd.isna(targets.iloc[0]["future_max_return_2d"])


def test_add_future_max_return_targets_preserves_index_and_does_not_mutate_input() -> None:
    prices = _price_frame(opens=[10, 20, 30, 40], highs=[15, 25, 35, 45])
    original_prices = prices.copy(deep=True)

    targets = add_future_max_return_targets(prices, horizons=(2,))

    pd.testing.assert_frame_equal(prices, original_prices)
    assert targets.index.equals(prices.index)


def test_add_future_max_return_targets_returns_float_dtype() -> None:
    targets = add_future_max_return_targets(
        _price_frame(opens=[10, 20, 30], highs=[15, 25, 35]),
        horizons=(2,),
    )

    assert pd.api.types.is_float_dtype(targets["future_max_return_2d"])


@pytest.mark.parametrize("horizons", [(), (2, 2), (0,), (-1,), (True,), (1.5,)])
def test_add_future_max_return_targets_rejects_invalid_horizons(
    horizons: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_future_max_return_targets(
            _price_frame(opens=[10, 20], highs=[15, 25]),
            horizons=horizons,
        )


def test_add_future_max_return_targets_rejects_missing_required_columns() -> None:
    prices = _price_frame(opens=[10, 20], highs=[15, 25]).drop(columns=["high"])

    with pytest.raises(ValueError, match="high"):
        add_future_max_return_targets(prices, horizons=(2,))


def test_add_future_max_return_targets_rejects_flat_identifier_columns() -> None:
    prices = _price_frame(opens=[10, 20], highs=[15, 25]).reset_index()

    with pytest.raises(ValueError, match="MultiIndex"):
        add_future_max_return_targets(prices, horizons=(2,))


def test_max_return_target_set_applies_configured_horizons() -> None:
    prices = _price_frame(opens=range(1, 21), highs=range(1, 21))

    targets = MAX_RETURN_TARGET_SET.apply(prices)

    assert MAX_RETURN_TARGET_SET.identifier == "max_return_targets:1"
    assert MAX_RETURN_TARGET_SET.target_columns == (
        "future_max_return_5d",
        "future_max_return_10d",
        "future_max_return_15d",
    )
    for column in MAX_RETURN_TARGET_SET.target_columns:
        assert column in targets.columns


def _price_frame(
    *,
    opens: list[object] | range,
    highs: list[object] | range,
    provider: str = "yfinance",
    ticker: str = "AAA.ST",
) -> pd.DataFrame:
    opens = list(opens)
    highs = list(highs)
    start_date = date(2026, 1, 1)
    return (
        pd.DataFrame(
            {
                "provider": provider,
                "ticker": ticker,
                "trading_date": [
                    pd.Timestamp(start_date + timedelta(days=index)) for index in range(len(opens))
                ],
                "open": opens,
                "high": highs,
            }
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
