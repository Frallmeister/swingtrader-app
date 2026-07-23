import pandas as pd
import pytest

from swingtrader.data.features.volume import add_volume_features
from swingtrader.indicators.volume import turnover_zscore


def test_add_volume_features_preserves_source_data_and_adds_zscore() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_volume_features(
        prices,
        turnover_zscore_length=3,
    )

    assert list(result.columns) == [
        *prices.columns,
        "turnover_zscore",
    ]
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    expected = turnover_zscore(
        prices,
        length=3,
        log=True,
    )
    pd.testing.assert_series_equal(
        result["turnover_zscore"],
        expected,
        check_exact=False,
    )


def test_add_volume_features_uses_log_turnover_by_default() -> None:
    prices = _indexed_prices()

    result = add_volume_features(
        prices,
        turnover_zscore_length=3,
    )

    expected = turnover_zscore(
        prices,
        length=3,
        log=True,
    )
    raw_expected = turnover_zscore(
        prices,
        length=3,
        log=False,
    )

    pd.testing.assert_series_equal(
        result["turnover_zscore"],
        expected,
        check_exact=False,
    )
    assert not result["turnover_zscore"].equals(raw_expected)


def test_add_volume_features_can_use_raw_turnover() -> None:
    prices = _indexed_prices()

    result = add_volume_features(
        prices,
        turnover_zscore_length=3,
        turnover_zscore_log=False,
    )

    expected = turnover_zscore(
        prices,
        length=3,
        log=False,
    )
    pd.testing.assert_series_equal(
        result["turnover_zscore"],
        expected,
        check_exact=False,
    )


def test_add_volume_features_uses_custom_zscore_length() -> None:
    prices = _indexed_prices()

    short_window = add_volume_features(
        prices,
        turnover_zscore_length=3,
    )
    long_window = add_volume_features(
        prices,
        turnover_zscore_length=4,
    )

    short_values = short_window.loc[
        ("yfinance", "AAA.ST"),
        "turnover_zscore",
    ]
    long_values = long_window.loc[
        ("yfinance", "AAA.ST"),
        "turnover_zscore",
    ]

    assert short_values.iloc[:2].isna().all()
    assert short_values.iloc[2:].notna().all()

    assert long_values.iloc[:3].isna().all()
    assert long_values.iloc[3:].notna().all()


def test_add_volume_features_calculates_each_ticker_independently() -> None:
    prices = _indexed_prices()

    result = add_volume_features(
        prices,
        turnover_zscore_length=3,
        turnover_zscore_log=False,
    )

    pd.testing.assert_index_equal(result.index, prices.index)

    aaa = result.loc[
        ("yfinance", "AAA.ST"),
        "turnover_zscore",
    ]
    bbb = result.loc[
        ("yfinance", "BBB.ST"),
        "turnover_zscore",
    ]

    assert aaa.iloc[:2].isna().all()
    assert aaa.iloc[2:].notna().all()

    # BBB.ST has constant turnover. Its entire result remains missing,
    # demonstrating that AAA.ST's history cannot enter its rolling window.
    assert bbb.isna().all()


def test_add_volume_features_rejects_identifiers_as_columns() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        add_volume_features(
            prices,
            turnover_zscore_length=3,
        )


def test_add_volume_features_rejects_unsorted_input() -> None:
    prices = _indexed_prices().iloc[[1, 0, 2, 3, 4, 5, 6, 7, 8, 9]]

    with pytest.raises(ValueError, match="must be sorted"):
        add_volume_features(
            prices,
            turnover_zscore_length=3,
        )


@pytest.mark.parametrize("missing_column", ["close", "volume"])
def test_add_volume_features_requires_input_columns(
    missing_column: str,
) -> None:
    prices = _indexed_prices().drop(columns=missing_column)

    with pytest.raises(ValueError, match="Missing required columns"):
        add_volume_features(
            prices,
            turnover_zscore_length=3,
        )


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_add_volume_features_rejects_invalid_zscore_length(
    length: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        add_volume_features(
            _indexed_prices(),
            turnover_zscore_length=length,  # type: ignore[arg-type]
        )


def test_add_volume_features_rejects_zscore_length_below_two() -> None:
    with pytest.raises(ValueError, match="must be at least 2"):
        add_volume_features(
            _indexed_prices(),
            turnover_zscore_length=1,
        )


@pytest.mark.parametrize("log", [0, 1, "true", None])
def test_add_volume_features_rejects_non_boolean_log(
    log: object,
) -> None:
    with pytest.raises(ValueError, match="log parameter must be a boolean"):
        add_volume_features(
            _indexed_prices(),
            turnover_zscore_length=3,
            turnover_zscore_log=log,  # type: ignore[arg-type]
        )


def _indexed_prices() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"])


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance"] * 10,
            "ticker": [
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
            ],
            "trading_date": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                    "2026-07-07",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                    "2026-07-07",
                ]
            ).date,
            "close": [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "volume": [
                100.0,
                200.0,
                300.0,
                500.0,
                800.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
        }
    )
