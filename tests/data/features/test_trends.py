import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.trends as trend_module
from swingtrader.data.features.trends import (
    add_trend_features,
    ema,
    moving_average_features,
    ppo,
    ppo_features,
    ppo_histogram,
    ppo_percentile,
    ppo_percentile_features,
    ppo_signal,
    sma,
)


def test_add_trend_features_preserves_source_columns_and_adds_features() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    expected_columns = [
        *prices.columns,
        "sma_fast_to_sma_slow",
        "ema_fast_to_ema_slow",
        "ema_fast_to_sma_fast",
        "ppo",
        "ppo_signal",
        "ppo_histogram",
        "ppo_percentile",
    ]
    assert list(result.columns) == expected_columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    expected_sma_ratio = pd.Series(
        [np.nan, np.nan, 13.0 / 12.0 - 1.0, 15.0 / 14.0 - 1.0, np.nan, np.nan, 0.0, 0.0],
        name="sma_fast_to_sma_slow",
    )
    pd.testing.assert_series_equal(
        result["sma_fast_to_sma_slow"],
        expected_sma_ratio,
        check_exact=False,
    )

    pd.testing.assert_series_equal(
        result["ppo"],
        result["ema_fast_to_ema_slow"].rename("ppo"),
        check_exact=False,
    )
    assert "ppo_signal" in result.columns
    assert "ppo_histogram" in result.columns
    pd.testing.assert_series_equal(
        result["ppo_percentile"],
        pd.Series(
            [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
            name="ppo_percentile",
        ),
        check_exact=False,
    )
    assert "ppo" not in prices.columns


def test_add_trend_features_calls_generators_without_duplicate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    calls: list[str] = []
    validation_flags: list[bool] = []

    def fake_moving_average_features(
        observed: pd.DataFrame,
        *,
        fast_slow_lengths: tuple[int, int] = (20, 50),
        source: str = "adjusted_close",
        run_validation: bool = True,
    ) -> pd.DataFrame:
        calls.append("moving_average")
        validation_flags.append(run_validation)
        assert fast_slow_lengths == (2, 3)
        assert source == "adjusted_close"
        return pd.DataFrame(
            {
                "sma_fast_to_sma_slow": [0.0] * len(observed),
                "ema_fast_to_ema_slow": [0.0] * len(observed),
                "ema_fast_to_sma_fast": [0.0] * len(observed),
            },
            index=observed.index,
        )

    def fake_ppo_features(
        observed: pd.DataFrame,
        *,
        lengths: tuple[int, int, int] = (12, 26, 9),
        source: str = "adjusted_close",
        run_validation: bool = True,
    ) -> pd.DataFrame:
        calls.append("ppo")
        validation_flags.append(run_validation)
        assert lengths == (2, 3, 2)
        assert source == "adjusted_close"
        return pd.DataFrame(
            {
                "ppo": [0.0] * len(observed),
                "ppo_signal": [0.0] * len(observed),
                "ppo_histogram": [0.0] * len(observed),
            },
            index=observed.index,
        )

    def fake_ppo_percentile_features(
        observed: pd.DataFrame,
        *,
        min_history: int = 1,
        source: str = "ppo",
        run_validation: bool = True,
    ) -> pd.DataFrame:
        calls.append("ppo_percentile")
        validation_flags.append(run_validation)
        assert min_history == 1
        assert source == "ppo"
        assert "ppo" in observed.columns
        return pd.DataFrame({"ppo_percentile": [0.0] * len(observed)}, index=observed.index)

    monkeypatch.setattr(trend_module, "moving_average_features", fake_moving_average_features)
    monkeypatch.setattr(trend_module, "ppo_features", fake_ppo_features)
    monkeypatch.setattr(trend_module, "ppo_percentile_features", fake_ppo_percentile_features)

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    assert calls == ["ppo", "moving_average", "ppo_percentile"]
    assert validation_flags == [False, False, False]
    assert "ppo_percentile" in result.columns


def test_add_trend_features_accepts_identifiers_as_index_levels() -> None:
    prices = _prices().set_index(["provider", "ticker", "trading_date"])

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
    )

    pd.testing.assert_index_equal(result.index, prices.index)
    assert result.loc[("yfinance", "AAA.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2
    assert result.loc[("yfinance", "BBB.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2


def test_moving_average_features_returns_only_feature_columns() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = moving_average_features(prices, fast_slow_lengths=(2, 3))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "sma_fast_to_sma_slow",
        "ema_fast_to_ema_slow",
        "ema_fast_to_sma_fast",
    ]
    assert not {"provider", "ticker", "trading_date", "adjusted_close"}.intersection(result.columns)
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    pd.testing.assert_series_equal(
        result["sma_fast_to_sma_slow"],
        pd.Series(
            [np.nan, np.nan, 13.0 / 12.0 - 1.0, 15.0 / 14.0 - 1.0, np.nan, np.nan, 0.0, 0.0],
            name="sma_fast_to_sma_slow",
        ),
        check_exact=False,
    )


def test_ppo_features_returns_grouped_feature_columns() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = ppo_features(prices, lengths=(2, 3, 2))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["ppo", "ppo_signal", "ppo_histogram"]
    assert not {"provider", "ticker", "trading_date", "adjusted_close"}.intersection(result.columns)
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)
    pd.testing.assert_series_equal(
        result["ppo"],
        pd.Series(
            [np.nan, np.nan, 0.04888888888888887, 0.05523099415204676, np.nan, np.nan, 0.0, 0.0],
            name="ppo",
        ),
        check_exact=False,
    )


def test_ppo_percentile_features_returns_one_feature_column() -> None:
    data = _prices().assign(ppo=[1.0, 3.0, 2.0, 2.0, 5.0, 4.0, np.nan, 6.0])
    original = data.copy(deep=True)

    result = ppo_percentile_features(data, min_history=1)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["ppo_percentile"]
    assert not {"provider", "ticker", "trading_date", "adjusted_close", "ppo"}.intersection(
        result.columns
    )
    pd.testing.assert_index_equal(result.index, data.index)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_series_equal(
        result["ppo_percentile"],
        pd.Series(
            [np.nan, 1.0, 0.5, 2.0 / 3.0, np.nan, 0.0, np.nan, 1.0],
            name="ppo_percentile",
        ),
        check_exact=False,
    )


def test_feature_generators_can_skip_standalone_validation() -> None:
    prices = _prices()

    moving_average = moving_average_features(prices, fast_slow_lengths=(2, 3), run_validation=False)
    ppo_block = ppo_features(prices, lengths=(2, 3, 2), run_validation=False)
    percentile = ppo_percentile_features(
        prices.assign(ppo=ppo_block["ppo"]),
        min_history=1,
        run_validation=False,
    )

    assert list(moving_average.columns) == [
        "sma_fast_to_sma_slow",
        "ema_fast_to_ema_slow",
        "ema_fast_to_sma_fast",
    ]
    assert list(ppo_block.columns) == ["ppo", "ppo_signal", "ppo_histogram"]
    assert list(percentile.columns) == ["ppo_percentile"]


def test_sma_and_ema_calculate_one_sequence() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    simple = sma(prices, length=2)
    exponential = ema(prices, length=2)

    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0, 15.0], name="adjusted_close"),
    )
    pd.testing.assert_series_equal(
        exponential,
        pd.Series(
            [
                np.nan,
                11.333333333333332,
                13.11111111111111,
                15.037037037037036,
            ],
            name="adjusted_close",
        ),
        check_exact=False,
    )


def test_ppo_returns_percent_by_default_or_ratio_when_requested() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    ratio = ppo(prices, fast=2, slow=3, use_percent=False)
    percent = ppo(prices, fast=2, slow=3)

    expected_ratio = pd.Series(
        [np.nan, np.nan, 0.04888888888888887, 0.05523099415204676],
        name="adjusted_close",
    )
    pd.testing.assert_series_equal(ratio, expected_ratio, check_exact=False)
    pd.testing.assert_series_equal(percent, expected_ratio.mul(100), check_exact=False)


def test_ppo_signal_and_histogram_calculate_from_existing_ppo_columns() -> None:
    data = pd.Series([0.0, 1.0, 2.0, 3.0], name="ppo")

    signal = ppo_signal(data, length=2)
    histogram = ppo_histogram(data, signal)

    expected_signal = pd.Series(
        [np.nan, 0.6666666666666666, 1.5555555555555556, 2.518518518518518],
        name="ppo",
    )
    expected_histogram = pd.Series([np.nan, 1.0 / 3.0, 4.0 / 9.0, 13.0 / 27.0], name="ppo")
    pd.testing.assert_series_equal(signal, expected_signal, check_exact=False)
    pd.testing.assert_series_equal(histogram, expected_histogram, check_exact=False)


def test_ppo_percentile_calculates_grouped_point_in_time_rank() -> None:
    data = pd.Series([1.0, 3.0, 2.0, 2.0], name="ppo")

    percentile = ppo_percentile(data, min_history=1)

    expected = pd.Series(
        [np.nan, 1.0, 0.5, 2.0 / 3.0],
        name="ppo",
    )
    pd.testing.assert_series_equal(percentile, expected, check_exact=False)


def test_trend_helpers_reject_invalid_inputs() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="fast length"):
        add_trend_features(prices, fast_slow_lengths=(3, 2), ppo_lengths=(2, 3, 2))

    with pytest.raises(ValueError, match="positive integer"):
        sma(prices["adjusted_close"], length=0)

    with pytest.raises(ValueError, match="Missing required columns"):
        moving_average_features(prices, source="missing")

    with pytest.raises(ValueError, match="fast length"):
        ppo(prices["adjusted_close"], fast=3, slow=2)

    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        ppo_features(prices.drop(columns="adjusted_close"))

    with pytest.raises(ValueError, match="Missing required columns: ppo"):
        ppo_percentile_features(prices)

    with pytest.raises(ValueError, match="positive integer"):
        ppo_percentile_features(prices.assign(ppo=0.0), min_history=0)


@pytest.mark.parametrize("length", [0, -1, True, 2.5, "2"])
def test_moving_averages_reject_invalid_lengths(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sma(
            _prices()["adjusted_close"],
            length=length,  # type: ignore[arg-type]
        )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance"] * 8,
            "ticker": [
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
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
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                ]
            ).date,
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
