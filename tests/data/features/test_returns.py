import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.returns as return_module
from swingtrader.data.features.returns import add_return_features, return_features


def test_return_features_returns_only_feature_columns() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = return_features(prices, horizons=(1, 2))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["return_1d", "return_2d"]
    assert not {"provider", "ticker", "trading_date", "adjusted_close"}.intersection(result.columns)
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)
    pd.testing.assert_series_equal(
        result["return_1d"],
        pd.Series([np.nan, 0.1, 0.1, np.nan, -0.2, 0.1], name="return_1d"),
    )
    pd.testing.assert_series_equal(
        result["return_2d"],
        pd.Series([np.nan, np.nan, 0.21, np.nan, np.nan, -0.12], name="return_2d"),
    )


def test_add_return_features_preserves_source_columns_and_adds_features() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = add_return_features(prices, horizons=(1, 2))

    assert list(result.columns) == [*prices.columns, "return_1d", "return_2d"]
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)


def test_add_return_features_calls_generator_without_duplicate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()

    def fake_return_features(
        observed: pd.DataFrame,
        horizons: tuple[int, ...],
        *,
        source: str = "adjusted_close",
        run_validation: bool = True,
    ) -> pd.DataFrame:
        assert observed is not prices
        assert horizons == (1,)
        assert source == "adjusted_close"
        assert run_validation is False
        return pd.DataFrame({"return_1d": [0.0] * len(observed)}, index=observed.index)

    monkeypatch.setattr(return_module, "return_features", fake_return_features)

    result = add_return_features(prices, horizons=(1,))

    assert "return_1d" in result.columns


def test_return_features_accepts_identifiers_as_index_levels() -> None:
    prices = _prices().iloc[:3].set_index(["provider", "ticker", "trading_date"])

    result = return_features(prices, horizons=(1,))

    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_series_equal(
        result["return_1d"],
        pd.Series([np.nan, 0.1, 0.1], index=prices.index, name="return_1d"),
    )


@pytest.mark.parametrize(
    "horizons",
    [(), (1, 1), (0,), (-1,), (True,), (1.5,)],
)
def test_add_return_features_rejects_invalid_horizons(horizons: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="horizon"):
        return_features(_prices().iloc[:1], horizons=horizons)


def test_return_features_runs_standalone_validation_by_default() -> None:
    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        return_features(_prices().drop(columns="adjusted_close"), horizons=(1,))


def test_return_features_can_skip_standalone_validation() -> None:
    prices = _prices().iloc[[1, 0, 2]].reset_index(drop=True)

    result = return_features(prices, horizons=(1,), run_validation=False)

    assert list(result.columns) == ["return_1d"]


def test_add_return_features_handles_empty_input() -> None:
    prices = pd.DataFrame(columns=["provider", "ticker", "trading_date", "adjusted_close"])

    result = add_return_features(prices, horizons=(1, 5))

    assert result.empty
    assert result["return_1d"].dtype == "float64"
    assert result["return_5d"].dtype == "float64"


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance"] * 6,
            "ticker": ["AAA.ST", "AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST", "BBB.ST"],
            "trading_date": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                ]
            ).date,
            "adjusted_close": [100.0, 110.0, 121.0, 50.0, 40.0, 44.0],
        }
    )
