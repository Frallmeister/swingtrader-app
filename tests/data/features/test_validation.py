import pandas as pd
import pytest

from swingtrader.data.features._validation import (
    validate_market_price_index,
    validate_required_columns,
)


def test_validate_market_price_index_accepts_canonical_multiindex() -> None:
    validate_market_price_index(_prices())


def test_validate_market_price_index_accepts_canonical_series() -> None:
    validate_market_price_index(_prices()["adjusted_close"])


def test_validate_market_price_index_rejects_identifiers_as_columns() -> None:
    data = _prices().reset_index()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        validate_market_price_index(data)


def test_validate_market_price_index_rejects_wrong_level_order() -> None:
    data = _prices().reorder_levels(["ticker", "provider", "trading_date"])

    with pytest.raises(ValueError, match="in that exact order"):
        validate_market_price_index(data)


def test_validate_market_price_index_rejects_missing_level() -> None:
    data = _prices().droplevel("provider")

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        validate_market_price_index(data)


def test_validate_market_price_index_rejects_duplicate_index_entries() -> None:
    data = pd.concat([_prices().iloc[:1], _prices().iloc[:1]])

    with pytest.raises(ValueError, match="unique index"):
        validate_market_price_index(data)


def test_validate_market_price_index_rejects_unsorted_index() -> None:
    data = _prices().iloc[[1, 0, 2, 3]]

    with pytest.raises(
        ValueError, match="must be sorted by 'provider', 'ticker', and 'trading_date'"
    ):
        validate_market_price_index(data)


def test_validate_market_price_index_accepts_index_after_sorting() -> None:
    data = _prices().iloc[[1, 0, 2, 3]]

    validate_market_price_index(data.sort_index())


def test_validate_market_price_index_rejects_identifier_columns_that_duplicate_levels() -> None:
    data = _prices().assign(ticker="AAA.ST")

    with pytest.raises(ValueError, match="must not also appear as columns: ticker"):
        validate_market_price_index(data)


def test_validate_required_columns_accepts_present_columns() -> None:
    validate_required_columns(_prices(), required_columns={"adjusted_close"})


def test_validate_required_columns_rejects_missing_columns() -> None:
    data = _prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        validate_required_columns(data, required_columns={"adjusted_close"})


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance", "yfinance", "yfinance", "yfinance"],
            "ticker": ["AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST"],
            "trading_date": pd.to_datetime(
                ["2026-07-01", "2026-07-02", "2026-07-01", "2026-07-02"]
            ),
            "adjusted_close": [100.0, 101.0, 200.0, 202.0],
        }
    ).set_index(["provider", "ticker", "trading_date"])
