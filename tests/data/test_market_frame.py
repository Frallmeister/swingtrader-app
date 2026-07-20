import pandas as pd
import pytest

from swingtrader.data.market_frame import (
    apply_by_ticker,
    validate_market_price_index,
    validate_required_columns,
    validate_temporal_order,
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


def test_apply_by_ticker_isolates_groups_and_preserves_order() -> None:
    values = _prices()["adjusted_close"]

    result = apply_by_ticker(values, lambda group: group.cumsum())

    pd.testing.assert_index_equal(result.index, values.index)
    # A per-group cumulative sum stays within each ticker; had the tickers not
    # been isolated the BBB.ST totals would carry AAA.ST's running sum forward.
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series([100.0, 201.0], name="adjusted_close"),
    )
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([200.0, 402.0], name="adjusted_close"),
    )


def test_apply_by_ticker_applies_func_directly_to_single_ordered_series() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        name="adjusted_close",
    )

    result = apply_by_ticker(values, lambda group: group.cumsum())

    pd.testing.assert_series_equal(
        result,
        pd.Series([1.0, 3.0, 6.0], index=values.index, name="adjusted_close"),
    )


def test_validate_temporal_order_rejects_unordered_datetime_index() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2026-07-03", "2026-07-01", "2026-07-02"]),
        name="adjusted_close",
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        validate_temporal_order(values)


def test_validate_temporal_order_accepts_ordered_datetime_index() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        name="adjusted_close",
    )

    validate_temporal_order(values)


def test_validate_temporal_order_accepts_non_temporal_index() -> None:
    values = pd.Series([1.0, 2.0, 3.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    validate_temporal_order(values)


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
