import pandas as pd
import pytest

from swingtrader.data.features._validation import validate_feature_input, validate_temporal_order


def test_validate_feature_input_accepts_identifiers_as_columns() -> None:
    validate_feature_input(_prices(), required_columns={"adjusted_close"})


def test_validate_feature_input_accepts_identifiers_as_index_levels() -> None:
    validate_feature_input(
        _prices().set_index(["provider", "ticker", "trading_date"]),
        required_columns={"adjusted_close"},
    )


def test_validate_feature_input_rejects_duplicate_identifier_locations() -> None:
    data = _prices().set_index("provider", drop=False)

    with pytest.raises(ValueError, match="both as columns and index levels: provider"):
        validate_feature_input(data, required_columns={"adjusted_close"})


def test_validate_feature_input_rejects_split_identifier_locations() -> None:
    data = _prices().set_index("trading_date")

    with pytest.raises(ValueError, match="must all be columns or all be named index levels"):
        validate_feature_input(data, required_columns={"adjusted_close"})


def test_validate_feature_input_rejects_missing_required_columns() -> None:
    data = _prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        validate_feature_input(data, required_columns={"adjusted_close"})


def test_validate_temporal_order_accepts_ordered_dates_per_ticker() -> None:
    validate_temporal_order(_prices())


def test_validate_temporal_order_accepts_identifiers_as_index_levels() -> None:
    validate_temporal_order(_prices().set_index(["provider", "ticker", "trading_date"]))


@pytest.mark.parametrize(
    "trading_dates",
    [
        ["2026-07-01", "2026-07-01"],
        ["2026-07-02", "2026-07-01"],
    ],
)
def test_validate_temporal_order_rejects_non_increasing_dates(
    trading_dates: list[str],
) -> None:
    data = pd.DataFrame(
        {
            "provider": ["yfinance", "yfinance"],
            "ticker": ["AAA.ST", "AAA.ST"],
            "trading_date": pd.to_datetime(trading_dates).date,
            "adjusted_close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="strictly ordered by trading_date"):
        validate_temporal_order(data)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance", "yfinance", "yfinance", "yfinance"],
            "ticker": ["AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST"],
            "trading_date": pd.to_datetime(
                ["2026-07-01", "2026-07-02", "2026-07-01", "2026-07-02"]
            ).date,
            "adjusted_close": [100.0, 101.0, 200.0, 202.0],
        }
    )
