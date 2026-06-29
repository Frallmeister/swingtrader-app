from datetime import UTC, date, datetime

import pandas as pd  # type: ignore[import-untyped]
import pytest

from swingtrader.data.clients import yfinance as yfinance_client  # type: ignore[import-untyped]


def test_normalize_daily_prices_converts_ticker_first_columns_to_bronze_rows() -> None:
    fetched_at = datetime(2026, 6, 29, 9, 30, tzinfo=UTC)
    raw_prices = pd.DataFrame(
        data=[
            [10.0, 11.0, 9.5, 10.5, 1000, 0.0, 0.0, 20.0, 21.0, 19.5, 20.5, 2000, 1.5, 0.0],
            [10.5, 12.0, 10.0, 11.5, 1100, 0.0, 0.0, 20.5, 22.0, 20.0, 21.5, 2100, 0.0, 2.0],
        ],
        index=pd.to_datetime(["2026-06-26", "2026-06-27"]),
        columns=pd.MultiIndex.from_product(
            [
                ["AAK.ST", "VOLV-B.ST"],
                ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
            ]
        ),
    )

    prices = yfinance_client.normalize_daily_prices(
        raw_prices,
        tickers=["AAK.ST", "VOLV-B.ST"],
        fetched_at=fetched_at,
        request_id="test-request",
    )

    assert list(prices.columns) == yfinance_client.DAILY_PRICE_COLUMNS
    assert len(prices) == 4

    first_row = prices.iloc[0]
    assert first_row["provider"] == "yfinance"
    assert first_row["ticker"] == "AAK.ST"
    assert first_row["trading_date"] == date(2026, 6, 26)
    assert first_row["open"] == 10.0
    assert first_row["close"] == 10.5
    assert first_row["volume"] == 1000
    assert pd.isna(first_row["adjusted_close"])
    assert first_row["fetched_at"] == fetched_at
    assert first_row["request_id"] == "test-request"


def test_normalize_daily_prices_reads_adjusted_close_when_available() -> None:
    raw_prices = pd.DataFrame(
        data=[[10.0, 10.2]],
        index=pd.to_datetime(["2026-06-26"]),
        columns=pd.MultiIndex.from_tuples([("AAK.ST", "Close"), ("AAK.ST", "Adj Close")]),
    )

    prices = yfinance_client.normalize_daily_prices(
        raw_prices,
        tickers=["AAK.ST"],
        fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
        request_id="test-request",
    )

    assert prices.loc[0, "close"] == 10.0
    assert prices.loc[0, "adjusted_close"] == 10.2


def test_normalize_daily_prices_converts_field_first_columns_to_bronze_rows() -> None:
    raw_prices = pd.DataFrame(
        data=[[10.0, 20.0, 10.5, 20.5]],
        index=pd.to_datetime(["2026-06-26"]),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Open", "AAK.ST"),
                ("Open", "VOLV-B.ST"),
                ("Close", "AAK.ST"),
                ("Close", "VOLV-B.ST"),
            ]
        ),
    )

    prices = yfinance_client.normalize_daily_prices(
        raw_prices,
        tickers=["AAK.ST", "VOLV-B.ST"],
        fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
        request_id="test-request",
    )

    assert prices.loc[0, "ticker"] == "AAK.ST"
    assert prices.loc[0, "open"] == 10.0
    assert prices.loc[0, "close"] == 10.5
    assert prices.loc[1, "ticker"] == "VOLV-B.ST"
    assert prices.loc[1, "open"] == 20.0
    assert prices.loc[1, "close"] == 20.5


def test_download_daily_prices_calls_yfinance_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_prices = pd.DataFrame(
        data=[[10.0, 11.0, 9.5, 10.5, 1000, 0.0, 0.0]],
        index=pd.to_datetime(["2026-06-26"]),
        columns=pd.MultiIndex.from_product(
            [
                ["AAK.ST"],
                ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
            ]
        ),
    )
    calls = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return raw_prices

    monkeypatch.setattr(yfinance_client.yf, "download", fake_download)

    prices = yfinance_client.download_daily_prices(
        tickers=["AAK.ST"],
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 29),
        fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
        request_id="test-request",
    )

    assert calls == [
        {
            "tickers": ["AAK.ST"],
            "start": "2026-06-01",
            "end": "2026-06-29",
            "group_by": "ticker",
            "actions": True,
            "auto_adjust": False,
            "progress": False,
            "threads": True,
        }
    ]
    assert prices.loc[0, "ticker"] == "AAK.ST"
    assert prices.loc[0, "request_id"] == "test-request"


def test_download_daily_prices_rejects_empty_ticker_list() -> None:
    with pytest.raises(ValueError, match="At least one ticker"):
        yfinance_client.download_daily_prices(
            tickers=[],
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 29),
        )


def test_normalize_daily_prices_rejects_naive_fetched_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        yfinance_client.normalize_daily_prices(
            pd.DataFrame({"Close": [10.0]}, index=pd.to_datetime(["2026-06-26"])),
            tickers=["AAK.ST"],
            fetched_at=datetime(2026, 6, 29, 9, 30),
            request_id="test-request",
        )
