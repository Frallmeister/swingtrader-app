from datetime import date
from pathlib import Path

import pytest

from swingtrader.data.ingestion.market_data_settings import (
    MarketDataSettingsError,
    load_market_data_settings,
)


def test_load_market_data_settings_reads_packaged_config() -> None:
    settings = load_market_data_settings()

    assert settings.provider == "yfinance"
    assert settings.initial_start_date == date(2000, 1, 1)


def test_load_market_data_settings_reads_explicit_config(tmp_path: Path) -> None:
    path = tmp_path / "market_data.yml"
    path.write_text(
        """
kind: market_data_settings
provider: yfinance
initial_start_date: 2010-01-01
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_market_data_settings(path)

    assert settings.initial_start_date == date(2010, 1, 1)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (
            """
kind: wrong
provider: yfinance
initial_start_date: 2000-01-01
""".lstrip(),
            "Expected kind",
        ),
        (
            """
kind: market_data_settings
provider: other
initial_start_date: 2000-01-01
""".lstrip(),
            "Unsupported market data provider",
        ),
        (
            """
kind: market_data_settings
provider: yfinance
initial_start_date: not-a-date
""".lstrip(),
            "initial_start_date",
        ),
    ],
)
def test_load_market_data_settings_rejects_invalid_config(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    path = tmp_path / "market_data.yml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(MarketDataSettingsError, match=match):
        load_market_data_settings(path)
