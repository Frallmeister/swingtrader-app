"""Load source-controlled settings for market data ingestion jobs.

The settings file captures project-level market data defaults that should be reviewed and
committed, such as the first historical date to request for tickers with no bronze rows.
"""

from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml

from swingtrader.data.clients import yfinance as yfinance_client

ConfigFile = Path | Traversable

MARKET_DATA_SETTINGS_FILE = "market_data.yml"
MARKET_DATA_SETTINGS_KIND = "market_data_settings"


@dataclass(frozen=True)
class MarketDataSettings:
    """Configuration for market data ingestion jobs.

    Attributes
    ----------
    provider
        Market data provider identifier supported by the current ingestion path.
    initial_start_date
        Inclusive start date used for tickers that do not yet have bronze daily price rows.
    """

    provider: str
    initial_start_date: date


class MarketDataSettingsError(Exception):
    """Raised when market data settings are missing, malformed, or unsupported."""


def load_market_data_settings(path: ConfigFile | None = None) -> MarketDataSettings:
    """Load market data settings from YAML configuration.

    Parameters
    ----------
    path
        Optional path to a settings file. When omitted, the packaged project settings are used.

    Returns
    -------
    MarketDataSettings
        Parsed market data settings.

    Raises
    ------
    MarketDataSettingsError
        Raised when the file is missing, is not a mapping, has the wrong ``kind``, references
        an unsupported provider, or contains an invalid ``initial_start_date``.
    """
    settings_path = path or files("swingtrader.configs").joinpath(MARKET_DATA_SETTINGS_FILE)
    if not settings_path.is_file():
        msg = f"Missing market data settings file: {MARKET_DATA_SETTINGS_FILE}"
        raise MarketDataSettingsError(msg)

    data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MarketDataSettingsError("Market data settings must be a YAML mapping.")

    _validate_kind(data.get("kind"))
    provider = _parse_provider(data.get("provider"))
    initial_start_date = _parse_initial_start_date(data.get("initial_start_date"))
    return MarketDataSettings(
        provider=provider,
        initial_start_date=initial_start_date,
    )


def _validate_kind(kind: Any) -> None:
    if kind != MARKET_DATA_SETTINGS_KIND:
        msg = f"Expected kind = {MARKET_DATA_SETTINGS_KIND!r} but got {kind!r}."
        raise MarketDataSettingsError(msg)


def _parse_provider(provider: Any) -> str:
    if provider != yfinance_client.PROVIDER:
        msg = f"Unsupported market data provider: {provider!r}."
        raise MarketDataSettingsError(msg)
    return provider


def _parse_initial_start_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        msg = "initial_start_date must be an ISO date string."
        raise MarketDataSettingsError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = "initial_start_date must be an ISO date string."
        raise MarketDataSettingsError(msg) from exc
