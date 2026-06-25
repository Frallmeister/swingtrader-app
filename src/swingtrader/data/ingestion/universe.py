"""Create versioned ticker universe YAML files.

This module contains developer utilities for building available-universe catalog files under
``swingtrader/configs/universes``. The public entrypoint is
``create_ticker_universe_file``; the remaining helpers are implementation details used to
fetch yfinance metadata and write consistently formatted YAML.
"""

import time
from datetime import datetime
from importlib.resources import files

import yaml
import yfinance as yf
from tqdm import tqdm


def _create_template(list_name: str, description: str, date: datetime) -> dict:
    """Create the base universe document structure.

    Parameters
    ----------
    list_name
        Stable logical identifier stored inside the YAML file.
    description
        Human-readable description of the ticker universe.
    date
        Date that the universe definition represents.

    Returns
    -------
    dict
        Universe document with metadata fields and an empty ``symbols`` list.
    """
    return {
        "kind": "ticker_universe",
        "list_name": list_name,
        "description": description,
        "as_of_date": date,
        "source": "yfinance",
        "symbols": [],
    }


def _load_tickers(tickers: list[str], template: dict, sleep_seconds: float) -> dict:
    """Fetch yfinance metadata and append it to a universe document.

    Parameters
    ----------
    tickers
        Ordered ticker symbols to include in the universe.
    template
        Universe document created by ``_create_template``. The document is mutated by
        appending one symbol record per ticker.
    sleep_seconds
        Delay between yfinance requests to avoid sending requests too aggressively.

    Returns
    -------
    dict
        The input universe document with populated symbol metadata.
    """
    info_keys = {
        "shortName": "name",
        "country": "country",
        "currency": "currency",
        "quoteType": "asset_type",
        "industry": "industry",
        "sector": "sector",
        "exchange": "exchange",
    }

    for ticker in tqdm(tickers):
        data = yf.Ticker(ticker)
        info = data.info
        symbols = dict()
        symbols["ticker"] = ticker
        for k, v in info_keys.items():
            symbols[v] = info.get(k)
        template["symbols"].append(symbols)
        time.sleep(sleep_seconds)
    return template


def _write_yaml(data: dict, filename: str) -> None:
    """Write a universe document to the packaged universe config directory.

    Parameters
    ----------
    data
        Universe document to serialize.
    filename
        Output file stem. The file is written as ``<filename>.yml`` under
        ``swingtrader.configs.universes``.
    """

    class IndentedDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow=flow, indentless=False)

    output_path = files("swingtrader.configs.universes").joinpath(f"{filename}.yml")

    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(
            data,
            file,
            Dumper=IndentedDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def create_ticker_universe_file(
    list_name: str,
    description: str,
    as_of_date: datetime,
    tickers: list[str],
    filename: str,
    sleep_seconds: float = 0.5,
) -> None:
    """Create a ticker universe YAML file enriched with yfinance metadata.

    Parameters
    ----------
    list_name
        Stable logical identifier stored inside the YAML file. This can differ from the
        physical file name.
    description
        Human-readable description of the ticker universe.
    as_of_date
        Date that the universe definition represents.
    tickers
        Ordered ticker symbols to include in the universe. The output YAML preserves this
        order.
    filename
        Output file stem. The file is written as ``<filename>.yml`` under
        ``swingtrader.configs.universes``.
    sleep_seconds
        Delay between yfinance requests. Defaults to ``0.5`` seconds.
    """
    template = _create_template(list_name=list_name, description=description, date=as_of_date)
    data = _load_tickers(tickers=tickers, template=template, sleep_seconds=sleep_seconds)
    _write_yaml(data=data, filename=filename)
