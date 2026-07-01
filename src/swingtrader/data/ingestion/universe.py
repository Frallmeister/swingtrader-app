"""Create curated ticker universe YAML files as a local development utility.

This module builds available-universe catalog files from provider metadata. It is intended for
manual/bootstrap workflows that write reviewed YAML artifacts into the repository, not for
normal application runtime.
"""

import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import yaml
import yfinance as yf
from tqdm import tqdm

YFINANCE_INFO_FIELDS = {
    "shortName": "name",
    "country": "country",
    "currency": "currency",
    "quoteType": "asset_type",
    "industry": "industry",
    "sector": "sector",
    "exchange": "exchange",
}


def create_ticker_universe_file(
    list_name: str,
    description: str,
    as_of_date: date | datetime,
    tickers: Sequence[str],
    output_path: str | Path,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.5,
) -> Path:
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
    output_path
        Explicit path to the YAML file to write.
    overwrite
        Whether to replace an existing file. Defaults to ``False`` to keep local/bootstrap
        generation safe by default.
    sleep_seconds
        Delay between yfinance requests. Defaults to ``0.5`` seconds.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    resolved_output_path = Path(output_path)
    _ensure_can_write(resolved_output_path, overwrite=overwrite)
    symbols = fetch_yfinance_ticker_metadata(tickers=tickers, sleep_seconds=sleep_seconds)
    document = build_ticker_universe_document(
        list_name=list_name,
        description=description,
        as_of_date=as_of_date,
        symbols=symbols,
    )
    return write_ticker_universe_yaml(
        data=document,
        output_path=resolved_output_path,
        overwrite=overwrite,
    )


def build_ticker_universe_document(
    *,
    list_name: str,
    description: str,
    as_of_date: date | datetime,
    symbols: Sequence[Mapping[str, object]],
    source: str = "yfinance",
) -> dict[str, object]:
    """Build a ticker universe document without fetching data or writing files."""
    return {
        "kind": "ticker_universe",
        "list_name": list_name,
        "description": description,
        "as_of_date": as_of_date,
        "source": source,
        "symbols": [dict(symbol) for symbol in symbols],
    }


def fetch_yfinance_ticker_metadata(
    tickers: Sequence[str],
    *,
    sleep_seconds: float = 0.5,
) -> list[dict[str, object]]:
    """Fetch yfinance metadata for ordered ticker symbols.

    Parameters
    ----------
    tickers
        Ordered ticker symbols to include in the universe.
    sleep_seconds
        Delay between yfinance requests to avoid sending requests too aggressively.

    Returns
    -------
    list[dict[str, object]]
        One metadata dictionary per ticker, preserving the input order.
    """
    symbols = []
    for ticker in tqdm(tickers):
        data = yf.Ticker(ticker)
        info = data.info
        symbol = {"ticker": ticker}
        for source_key, output_key in YFINANCE_INFO_FIELDS.items():
            symbol[output_key] = info.get(source_key)
        symbols.append(symbol)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return symbols


def write_ticker_universe_yaml(
    data: Mapping[str, object],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a ticker universe document to an explicit YAML path.

    Parameters
    ----------
    data
        Universe document to serialize.
    output_path
        Explicit path to the YAML file to write.
    overwrite
        Whether to replace an existing file. Defaults to ``False``.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    resolved_output_path = Path(output_path)
    _ensure_can_write(resolved_output_path, overwrite=overwrite)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    class IndentedDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow=flow, indentless=False)

    with resolved_output_path.open("w", encoding="utf-8") as file:
        yaml.dump(
            dict(data),
            file,
            Dumper=IndentedDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
    return resolved_output_path


def _ensure_can_write(output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        msg = f"Refusing to overwrite existing universe file: {output_path}"
        raise FileExistsError(msg)
