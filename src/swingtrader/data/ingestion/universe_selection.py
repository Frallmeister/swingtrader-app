"""Resolve the active ticker universe from packaged YAML configuration.

This module reads ``active_tickers.yml`` and the referenced available universe files,
then returns the sorted set of tickers that should be ingested. It validates that
included and excluded tickers exist in their configured universe and that referenced
universe files do not overlap.
"""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import yaml

ConfigDir = Path | Traversable
ConfigFile = Path | Traversable


def resolve_active_tickers(config_dir: ConfigDir | None = None) -> list[str]:
    if config_dir is None:
        config_dir = files("swingtrader.configs.universes")
    active_tickers_data = _parse_active_tickers(
        _get_config_file(config_dir=config_dir, list_name="active_tickers")
    )
    available_universes = _parse_available_universes(
        config_dir=config_dir,
        active_tickers_data=active_tickers_data,
    )

    active_tickers_output = []
    for active_ticker_dict in active_tickers_data:
        list_name = active_ticker_dict["list_name"]
        include = active_ticker_dict["include"]
        exclude = active_ticker_dict.get("exclude", [])
        if isinstance(include, str) and (include == "all"):
            active_tickers_output.extend(available_universes[list_name])
        elif isinstance(include, list):
            for ticker in include:
                if ticker in available_universes[list_name]:
                    active_tickers_output.append(ticker)
                else:
                    msg = f"Ticker '{ticker}' in active tickers but not in {list_name} list."
                    raise UniverseConfigError(msg)
        for ticker_to_exclude in exclude:
            if ticker_to_exclude in active_tickers_output:
                active_tickers_output.remove(ticker_to_exclude)
            else:
                msg = (
                    f"Can't exclude ticker {ticker_to_exclude} "
                    "since it's not in the available universe"
                )
                raise UniverseConfigError(msg)
    return sorted(active_tickers_output)


def _get_config_file(config_dir: ConfigDir, list_name: str) -> ConfigFile:
    path = config_dir.joinpath(f"{list_name}.yml")
    if not path.is_file():
        msg = f"Missing universe config file: {list_name}.yml"
        raise UniverseConfigError(msg)
    return path


def _parse_available_universes(
    config_dir: ConfigDir,
    active_tickers_data: list[dict],
) -> dict:
    universes = {}
    for active_ticker_dict in active_tickers_data:
        list_name = active_ticker_dict["list_name"]
        if list_name not in universes:
            universes[list_name] = _get_universe_tickers(
                _get_config_file(config_dir=config_dir, list_name=list_name)
            )

    _validate_no_list_overlap(list(universes.values()))
    return universes


def _get_universe_tickers(path: ConfigFile) -> list:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kind = data["kind"]
    symbols = data["symbols"]

    if kind != "ticker_universe":
        msg = f"Expected kind = 'ticker_universe' but got {kind}"
        raise UniverseConfigError(msg)
    list_of_tickers = [d["ticker"] for d in symbols]
    _verify_distinct_list(list_of_tickers)
    return list_of_tickers


def _parse_active_tickers(path: ConfigFile) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kind = data["kind"]
    if kind != "active_tickers":
        msg = f"Expected kind = 'active_tickers' but got {kind}"
        raise UniverseConfigError(msg)
    return data["universes"]


def _verify_distinct_list(values: list):
    if len(values) != len(set(values)):
        raise UniverseConfigError("The ticker list must contain only unique tickers.")


def _validate_no_list_overlap(lists: list):
    seen: set[str] = set()
    for lst in lists:
        current = set(lst)
        if seen & current:
            raise UniverseConfigError("The universe files should not have overlapping tickers.")
        seen.update(current)


class UniverseConfigError(Exception):
    """Errors with the universe yaml files."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
