"""
MODULE DOCSTRING
"""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import yaml


ConfigDir = Path | Traversable
LOADER = yaml.loader.SafeLoader


def resolve_active_tickers(config_dir: Path|None = None) -> list[str]:
    if config_dir is None:
        config_dir = files("swingtrader.configs.universes")
    yaml_files = _discover_yaml_files(config_dir=config_dir)
    available_universes = _parse_available_universes(yaml_files)
    active_tickers_data = _parse_active_tickers(yaml_files["active_tickers"])

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
                msg = f"Can't exclude ticker {ticker_to_exclude} since it's not in the available universe"
                raise UniverseConfigError(msg)
    return sorted(active_tickers_output)


def _discover_yaml_files(config_dir: ConfigDir) -> dict:
    """Find all yaml files and return them in a dict with filename-Path kwargs."""
    files = {p.stem: p for p in config_dir.glob("*.yml")}
    return files


def _parse_available_universes(files) -> dict:
    universes = {}
    for name, path in files.items():
        if name != "active_tickers":
            universes[name] = _get_universe_tickers(path)
    
    _validate_no_list_overlap([v for k, v in universes.items() if k != "active_tickers"])
    return universes


def _get_universe_tickers(path: Path) -> list:
    with path.open(encoding="utf-8") as fp:
        data = yaml.load(fp, Loader=LOADER)
    kind = data["kind"]
    symbols = data["symbols"]

    if kind != "ticker_universe":
        msg = f"Expected kind = 'ticker_universe' but got {kind}"
        raise UniverseConfigError(msg)
    list_of_tickers = [d["ticker"] for d in symbols]
    _verify_distinct_list(list_of_tickers)
    return list_of_tickers


def _parse_active_tickers(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fp:
        data = yaml.load(fp, Loader=LOADER)
    kind = data["kind"]
    if kind != "active_tickers":
        msg = f"Expected kind = 'active_tickers' but got {kind}"
        raise UniverseConfigError(msg)
    return data["universes"]


def _verify_distinct_list(values: list):
    if len(values) != len(set(values)):
        raise UniverseConfigError("The ticker list must contain only unique tickers.")



def _validate_no_list_overlap(lists: list):
    seen = set()
    for lst in lists:
        current = set(lst)
        if seen & current:
            raise UniverseConfigError("The universe files should not have overlaping tickers.")
        seen.update(current)


class UniverseConfigError(Exception):
    """Errors with the universe yaml files."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)



if __name__ == "__main__":
    out = resolve_active_tickers()


