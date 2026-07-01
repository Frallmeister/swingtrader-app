from datetime import date
from pathlib import Path

import pytest
import yaml

from swingtrader.data.ingestion import universe
from swingtrader.data.ingestion.universe_selection import resolve_active_tickers


def test_build_ticker_universe_document_preserves_metadata_and_order() -> None:
    document = universe.build_ticker_universe_document(
        list_name="se_large_cap",
        description="Swedish Large Cap available ticker universe",
        as_of_date=date(2026, 6, 23),
        symbols=[
            {"ticker": "VOLV-B.ST", "name": "Volvo B", "currency": "SEK"},
            {"ticker": "AAK.ST", "name": "AAK", "currency": "SEK"},
        ],
    )

    assert document == {
        "kind": "ticker_universe",
        "list_name": "se_large_cap",
        "description": "Swedish Large Cap available ticker universe",
        "as_of_date": date(2026, 6, 23),
        "source": "yfinance",
        "symbols": [
            {"ticker": "VOLV-B.ST", "name": "Volvo B", "currency": "SEK"},
            {"ticker": "AAK.ST", "name": "AAK", "currency": "SEK"},
        ],
    }


def test_write_ticker_universe_yaml_writes_resolver_compatible_file(tmp_path: Path) -> None:
    output_path = tmp_path / "universes" / "se_large_cap.yml"
    document = universe.build_ticker_universe_document(
        list_name="se_large_cap",
        description="Swedish Large Cap available ticker universe",
        as_of_date=date(2026, 6, 23),
        symbols=[
            {"ticker": "AAK.ST", "asset_type": "EQUITY"},
            {"ticker": "VOLV-B.ST", "asset_type": "EQUITY"},
        ],
    )

    written_path = universe.write_ticker_universe_yaml(data=document, output_path=output_path)

    loaded_document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert written_path == output_path
    assert loaded_document == document


def test_write_ticker_universe_yaml_does_not_overwrite_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "se_large_cap.yml"
    output_path.write_text("existing file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        universe.write_ticker_universe_yaml(data={}, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == "existing file"


def test_write_ticker_universe_yaml_can_overwrite_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "se_large_cap.yml"
    output_path.write_text("existing file", encoding="utf-8")
    document = universe.build_ticker_universe_document(
        list_name="se_large_cap",
        description="Swedish Large Cap available ticker universe",
        as_of_date=date(2026, 6, 23),
        symbols=[{"ticker": "AAK.ST"}],
    )

    universe.write_ticker_universe_yaml(data=document, output_path=output_path, overwrite=True)

    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == document


def test_create_ticker_universe_file_wires_fetch_build_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "se_large_cap.yml"
    calls = []

    def fake_fetch_yfinance_ticker_metadata(*, tickers, sleep_seconds):
        calls.append({"tickers": tickers, "sleep_seconds": sleep_seconds})
        return [
            {"ticker": "AAK.ST", "name": "AAK", "asset_type": "EQUITY"},
            {"ticker": "VOLV-B.ST", "name": "Volvo B", "asset_type": "EQUITY"},
        ]

    monkeypatch.setattr(
        universe,
        "fetch_yfinance_ticker_metadata",
        fake_fetch_yfinance_ticker_metadata,
    )

    written_path = universe.create_ticker_universe_file(
        list_name="se_large_cap",
        description="Swedish Large Cap available ticker universe",
        as_of_date=date(2026, 6, 23),
        tickers=["AAK.ST", "VOLV-B.ST"],
        output_path=output_path,
        sleep_seconds=0,
    )

    loaded_document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert written_path == output_path
    assert calls == [{"tickers": ["AAK.ST", "VOLV-B.ST"], "sleep_seconds": 0}]
    assert loaded_document["symbols"] == [
        {"ticker": "AAK.ST", "name": "AAK", "asset_type": "EQUITY"},
        {"ticker": "VOLV-B.ST", "name": "Volvo B", "asset_type": "EQUITY"},
    ]


def test_create_ticker_universe_file_refuses_existing_file_before_fetching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "se_large_cap.yml"
    output_path.write_text("existing file", encoding="utf-8")

    def fail_if_called(*, tickers, sleep_seconds):
        raise AssertionError("metadata should not be fetched before overwrite validation")

    monkeypatch.setattr(universe, "fetch_yfinance_ticker_metadata", fail_if_called)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        universe.create_ticker_universe_file(
            list_name="se_large_cap",
            description="Swedish Large Cap available ticker universe",
            as_of_date=date(2026, 6, 23),
            tickers=["AAK.ST"],
            output_path=output_path,
        )


def test_generated_universe_file_can_be_read_by_active_ticker_resolver(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "se_large_cap.yml"
    document = universe.build_ticker_universe_document(
        list_name="se_large_cap",
        description="Swedish Large Cap available ticker universe",
        as_of_date=date(2026, 6, 23),
        symbols=[
            {"ticker": "AAK.ST", "asset_type": "EQUITY"},
            {"ticker": "VOLV-B.ST", "asset_type": "EQUITY"},
        ],
    )
    universe.write_ticker_universe_yaml(data=document, output_path=output_path)
    (tmp_path / "active_tickers.yml").write_text(
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include: all
    exclude:
      - VOLV-B.ST
""".lstrip(),
        encoding="utf-8",
    )

    active_tickers = resolve_active_tickers(config_dir=tmp_path)

    assert active_tickers == ["AAK.ST"]
