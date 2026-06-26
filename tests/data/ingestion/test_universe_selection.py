from pathlib import Path

import pytest

from swingtrader.data.ingestion.universe_selection import (
    UniverseConfigError,
    resolve_active_tickers,
)


def write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def universe_config_dir(tmp_path: Path) -> Path:
    write_config(
        tmp_path / "se_large_cap.yml",
        """
kind: ticker_universe
list_name: se_large_cap
description: Test Swedish Large Cap universe
as_of_date: 2026-06-23
source: test
symbols:
  - ticker: AAK.ST
    asset_type: EQUITY
  - ticker: ALIF-B.ST
    asset_type: EQUITY
  - ticker: YUBICO.ST
    asset_type: EQUITY
""".lstrip(),
    )
    return tmp_path


def test_resolve_active_tickers_includes_all_and_excludes_tickers(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include: all
    exclude:
      - YUBICO.ST
""".lstrip(),
    )

    active_tickers = resolve_active_tickers(config_dir=universe_config_dir)

    assert active_tickers == ["AAK.ST", "ALIF-B.ST"]


def test_resolve_active_tickers_returns_sorted_order(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include:
      - YUBICO.ST
      - AAK.ST
""".lstrip(),
    )

    active_tickers = resolve_active_tickers(config_dir=universe_config_dir)

    assert active_tickers == ["AAK.ST", "YUBICO.ST"]


def test_resolve_active_tickers_loads_multiple_universe_files(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "se_mid_cap.yml",
        """
kind: ticker_universe
list_name: se_mid_cap
description: Test Swedish Mid Cap universe
as_of_date: 2026-06-23
source: test
symbols:
  - ticker: SBB-B.ST
    asset_type: EQUITY
  - ticker: NEOBO.ST
    asset_type: EQUITY
  - ticker: CIBUS.ST
    asset_type: EQUITY
""".lstrip(),
    )
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include:
      - YUBICO.ST
  - list_name: se_mid_cap
    include:
      - NEOBO.ST
      - CIBUS.ST
""".lstrip(),
    )

    active_tickers = resolve_active_tickers(config_dir=universe_config_dir)

    assert active_tickers == ["CIBUS.ST", "NEOBO.ST", "YUBICO.ST"]


def test_resolve_active_tickers_ignores_unreferenced_yaml(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include: all
""".lstrip(),
    )
    write_config(
        universe_config_dir / "unused.yml",
        """
kind: not_a_ticker_universe
symbols: []
""".lstrip(),
    )

    active_tickers = resolve_active_tickers(config_dir=universe_config_dir)

    assert active_tickers == ["AAK.ST", "ALIF-B.ST", "YUBICO.ST"]


def test_resolve_active_tickers_rejects_unknown_included_ticker(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include:
      - MISSING.ST
""".lstrip(),
    )

    with pytest.raises(UniverseConfigError, match="MISSING.ST"):
        resolve_active_tickers(config_dir=universe_config_dir)


def test_resolve_active_tickers_rejects_unknown_excluded_ticker(
    universe_config_dir: Path,
) -> None:
    write_config(
        universe_config_dir / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include: all
    exclude:
      - MISSING.ST
""".lstrip(),
    )

    with pytest.raises(UniverseConfigError, match="MISSING.ST"):
        resolve_active_tickers(config_dir=universe_config_dir)
