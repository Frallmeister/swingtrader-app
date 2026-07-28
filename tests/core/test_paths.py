from pathlib import Path

import pytest

from swingtrader.core.paths import find_repo_root


def test_find_repo_root_returns_directory_containing_marker(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert find_repo_root(start=nested) == tmp_path


def test_find_repo_root_raises_when_marker_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not locate repo root"):
        find_repo_root(start=tmp_path)
