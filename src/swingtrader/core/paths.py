"""Filesystem path helpers for resolving repository locations."""

from pathlib import Path

REPO_ROOT_MARKER = "pyproject.toml"


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up until a marker file is found.

    Parameters
    ----------
    start
        Directory to begin the search from. Defaults to the current working directory.
        This is useful because interactive contexts (for example, notebooks) may run with a
        working directory that differs from the repository root.

    Returns
    -------
    Path
        The first ancestor directory (including ``start``) that contains
        ``pyproject.toml``.

    Raises
    ------
    RuntimeError
        If no ancestor directory contains ``pyproject.toml``.
    """
    current = (start or Path.cwd()).resolve()
    while not (current / REPO_ROOT_MARKER).exists():
        if current == current.parent:
            msg = f"Could not locate repo root (no {REPO_ROOT_MARKER} found)."
            raise RuntimeError(msg)
        current = current.parent
    return current
