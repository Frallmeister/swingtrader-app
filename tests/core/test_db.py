import ast
from pathlib import Path

from sqlalchemy import inspect

from swingtrader.core.config import DATABASE_URL_ENV_VAR, get_database_url
from swingtrader.core.db import create_database_engine, resolve_database_engine


def test_get_database_url_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, "sqlite+pysqlite:///from-env.sqlite")

    database_url = get_database_url("sqlite+pysqlite:///explicit.sqlite")

    assert database_url == "sqlite+pysqlite:///explicit.sqlite"


def test_get_database_url_reads_environment_override(monkeypatch) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, "sqlite+pysqlite:///from-env.sqlite")

    database_url = get_database_url()

    assert database_url == "sqlite+pysqlite:///from-env.sqlite"


def test_create_database_engine_creates_sqlite_parent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "swingtrader.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"

    engine = create_database_engine(database_url)

    assert database_path.parent.is_dir()
    assert inspect(engine).get_table_names() == []


def test_resolve_database_engine_rejects_engine_and_database_url(tmp_path: Path) -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'swingtrader.sqlite').as_posix()}"

    try:
        resolve_database_engine(database_url=database_url, engine=engine)
    except ValueError as error:
        assert str(error) == "Pass either engine or database_url, not both."
    else:
        raise AssertionError("Expected ValueError for duplicate engine configuration")


def test_core_package_does_not_import_data_package() -> None:
    core_package = Path(__file__).parents[2] / "src" / "swingtrader" / "core"
    violations: list[str] = []

    for module_path in sorted(core_package.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = [node.module]
            else:
                continue

            for imported_module in imported_modules:
                if imported_module == "swingtrader.data" or imported_module.startswith(
                    "swingtrader.data."
                ):
                    relative_path = module_path.relative_to(core_package.parents[2])
                    violations.append(f"{relative_path}: {imported_module}")

    assert violations == []
