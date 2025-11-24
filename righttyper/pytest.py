import pytest
from righttyper.righttyper_utils import set_test_files_and_modules

def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Collects the names of pytest test modules."""

    files: set[str] = set()
    modules: set[str] = set()

    for item in items:
        if (mod := getattr(item, "module", None)):
            modules.add(mod.__name__)
        if (path := getattr(item, "fspath", None)):
            files.add(path)

    set_test_files_and_modules(files, modules)
