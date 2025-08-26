import pytest

pytest_modules: set[str] = set()

def pytest_collection_modifyitems(session, config, items):
    """Collects the names of pytest test modules."""

    for item in items:
        if (mod := getattr(item, "module", None)):
            pytest_modules.add(mod.__name__)
