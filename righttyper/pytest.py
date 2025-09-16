import pytest

from righttyper.righttyper_runtime import should_skip_function
from righttyper.righttyper_utils import skip_this_file

pytest_modules: set[str] = set()
pytest_files : set[str] = set()

def pytest_collection_modifyitems(session, config, items):
    """Collects the names of pytest test modules."""

    for item in items:
        if (mod := getattr(item, "module", None)):
            pytest_modules.add(mod.__name__)
        if (path := getattr(item, "fspath", None)):
            pytest_files.add(path)

    # Clear caches, as the decision about files to skip
    # may depend upon the file names in pytest_files
    skip_this_file.cache_clear()
    should_skip_function.cache_clear()
