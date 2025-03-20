import pytest

@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_makereport(item, call):
    """Saves the test result for use within a fixture teardown."""
    report = yield
    item._report = report
    return report
